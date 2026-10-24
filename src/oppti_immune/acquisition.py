"""
acquisition.py — Download all datasets required by the pipeline.

Datasets
--------
- CPTAC proteomics (13 cancer types) via the `cptac` Python package
- Immunotherapy cohorts from GEO (via GEOquery in R subprocess) and
  Bioconductor (IMvigor210CoreBiologies)
"""

from __future__ import annotations

import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import get_logger, load_config, data_path, tag_samples

logger = get_logger(__name__)


# ── CPTAC ─────────────────────────────────────────────────────────────────────

def download_cptac(cfg: dict) -> dict[str, dict[str, pd.DataFrame]]:
    """Download all CPTAC datasets defined in config.

    Returns a nested dict:
        { cancer_type: { 'proteomics': df, 'transcriptomics': df,
                         'phosphoproteomics': df | None,
                         'clinical': df, 'cnv': df | None } }
    """
    try:
        import cptac
    except ImportError:
        raise ImportError("Install cptac: pip install cptac")

    cancer_types = cfg["cptac"]["cancer_types"]
    out_dir = data_path(cfg, "cptac")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, dict[str, pd.DataFrame]] = {}

    for ct in cancer_types:
        logger.info(f"Downloading CPTAC {ct.upper()}...")
        try:
            ds_cls = getattr(cptac, ct.capitalize())
            ds = ds_cls()   # downloads on demand (cptac>=2.0, Zenodo hosted)
        except Exception as e:
            logger.warning(f"Could not load {ct}: {e}")
            continue
        result: dict[str, pd.DataFrame | None] = {}

        # NOTE: cptac >=2.0 moved the download API (hosted on Zenodo) and
        # requires explicit `source=`; get_dataframe resolves available sources
        # only after files are downloaded. proteomics=bcm, transcriptomics=broad,
        # clinical=mssm, cnv=bcm are the canonical Tumor cohorts.
        SOURCE_MAP = {"proteomics": "bcm", "transcriptomics": "bcm",
                      "clinical": "mssm", "cnv": "bcm"}

        def _fetch(kind: str, source: str, flatten_multiindex: bool) -> pd.DataFrame | None:
            try:
                getter = {
                    "proteomics": ds.get_proteomics,
                    "transcriptomics": ds.get_transcriptomics,
                    "clinical": ds.get_clinical,
                    "cnv": ds.get_CNV,
                }[kind]
                df = getter(source=source)
                if isinstance(df.columns, pd.MultiIndex):
                    if flatten_multiindex:
                        df.columns = df.columns.get_level_values(0)
                    else:
                        df.columns = [
                            f"{g}_{s}" for g, s in zip(
                                df.columns.get_level_values(0),
                                df.columns.get_level_values(1),
                            )
                        ]
                if df.columns.has_duplicates:
                    df = df.T.groupby(level=0).mean().T
                return df
            except Exception as e:
                logger.warning(f"{ct} {kind} failed: {type(e).__name__}: {e}")
                return None

        # Proteomics
        try:
            prot = _fetch("proteomics", SOURCE_MAP["proteomics"], flatten_multiindex=True)
            if prot is not None:
                prot = tag_samples(prot, ct)
                result["proteomics"] = prot
                prot.to_parquet(out_dir / f"{ct}_proteomics.parquet")
        except Exception as e:
            logger.warning(f"{ct} proteomics save failed: {type(e).__name__}: {e}")

        # Transcriptomics
        try:
            rna = _fetch("transcriptomics", SOURCE_MAP["transcriptomics"], flatten_multiindex=True)
            if rna is not None:
                rna = tag_samples(rna, ct)
                result["transcriptomics"] = rna
                rna.to_parquet(out_dir / f"{ct}_transcriptomics.parquet")
        except Exception as e:
            logger.warning(f"{ct} transcriptomics save failed: {type(e).__name__}: {e}")

        # Clinical
        try:
            clin = _fetch("clinical", SOURCE_MAP["clinical"], flatten_multiindex=False)
            if clin is not None:
                clin = tag_samples(clin, ct)
                result["clinical"] = clin
                clin.to_csv(out_dir / f"{ct}_clinical.csv")
        except Exception as e:
            logger.warning(f"{ct} clinical save failed: {type(e).__name__}: {e}")

        # CNV
        try:
            cnv = _fetch("cnv", SOURCE_MAP["cnv"], flatten_multiindex=True)
            if cnv is not None:
                cnv = tag_samples(cnv, ct)
                result["cnv"] = cnv
                cnv.to_parquet(out_dir / f"{ct}_cnv.parquet")
        except Exception as e:
            logger.warning(f"{ct} cnv save failed: {type(e).__name__}: {e}")

        datasets[ct] = result
        logger.info(
            f"  {ct.upper()}: {len(result.get('proteomics', pd.DataFrame()))} samples, "
            f"proteome={'✓' if result.get('proteomics') is not None else '✗'}, "
            f"RNA={'✓' if result.get('transcriptomics') is not None else '✗'}, "
            f"clinical={'✓' if result.get('clinical') is not None else '✗'}"
        )

    logger.info(f"Downloaded {len(datasets)} CPTAC cancer types.")
    return datasets


# ── Immunotherapy cohorts (GEO via R) ─────────────────────────────────────────

GEO_R_TEMPLATE = """
suppressPackageStartupMessages({{
  library(GEOquery)
  library(Biobase)
}})
cat("Downloading {geo_id}...\\n")
tryCatch({{
  gse <- getGEO("{geo_id}", GSEMatrix=TRUE, AnnotGPL=FALSE)
  eset <- gse[[1]]
  expr  <- as.data.frame(exprs(eset))
  pdata <- pData(eset)
  write.csv(expr,  "{expr_out}")
  write.csv(pdata, "{clin_out}")
  cat("Done: {geo_id}\\n")
}}, error=function(e) {{
  cat("ERROR downloading {geo_id}:", conditionMessage(e), "\\n")
}})
"""

IMVIGOR_R_TEMPLATE = """
suppressPackageStartupMessages({{
  library(IMvigor210CoreBiologies)
  library(Biobase)
}})
data(cds)
expr  <- as.data.frame(counts(cds))
pdata <- pData(cds)
write.csv(expr,  "{expr_out}")
write.csv(pdata, "{clin_out}")
cat("IMvigor210 downloaded.\\n")
"""


def download_immunotherapy_cohorts(cfg: dict) -> None:
    """Download all immunotherapy validation cohorts.

    GEO cohorts are downloaded via a transient Rscript call.
    IMvigor210 uses the Bioconductor package.
    """
    out_dir = data_path(cfg, "immunotherapy")
    out_dir.mkdir(parents=True, exist_ok=True)

    cohorts = cfg.get("immunotherapy_cohorts", {})

    for name, cohort_cfg in cohorts.items():
        source = cohort_cfg.get("source", "geo")
        expr_out = str(out_dir / f"{name}_expr.csv")
        clin_out = str(out_dir / f"{name}_clinical.csv")

        if Path(expr_out).exists() and Path(clin_out).exists():
            logger.info(f"{name}: already downloaded, skipping.")
            continue

        logger.info(f"Downloading {name} ({source})...")

        if source == "geo":
            geo_id = cohort_cfg["geo_id"]
            r_script = GEO_R_TEMPLATE.format(
                geo_id=geo_id, expr_out=expr_out, clin_out=clin_out
            )
        elif source == "bioconductor" and cohort_cfg.get("package") == "IMvigor210CoreBiologies":
            r_script = IMVIGOR_R_TEMPLATE.format(
                expr_out=expr_out, clin_out=clin_out
            )
        else:
            logger.warning(f"Unknown source '{source}' for {name}, skipping.")
            continue

        try:
            result = subprocess.run(
                ["Rscript", "-e", r_script],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"{name} R download failed:\n{result.stderr}")
            else:
                logger.info(f"{name}: downloaded successfully.")
        except subprocess.TimeoutExpired:
            logger.error(f"{name}: download timed out after 300 s.")
        except FileNotFoundError:
            logger.error("Rscript not found. Install R and ensure it is in PATH.")


def run_all(cfg: dict) -> None:
    """Download all datasets: CPTAC + immunotherapy cohorts."""
    logger.info("=== Step 1: Data Acquisition ===")
    download_cptac(cfg)
    download_immunotherapy_cohorts(cfg)
    logger.info("=== Data Acquisition Complete ===")


if __name__ == "__main__":
    cfg = load_config()
    run_all(cfg)

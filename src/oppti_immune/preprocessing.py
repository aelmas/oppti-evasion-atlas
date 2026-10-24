"""
preprocessing.py — Normalization, QC, filtering, and pan-cancer matrix assembly.

Steps
-----
1. Load per-cancer CPTAC parquet files
2. Filter tumor samples (drop adjacent normals)
3. Filter proteins with >threshold% missing values
4. MAD normalization (matching the original OPPTI paper exactly)
5. Subset to the Immune Evasion Proteome (IEP) gene panel
6. Concatenate into pan-cancer matrices (proteomics + transcriptomics)
7. Save pan-cancer matrices and per-cancer summary stats
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    get_logger,
    load_config,
    load_gene_panel,
    data_path,
    results_path,
    save_parquet,
    save_csv,
    tag_samples,
)

logger = get_logger(__name__)

# Patterns that mark adjacent-normal samples in CPTAC indices
# (`\.N` occurs mid-index after tag_samples appends e.g. "_luad")
NORMAL_PATTERNS = [r"\.N$", r"\.N(?:[._+]|$)"]   # non-capturing groups: avoids pandas regex warnings in str.contains


# ── Core normalization ────────────────────────────────────────────────────────

def mad_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Median-Absolute-Deviation normalization (protein-wise, across samples).

    Each protein is centered on its median and scaled by its MAD.
    Proteins with zero MAD are scaled by the minimum non-zero MAD.

    Parameters
    ----------
    df : samples × proteins DataFrame (numeric, may contain NaN)

    Returns
    -------
    Normalized DataFrame of same shape.
    """
    median = df.median(axis=0)
    mad    = (df - median).abs().median(axis=0)

    # Replace zero MAD to avoid division by zero
    min_nonzero = mad[mad > 0].min()
    if np.isnan(min_nonzero):
        min_nonzero = 1.0
    mad = mad.where(mad > 0, min_nonzero)

    return (df - median) / mad


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalization (mean=0, std=1 per protein)."""
    mean = df.mean(axis=0)
    std  = df.std(axis=0).replace(0, 1.0)
    return (df - mean) / std


def apply_normalization(df: pd.DataFrame, method: str = "mad") -> pd.DataFrame:
    """Dispatch to the requested normalization method."""
    method = method.lower()
    if method == "mad":
        return mad_normalize(df)
    elif method == "zscore":
        return zscore_normalize(df)
    elif method == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        qt   = QuantileTransformer(output_distribution="normal", random_state=42)
        vals = qt.fit_transform(df.fillna(df.median()))
        return pd.DataFrame(vals, index=df.index, columns=df.columns)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_tumor_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Remove adjacent-normal samples from the sample index."""
    import re
    pattern = "|".join(NORMAL_PATTERNS)
    mask    = ~df.index.astype(str).str.contains(pattern, regex=True)
    dropped = (~mask).sum()
    if dropped:
        logger.debug(f"  Dropped {dropped} normal/adjacent samples.")
    return df[mask]


def extract_gene_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Strip 'GENE_ENSPxxx' or 'GENE_ENSGxxx' → 'GENE' column names.

    CPTAC proteomics tables use `GENE_ENSPxxx.N` naming and transcriptomics
    tables use `GENE_ENSGxxx.version` naming. This function extracts the
    leading gene symbol and collapses duplicate columns by taking the mean.
    """
    cols = df.columns.astype(str)
    has_ensp = cols.str.contains("_ENSP").any()
    has_ensg = cols.str.contains("_ENSG").any()
    if not has_ensp and not has_ensg:
        return df
    new_cols = cols.str.split("_", n=1).str[0]
    df = df.copy()
    df.columns = new_cols
    # Collapse duplicate gene columns by mean
    if df.columns.has_duplicates:
        df = df.T.groupby(level=0).mean().T
    return df


def filter_missing_proteins(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """Remove proteins with > threshold fraction of missing values."""
    frac_missing = df.isnull().mean(axis=0)
    keep = frac_missing < threshold
    removed = (~keep).sum()
    if removed:
        logger.debug(f"  Removed {removed} proteins exceeding {threshold:.0%} missingness.")
    return df.loc[:, keep]


def subset_to_panel(df: pd.DataFrame, panel_genes: list[str]) -> pd.DataFrame:
    """Keep only columns matching the IEP gene panel (intersection)."""
    available = [g for g in panel_genes if g in df.columns]
    missing   = len(panel_genes) - len(available)
    if missing:
        logger.debug(f"  {missing}/{len(panel_genes)} panel genes not found in data.")
    return df[available]


# ── Per-cancer preprocessing ──────────────────────────────────────────────────

def preprocess_cancer(
    cancer: str,
    cfg: dict,
    panel_genes: list[str],
) -> dict[str, pd.DataFrame | None]:
    """Load, filter, normalize, and subset one cancer type.

    Returns
    -------
    dict with keys: 'prot_full', 'prot_iep', 'rna_full', 'rna_iep'
    """
    cptac_dir = data_path(cfg, "cptac")
    norm_method  = cfg["preprocessing"]["normalization"]
    miss_thresh  = cfg["preprocessing"]["missing_threshold"]
    min_samples  = cfg["preprocessing"]["min_samples_per_type"]

    result: dict[str, pd.DataFrame | None] = {
        "prot_full": None, "prot_iep": None,
        "rna_full":  None, "rna_iep":  None,
    }

    # ── Proteomics ────────────────────────────────────────────────────────────
    prot_file = cptac_dir / f"{cancer}_proteomics.parquet"
    if prot_file.exists():
        prot = pd.read_parquet(prot_file)
        prot = extract_gene_prefix(prot)
        prot = filter_tumor_samples(prot)

        if len(prot) < min_samples:
            logger.warning(f"{cancer}: only {len(prot)} tumor samples — skipping.")
            return result

        prot = filter_missing_proteins(prot, miss_thresh)
        prot_norm = apply_normalization(prot, norm_method)
        prot_norm["cancer_type"] = cancer

        result["prot_full"] = prot_norm
        result["prot_iep"]  = subset_to_panel(
            prot_norm.drop(columns="cancer_type"), panel_genes
        ).assign(cancer_type=cancer)

        logger.info(
            f"{cancer.upper()}: {len(prot)} tumors, "
            f"{prot_norm.shape[1]-1} proteins after QC, "
            f"{result['prot_iep'].shape[1]-1} IEP proteins"
        )
    else:
        logger.warning(f"{cancer}: proteomics file not found at {prot_file}")

    # ── Transcriptomics ───────────────────────────────────────────────────────
    rna_file = cptac_dir / f"{cancer}_transcriptomics.parquet"
    if rna_file.exists():
        rna = pd.read_parquet(rna_file)
        rna = extract_gene_prefix(rna)
        rna = filter_tumor_samples(rna)

        # IEP subset (normalize only 164 cols — fast)
        iep_cols = [g for g in panel_genes if g in rna.columns]
        if iep_cols:
            rna_iep = apply_normalization(rna[iep_cols], norm_method)
            result["rna_iep"] = rna_iep
            logger.info(
                f"  RNA IEP: {rna_iep.shape[0]} samples × "
                f"{rna_iep.shape[1]} IEP genes"
            )

        # Full transcriptome: keep as raw (or lightly normalized) for imputation.
        # Skip MAD on 60k columns — too slow. Z-score is cheap and vectorized.
        result["rna_full"] = ((rna - rna.mean()) / rna.std().replace(0, 1)).fillna(0)

    return result


# ── Pan-cancer assembly ───────────────────────────────────────────────────────

def build_pan_cancer_matrices(cfg: dict) -> dict[str, pd.DataFrame]:
    """Process all cancer types and concatenate into pan-cancer matrices.

    Returns
    -------
    dict with keys: 'pan_prot_iep', 'pan_rna_iep', 'pan_prot_full', 'pan_rna_full'
    """
    panel = load_gene_panel(cfg)
    panel_genes = panel["gene"].tolist()

    cancer_types = cfg["cptac"]["cancer_types"]
    all_prot_iep:  list[pd.DataFrame] = []
    all_rna_iep:   list[pd.DataFrame] = []
    all_prot_full: list[pd.DataFrame] = []
    all_rna_full:  list[pd.DataFrame] = []

    for ct in cancer_types:
        res = preprocess_cancer(ct, cfg, panel_genes)
        if res["prot_iep"] is not None:
            all_prot_iep.append(res["prot_iep"])
        if res["rna_iep"] is not None:
            all_rna_iep.append(res["rna_iep"])
        if res["prot_full"] is not None:
            all_prot_full.append(res["prot_full"])
        if res["rna_full"] is not None:
            all_rna_full.append(res["rna_full"])

    matrices = {}

    if all_prot_iep:
        matrices["pan_prot_iep"] = pd.concat(all_prot_iep, axis=0, sort=False)
    if all_rna_iep:
        matrices["pan_rna_iep"]  = pd.concat(all_rna_iep,  axis=0, sort=False)
    if all_prot_full:
        matrices["pan_prot_full"] = pd.concat(all_prot_full, axis=0, sort=False)
    if all_rna_full:
        matrices["pan_rna_full"]  = pd.concat(all_rna_full,  axis=0, sort=False)

    # Save
    proc_dir = data_path(cfg, "processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    for key, df in matrices.items():
        out = proc_dir / f"{key}.parquet"
        # Write sample ID as both index and explicit column so R can read cleanly
        df_out = df.copy()
        df_out.index.name = "sample_id"
        df_out.reset_index(drop=False, inplace=True)
        df_out.to_parquet(out, index=False)
        logger.info(f"Saved {key}: {df.shape} → {out}")

    # Summary stats
    if "pan_prot_iep" in matrices:
        summary = (
            matrices["pan_prot_iep"]
            .groupby("cancer_type")
            .apply(lambda g: pd.Series({
                "n_samples":  len(g),
                "n_proteins": g.drop(columns="cancer_type", errors="ignore").notna().any().sum(),
            }))
        )
        save_csv(summary, results_path(cfg, "preprocessing_summary.csv"))
        logger.info(f"\nPan-cancer IEP summary:\n{summary.to_string()}")

    return matrices


# ── Master entry point ────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, pd.DataFrame]:
    """Run the full preprocessing pipeline."""
    logger.info("=== Step 2: Preprocessing & Normalization ===")
    matrices = build_pan_cancer_matrices(cfg)

    total = sum(
        len(v) for k, v in matrices.items() if "full" not in k and "prot" in k
    )
    logger.info(
        f"=== Preprocessing Complete: {total} total tumor samples, "
        f"{matrices.get('pan_prot_iep', pd.DataFrame()).shape[1]-1} IEP proteins ==="
    )
    return matrices


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)

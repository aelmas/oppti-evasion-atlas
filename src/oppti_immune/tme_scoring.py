"""
tme_scoring.py — Immune cell abundance estimation from MAD-normalized RNA.

Uses mean Z-score aggregation of curated immune gene signatures
(analogous to ssGSEA but simpler and appropriate for pre-normalized data).

Signature sources:
  - Bindea et al. 2013 (Cancer Immunity)
  - Rooney et al. 2015 (Cell) — cytolytic activity
  - Tirosh et al. 2016 (Science) — exhaustion
  - TIMER 2.0 marker genes
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from .utils import get_logger

logger = get_logger(__name__)

# ── Immune gene signatures ────────────────────────────────────────────────────
# Mean Z-score of these genes (in MAD-normalized RNA space) estimates abundance.

IMMUNE_SIGNATURES: dict[str, list[str]] = {
    "CD8+ T-cells": [
        "CD8A", "CD8B", "GZMA", "GZMB", "GZMH", "GZMK", "PRF1",
        "IFNG", "CXCR3", "TBX21",
    ],
    "CD4+ T-cells": [
        "CD4", "IL2RA", "MAL", "PTPRC", "TRAC", "TRBC1",
    ],
    "Regulatory T-cells": [
        "FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "ENTPD1",
    ],
    "Cytolytic activity": [
        "GZMA", "PRF1",   # Rooney et al. 2015
    ],
    "B-cells": [
        "CD19", "MS4A1", "CD79A", "CD79B", "BLNK", "BANK1", "PAX5",
    ],
    "NK cells": [
        "NCAM1", "KLRB1", "KLRD1", "NKG7", "GNLY", "FCGR3A",
        "FGFBP2", "S1PR5",
    ],
    "Macrophages M1": [
        "CD68", "NOS2", "TNF", "IL6", "IL12B", "CXCL10", "FCGR1A",
    ],
    "Macrophages M2": [
        "CD163", "MRC1", "CLEC4M", "TGFB1", "IL10", "VSIG4", "FOLR2",
    ],
    "Dendritic cells": [
        "ITGAX", "CD86", "CLEC10A", "FCER1A", "CLEC9A", "XCR1",
    ],
    "Cancer associated fibroblasts": [
        "FAP", "ACTA2", "FN1", "COL1A1", "COL1A2", "POSTN", "PDGFRB",
    ],
    "Endothelial cells": [
        "PECAM1", "VWF", "ENG", "CLDN5", "KDR", "ROBO4",
    ],
    "T-cell exhaustion": [
        "HAVCR2", "TIGIT", "LAG3", "PDCD1", "CTLA4", "TOX",
        "ENTPD1", "NR4A1",
    ],
    "Tumor inflammation signature": [
        # Banchereau/Ayers TIS (J Clin Oncol 2017, 8-gene signature)
        "CD8A", "GZMB", "CXCL9", "CXCR6", "CD274", "PDCD1LG2",
        "CD276", "LAG3",
    ],
}


def compute_signature_scores(
    rna: pd.DataFrame,          # samples × genes (MAD-normalized)
    signatures: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Mean Z-score aggregation of immune gene signatures.

    Parameters
    ----------
    rna        : samples × genes, already MAD-normalized (mean ≈ 0, sd ≈ 1)
    signatures : gene-set dictionary; defaults to IMMUNE_SIGNATURES

    Returns
    -------
    DataFrame samples × cell-type-names (mean Z-scores)
    """
    if signatures is None:
        signatures = IMMUNE_SIGNATURES

    scores = {}
    for cell_type, genes in signatures.items():
        available = [g for g in genes if g in rna.columns]
        if len(available) < 2:
            logger.warning(
                f"  {cell_type}: only {len(available)}/{len(genes)} signature "
                f"genes found — skipping"
            )
            continue
        if len(available) < len(genes):
            missing = set(genes) - set(available)
            logger.debug(f"  {cell_type}: missing {missing}")

        scores[cell_type] = rna[available].mean(axis=1)

    df = pd.DataFrame(scores, index=rna.index)
    logger.info(
        f"Computed {df.shape[1]} immune signature scores "
        f"across {df.shape[0]} samples."
    )
    return df


# ── OPPTI × TME correlation ───────────────────────────────────────────────────

def correlate_outliers_with_tme(
    outlier_binary: pd.DataFrame,   # samples × IEP proteins (0/1)
    tme_scores:     pd.DataFrame,   # samples × immune metrics
    min_samples:    int = 20,
) -> pd.DataFrame:
    """Spearman correlation of each IEP outlier protein vs. each immune metric.

    Only uses samples present in both matrices.

    Returns
    -------
    DataFrame: gene, immune_metric, rho, pvalue, padj, n_samples
    """
    shared = outlier_binary.index.intersection(tme_scores.index)
    logger.info(
        f"Correlating {len(outlier_binary.columns)} IEP proteins × "
        f"{len(tme_scores.columns)} immune metrics "
        f"across {len(shared)} shared samples."
    )

    ob = outlier_binary.loc[shared]
    tm = tme_scores.loc[shared]

    records: list[dict] = []
    for gene in ob.columns:
        g_vals = ob[gene].values.astype(float)
        for metric in tm.columns:
            m_vals = tm[metric].values
            ok = ~np.isnan(g_vals) & ~np.isnan(m_vals)
            if ok.sum() < min_samples:
                continue
            rho, pval = spearmanr(g_vals[ok], m_vals[ok])
            records.append({
                "gene":          gene,
                "immune_metric": metric,
                "rho":           float(rho),
                "pvalue":        float(pval),
                "n_samples":     int(ok.sum()),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        _, df["padj"], _, _ = multipletests(df["pvalue"], method="fdr_bh")
        df = df.sort_values("pvalue")

    return df


# ── Master entry point ────────────────────────────────────────────────────────

def run(cfg: dict, rna: pd.DataFrame, outlier_binary: pd.DataFrame) -> pd.DataFrame:
    """Compute TME scores and correlate with OPPTI outliers.

    Returns the correlation DataFrame (also saved to disk by caller).
    """
    logger.info("=== Step 5: TME Immune Signature Scoring ===")

    # Remove cancer_type column if present
    rna_vals = rna.drop(columns=["cancer_type"], errors="ignore")

    tme_scores = compute_signature_scores(rna_vals)

    # Save TME scores
    from .utils import results_path, save_csv
    save_csv(
        tme_scores.reset_index(names="sample"),
        results_path(cfg, "aim3_tme_signature_scores.csv"),
        index=False,
    )
    logger.info(
        f"Saved TME signature scores → aim3_tme_signature_scores.csv"
    )

    # Correlate
    corr_df = correlate_outliers_with_tme(outlier_binary, tme_scores)
    save_csv(corr_df, results_path(cfg, "aim3_protein_immune_correlation.csv"), index=False)

    n_sig = (corr_df["padj"] < 0.05).sum() if not corr_df.empty else 0
    logger.info(
        f"Saved {len(corr_df)} correlations "
        f"({n_sig} FDR<5%) → aim3_protein_immune_correlation.csv"
    )

    if not corr_df.empty:
        logger.info("\nTop 10 significant correlations:")
        top = corr_df[corr_df["padj"] < 0.05].head(10)
        if not top.empty:
            logger.info(
                top[["gene", "immune_metric", "rho", "padj"]].to_string(index=False)
            )

    return corr_df

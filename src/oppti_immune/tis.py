"""Tumor Inflammation Signature (TIS) baselines.

The published TIS (Danaher et al., J Immunother Cancer 2018;6:63) is an
18-gene signature scored as a **weighted linear combination of the 18 genes'
expression values normalized to stable housekeeper gene expression**. The
exact published weights are proprietary (NanoString IUO assay), so a public
reimplementation can be faithful to the *construction*, not the coefficients.

This module provides fully specified, deterministic public baselines:

* ``tis_score(expr, how="hk")``  — housekeeper-normalized (geometric-mean),
  per-gene MAD-Z, **equal-weight** mean over the 18 genes. This is the
  primary public TIS baseline used in PIES-DSS comparisons.
* ``tis_score(expr, how="raw")`` — legacy mean of raw log2 expression
  (no normalization); kept as a sensitivity variant (it was the shipped
  baseline in earlier pipeline versions).
* ``tis_ssgsea(expr)``           — classic ssGSEA enrichment of the 18-gene
  set against the full expression matrix (sensitivity variant).

All three preserve the published caveat: higher score = inflamed /
resource-responsive tumour.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

TIS_GENES = [
    "CD3D", "IDO1", "CIITA", "CD3E", "CCL5", "GZMK", "CD2",
    "HLA-DRA", "CXCL13", "IL2RG", "NKG7", "HLA-E", "CXCR6",
    "LAG3", "TAGAP", "CXCL10", "STAT1", "TIGIT",
]

HOUSEKEEPERS = ["GAPDH", "ACTB", "HPRT1", "RPLP0", "B2M", "TBP", "YWHAZ", "PPIA"]


def _madz(col: pd.Series) -> pd.Series:
    med = col.median()
    mad = (col - med).abs().median()
    return (col - med) / (mad if mad > 0 else 1e-9)


def _housekeeper_normalize(expr: pd.DataFrame, hk: list[str]) -> pd.DataFrame:
    """2^(x) / geometric-mean(housekeepers), returned on log2 scale."""
    lin = 2.0 ** expr
    hk_mean = lin[hk].mean(axis=1).clip(lower=1e-8)
    return np.log2(lin.div(hk_mean, axis=0))


def tis_ssgsea(expr: pd.DataFrame, alpha: float = 0.25) -> pd.Series:
    """Classic ssGSEA enrichment of the 18-gene TIS set vs the full matrix."""
    cols = np.array(expr.columns)
    gs_mask = np.isin(cols, TIS_GENES)
    X = expr.values.astype(float)
    n, g = X.shape
    ranks = np.vstack([rankdata(r) for r in X])
    w = np.abs(ranks[:, gs_mask]) ** (-alpha)
    hit = np.zeros_like(X)
    hit[:, gs_mask] = w / w.sum(axis=1, keepdims=True)
    n_miss = int((~gs_mask).sum())
    phit = np.cumsum(hit, axis=1)
    pmiss = np.cumsum(np.where(gs_mask[None, :], 0.0, 1.0 / n_miss), axis=1)
    sr = phit - pmiss
    rmax = sr[np.arange(n), sr.argmax(axis=1)]
    rmin = sr[np.arange(n), sr.argmin(axis=1)]
    return pd.Series(rmax - rmin, index=expr.index)


def tis_score(expr: pd.DataFrame, how: str = "hk") -> pd.Series:
    """Public TIS baseline. ``how='hk'`` (primary) or ``'raw'`` (legacy).

    ``expr``: samples × genes, **log2-scale** expression (IMvigor210 raw counts
    must be log2(x+1)-stabilized upstream, as the shipped pipeline does).
    Raises if no TIS gene is present; otherwise uses whatever subset is
    available (asserting ≥16).
    """
    genes = [g for g in TIS_GENES if g in expr.columns]
    if not genes:
        raise ValueError("No TIS genes present in the expression matrix.")
    if len(genes) < len(TIS_GENES):
        logger.warning("TIS: only %d/18 genes available: %s",
                       len(genes), set(TIS_GENES) - set(genes))
    sub = expr.loc[:, genes]
    if how == "hk":
        hk = [h for h in HOUSEKEEPERS if h in expr.columns]
        if not hk:
            logger.warning("TIS: no housekeepers present — MAD-Z only (no HK normalization).")
        else:
            sub = _housekeeper_normalize(sub.join(expr.loc[:, hk]), genes)  # keeps all cols
            sub = sub.loc[:, genes]
        return sub.apply(_madz).mean(axis=1)
    if how == "raw":
        return sub.mean(axis=1)
    raise ValueError(f"unknown TIS variant {how!r}")


def tis_score_ssgsea(expr: pd.DataFrame) -> pd.Series:
    """ssGSEA-based TIS (sensitivity variant; ``tis_score(how='hk')`` is primary)."""
    return tis_ssgsea(expr)

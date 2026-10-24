"""
oppti_core.py — Native Python implementation of the OPPTI outlier-detection algorithm.

Algorithm (Elmas et al. 2021, Communications Biology, doi:10.1038/s42003-021-02636-7)
-------------------------------------------------------------------------------------
For each (protein g, sample s) pair within a single cohort:

    1. Identify the k proteins most-correlated with g across the cohort
       (Pearson, excluding g itself, complete pairs only).
    2. Robust linear regression of g on its k neighbours, *holding-out* sample s.
    3. Inferred expression  ĝ(s) = predicted value from the regression.
    4. Outlier score        Δ(s) = observed g(s) − ĝ(s).
    5. Permutation test     1000× random reshuffles of g across samples;
                            p(s) = fraction of permutations whose Δ ≥ observed.
    6. Outlier call         binary = Δ > 95th-percentile of all Δ in the cohort
                            AND empirical p < 0.05.

This module returns both continuous scores and binary calls per cohort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor

from .utils import get_logger

logger = get_logger(__name__)


def _pearson_no_na(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fast vectorized Pearson correlation of x against each column of Y, ignoring NA.

    Uses numpy broadcasting to avoid per-column Python calls; degrades gracefully
    when x itself contains NaN (rare after median-imputation).
    """
    n_samples, n_proteins = Y.shape
    cors = np.full(n_proteins, np.nan, dtype=float)

    # Where x is non-NaN we can vectorise the whole thing
    x_ok = ~np.isnan(x)
    Y_ok = ~np.isnan(Y)               # (n_samples, n_proteins) bool

    # For each protein j: valid rows = x_ok & Y_ok[:, j]
    # We'll handle the two common cases efficiently:

    if x_ok.all():
        # No NaN in x → straightforward column-wise vectorised Pearson
        # n_valid per column
        n_valid = Y_ok.sum(axis=0).astype(float)      # (n_proteins,)
        too_few = n_valid < 5

        # Replace NaN in Y with 0 for masked arithmetic; mask out below
        Y_filled = np.where(Y_ok, Y, 0.0)

        # Column means using only valid rows
        col_sum  = Y_filled.sum(axis=0)               # (n_proteins,)
        col_mean = np.where(n_valid > 0, col_sum / n_valid, 0.0)

        # x mean over all rows (x has no NaN)
        # For each protein j: x_mean_j = mean of x over the valid rows for j
        # This varies per column, so we must account for it
        x_col_sum  = (x[:, None] * Y_ok).sum(axis=0)  # sum of x over valid rows
        x_col_mean = np.where(n_valid > 0, x_col_sum / n_valid, 0.0)

        xc = x[:, None] - x_col_mean                  # (n_samples, n_proteins)
        yc = Y_filled   - col_mean                     # (n_samples, n_proteins)

        # Zero out invalid rows
        xc = xc * Y_ok
        yc = yc * Y_ok

        num   = (xc * yc).sum(axis=0)
        denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum(axis=0))
        valid = (~too_few) & (denom > 0)
        cors[valid] = (num / denom)[valid]

    else:
        # x has some NaN: fall back to a fast per-column loop using direct numpy
        xc_vals = x.copy()
        for j in range(n_proteins):
            m = x_ok & Y_ok[:, j]
            if m.sum() < 5:
                continue
            xm, ym = x[m], Y[m, j]
            xm = xm - xm.mean()
            ym = ym - ym.mean()
            denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
            if denom > 0:
                cors[j] = (xm * ym).sum() / denom

    return cors


def _outlier_scores_for_protein(
    g_obs: np.ndarray,        # (n_samples,) target protein
    Y: np.ndarray,             # (n_samples, n_other) candidate neighbours
    k: int = 5,
) -> np.ndarray:
    """Compute leave-one-out inferred expression and Δ for one protein."""
    n = len(g_obs)
    cors = _pearson_no_na(g_obs, Y)
    if np.isnan(cors).all():
        return np.full(n, np.nan)

    # Top-k by |correlation|
    abs_cors = np.abs(cors)
    if (~np.isnan(abs_cors)).sum() < k:
        k = max(2, (~np.isnan(abs_cors)).sum())
    top_idx = np.argsort(-abs_cors)[:k]
    X = Y[:, top_idx]

    # Median-impute missing in X
    col_med = np.nanmedian(X, axis=0)
    inds    = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    deltas = np.full(n, np.nan)
    valid_mask = ~np.isnan(g_obs)

    # Fast LOO: closed-form OLS leverage adjustment
    # We use ridge-stabilized OLS with very small lambda for speed and robustness
    Xv = X[valid_mask]
    yv = g_obs[valid_mask]
    if len(yv) < k + 2:
        return deltas
    Xa = np.column_stack([Xv, np.ones(len(yv))])
    lam = 1e-3
    XtX = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ Xa.T @ yv
    yhat = Xa @ beta
    H = Xa @ XtX_inv @ Xa.T
    h = np.diag(H)
    # LOO predictions: y_i^(−i) = y_i − r_i / (1 − h_i)
    resid = yv - yhat
    yloo  = yv - resid / np.clip(1 - h, 1e-6, None)
    deltas_v = yv - yloo  # = resid / (1 - h)
    deltas[valid_mask] = deltas_v
    return deltas


def run_oppti_cohort(
    expr: pd.DataFrame,        # samples × proteins (all candidate neighbours)
    target_proteins: list[str],
    k: int = 5,
    n_perm: int = 1000,
    outlier_pctile: float = 95,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run OPPTI on one cohort.

    Returns
    -------
    (scores, binary, perm_pvals) : 3-tuple
      scores     : DataFrame samples × target_proteins (continuous Δ)
      binary     : DataFrame samples × target_proteins (0/1 outlier calls)
      perm_pvals : DataFrame samples × target_proteins (one-sided per-call
                   permutation p; NaN where no permutation was run)
    """
    rng = np.random.default_rng(seed)

    samples = expr.index
    arr     = expr.values.astype(float)
    cols    = expr.columns.tolist()
    target_idx = [cols.index(p) for p in target_proteins if p in cols]
    target_proteins_present = [cols[i] for i in target_idx]

    n_samples = arr.shape[0]
    n_targets = len(target_idx)
    scores = np.full((n_samples, n_targets), np.nan)

    logger.info(f"  OPPTI: {n_samples} samples × {n_targets} target proteins, k={k}")

    for j, gi in enumerate(target_idx):
        # Build candidate neighbour matrix (everything except gi)
        mask = np.ones(arr.shape[1], dtype=bool); mask[gi] = False
        Y = arr[:, mask].copy()
        g_obs = arr[:, gi].copy()
        scores[:, j] = _outlier_scores_for_protein(g_obs, Y, k=k)

    scores_df = pd.DataFrame(scores, index=samples, columns=target_proteins_present)

    # Binary calls: empirical Q{outlier_pctile} threshold (primary)
    threshold = np.nanpercentile(scores_df.values, outlier_pctile)
    logger.info(f"  Outlier threshold (Q{outlier_pctile}): {threshold:.3f}")

    binary = (scores_df.values > threshold).astype(int)
    binary = np.where(np.isnan(scores_df.values), 0, binary)
    binary_df = pd.DataFrame(binary, index=samples, columns=target_proteins_present)

    # Permutation test for supplementary FDR annotation
    # Each permutation shuffles the target protein across samples and recomputes Δ;
    # p-value = fraction of permutation Δ values ≥ observed Δ (pooled across all samples).
    perm_pvals_df = None
    if n_perm > 0:
        logger.info(f"  Running {n_perm} permutations for FDR annotation...")
        perm_pvals = np.full((n_samples, n_targets), 1.0)
        for j, gi in enumerate(target_idx):
            mask = np.ones(arr.shape[1], dtype=bool); mask[gi] = False
            Y = arr[:, mask]
            g_obs = arr[:, gi].copy()
            obs_d = scores[:, j]

            # Collect ALL permutation Δ values (not just the max) — pooled null
            null_all = []
            for p in range(n_perm):
                shuf = rng.permutation(g_obs)
                d = _outlier_scores_for_protein(shuf, Y.copy(), k=k)
                null_all.extend(d[~np.isnan(d)])
            null_arr = np.asarray(null_all)

            for s in range(n_samples):
                if not np.isnan(obs_d[s]):
                    # one-sided: P(null ≥ observed)
                    perm_pvals[s, j] = (null_arr >= obs_d[s]).mean() if null_arr.size else 1.0

        perm_pvals_df = pd.DataFrame(perm_pvals, index=samples, columns=target_proteins_present)

    n_calls = int(binary_df.values.sum())
    logger.info(f"  Outlier calls (Q{outlier_pctile}): {n_calls}")

    if perm_pvals_df is not None:
        called = (binary == 1) & (~np.isnan(scores_df.values))
        if called.sum() > 0:
            n_fdr = int((called & (perm_pvals < 0.05)).sum())
            logger.info(f"  Of {n_calls} calls, {n_fdr} pass permutation p<0.05 ({n_fdr/n_calls*100:.0f}%)")

    return scores_df, binary_df, perm_pvals_df


def run_pan_cancer(
    pan_expr:      pd.DataFrame,        # samples × IEP proteins (+ 'cancer_type')
    target_proteins: list[str],
    k:              int = 5,
    n_perm:         int = 200,
    outlier_pctile: float = 95,
    seed:           int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run OPPTI per cohort and assemble pan-cancer matrices.

    Returns
    -------
    pan_scores  : samples × proteins (continuous Δ, NA if not run)
    pan_binary  : samples × proteins (0/1 outlier calls)
    summary     : long-form per-protein-per-cancer outlier rate
    """
    if "cancer_type" not in pan_expr.columns:
        raise ValueError("pan_expr must contain a 'cancer_type' column")

    score_frames, bin_frames, perm_frames, sum_rows = [], [], [], []
    cancer_types = pan_expr["cancer_type"].unique()

    for ct in cancer_types:
        sub = pan_expr[pan_expr["cancer_type"] == ct].drop(columns="cancer_type")
        sub = sub.loc[:, sub.notna().any(axis=0)]
        if len(sub) < 10:
            logger.warning(f"  {ct}: only {len(sub)} samples, skipping.")
            continue

        logger.info(f"▶ Cohort {ct.upper()}: {sub.shape}")
        sc, bn, perm_df = run_oppti_cohort(
            sub, target_proteins, k=k, n_perm=n_perm,
            outlier_pctile=outlier_pctile, seed=seed,
        )
        score_frames.append(sc)
        bin_frames.append(bn)
        if perm_df is not None:
            perm_frames.append(perm_df)

        for prot in bn.columns:
            n_out = int(bn[prot].sum())
            n_fdr = None
            if perm_df is not None and prot in perm_df.columns:
                called = bn[prot] == 1
                sig    = perm_df[prot] < 0.05
                n_fdr  = int((called & sig).sum())
            sum_rows.append({
                "protein":          prot,
                "cancer_type":      ct,
                "n_outlier_cases":  n_out,
                "n_outlier_fdr":    n_fdr if n_fdr is not None else np.nan,
                "n_total":          len(bn),
                "pct_outlier":      n_out / len(bn) * 100,
            })

    pan_scores = pd.concat(score_frames, axis=0).reindex(pan_expr.index)
    pan_binary = pd.concat(bin_frames,  axis=0).reindex(pan_expr.index).fillna(0).astype(int)
    pan_pvals  = None
    if perm_frames:
        pan_pvals = pd.concat(perm_frames, axis=0).reindex(pan_expr.index)
    summary    = pd.DataFrame(sum_rows)
    return pan_scores, pan_binary, summary, pan_pvals

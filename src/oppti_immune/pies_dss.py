"""
pies_dss.py — PIES-DSS: Decision Support System variant of PIES
===============================================================

Motivation
----------
The original PIES uses `a priori` hand-set module weights (checkpoint +3.0,
antigen presentation -2.5, ...). These were curated from the immune-evasion
literature, which invites the criticism that the weights were tuned (directly
or indirectly) on the development cohort. PIES-DSS replaces the hand-set
weights with **learned module weights**: the per-module outlier count features
are combined by an L2-regularised logistic regression whose coefficients are
fitted inside a nested cross-validation loop, so that:

  * the out-of-sample AUC estimate is unbiased (outer folds are never touched
    during training),
  * the learned coefficients remain interpretable as "per-outlier-event log-odds
    contributions of each immune-evasion module",
  * the sign of each module weight is learned rather than asserted — including
    the biologically debatable mismatch_repair sign.

Score definition (RNA input, deployable) ---------------------------------
For each sample s and module m:

    x[s, m] = Σ_{g ∈ module m, Z_rna(s,g) > threshold} (1 + pt_frac[g])

    P(DSS)  = P(non-responder | x)   via logistic regression

i.e., higher score → more predicted immune evasion → more likely to not
respond to ICB. The model still consumes only RNA, which preserves the
deployment advantage of the RNA-direct pipeline; the CPTAC-derived RNA-protein
discordance fractions (pt_frac) enter through feature construction rather than
hand-set multipliers.

Statistical design
------------------
* Development cohort  : IMvigor210 (largest cohort, n≈298 labeled).
* Outer evaluation    : nested CV — outer StratifiedKFold(5) gives the unbiased
  AUC; inner StratifiedKFold(5) selects the regularisation C per outer fold.
* Held-out cohorts    : Hugo_2016 and Riaz_2017 — weights learned on IMvigor210
  are applied directly; the only cohort-local step is the within-cohort MAD-Z
  normalisation that defines outlier calls (identical to the RNA-direct PIES).
* Uncertainty         : stratified-bootstrap CI on held-out AUCs; label-
  permutation nulls; DeLong paired tests for PIES-DSS vs PIES vs TIS.
* Baselines           : hand-weight PIES (same outlier calls), TIS signature.

Outputs (results/) --------------------------------------------------------
* aim4_pies_dss_nested_cv.csv        : nested-CV AUC per fold + summary
* aim4_pies_dss_weights.csv          : learned module weights (with CIs)
* aim4_pies_dss_heldout.csv          : Hugo/Riaz evaluation of the final model
* aim4_pies_dss_comparison.csv       : paired PIES-DSS vs PIES vs TIS metrics
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .pies import TIS_GENES
from .utils import (
    get_logger, harmonize_response, load_config, load_gene_panel,
    results_path, save_csv,
)

logger = get_logger("pies_dss")

C_GRID = np.array([0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0])
MODULE_ORDER = [
    "checkpoint", "antigen_presentation", "cytokines", "ifn_pathway",
    "metabolic", "ecm_exclusion", "epigenetic", "mismatch_repair",
]


# ── Feature construction (shared with the RNA-direct PIES) ───────────────────

def madz_outliers(
    rna: pd.DataFrame,
    genes: list[str],
    med: pd.Series | None = None,
    mad: pd.Series | None = None,
    z_threshold: float = 1.645,
) -> pd.DataFrame:
    """MAD-normalise RNA (≈95th-percentile one-sided) and call binary outliers.

    If `med`/`mad` are None they are estimated from the data itself (in-sample
    mode); otherwise the supplied training-fold statistics are applied (CV mode).
    """
    available = [g for g in genes if g in rna.columns]
    sub = rna[available].astype(float)
    if med is None:
        med = sub.median()
    if mad is None:
        mad = (sub - med).abs().median().replace(0, 1e-6)
    mad = mad.replace(0, 1e-6)
    z = (sub[available] - med[available]) / mad[available]
    return (z > z_threshold).astype(int).where(~z.isna(), 0)


def module_features(
    outlier_binary: pd.DataFrame,
    gene_panel: pd.DataFrame,
    pt_frac: pd.Series,
) -> pd.DataFrame:
    """Per-sample module features: pt_frac-weighted outlier counts per module."""
    module_map = gene_panel.set_index("gene")["module"]
    genes = [g for g in outlier_binary.columns if g in module_map.index]
    pt = pt_frac.reindex(genes).fillna(0.0) / 100.0
    weighted = outlier_binary[genes].mul((1.0 + pt).values, axis=1)
    groups = module_map.reindex(genes)
    return weighted.T.groupby(groups).sum().T[MODULE_ORDER]


def cohort_scores_from_module_features(
    feats: pd.DataFrame, weights: dict[str, float] | pd.Series
) -> pd.Series:
    """Hand-weight PIES (linear) on module features, for the paired baseline."""
    w = pd.Series(weights, index=MODULE_ORDER).reindex(feats.columns).fillna(0.0)
    return feats @ w.values


def tis_score(rna: pd.DataFrame) -> pd.Series:
    """Mean-expression TIS score (same proxy as the PIES validation scripts)."""
    tis_avail = [g for g in TIS_GENES if g in rna.columns]
    if not tis_avail:
        raise ValueError("No TIS genes present in the expression matrix.")
    return rna[tis_avail].mean(axis=1)


# ── Core learners ─────────────────────────────────────────────────────────────

def fit_dss(
    X: pd.DataFrame, y_nonresp: np.ndarray, C: float
) -> LogisticRegression:
    """L2-regularised logistic regression on module-count features.

    Notes: no feature scaling — the raw per-module outlier counts are already on
    a common scale, so a coefficient IS the learned module weight (log-odds per
    pt_frac-weighted outlier event), preserving interpretability.
    """
    model = LogisticRegression(
        penalty="l2", C=C, solver="liblinear", max_iter=5000, random_state=42
    )
    model.fit(X.values, y_nonresp)
    return model


def select_C_via_inner_cv(
    X: pd.DataFrame, y_nonresp: np.ndarray, seed: int = 42
) -> float:
    """Inner 5-fold CV: pick C maximising mean inner AUC (ties → smaller C)."""
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros(len(C_GRID))
    for tr, va in inner.split(X, y_nonresp):
        for j, C in enumerate(C_GRID):
            model = fit_dss(X.iloc[tr], y_nonresp[tr], C)
            p = model.predict_proba(X.iloc[va].values)[:, 1]
            try:
                scores[j] += roc_auc_score(y_nonresp[va], p)
            except ValueError:
                scores[j] += 0.5
    scores /= inner.get_n_splits()
    j_best = int(np.argmax(scores))
    return float(C_GRID[j_best])


# ── Nested CV on the development cohort ──────────────────────────────────────

def nested_cv_dss(
    feats: pd.DataFrame,
    response_binary: pd.Series,
    seed: int = 42,
    n_outer: int = 5,
    hand_weights: dict[str, float] | pd.Series | None = None,
    tis_scores: pd.Series | None = None,
) -> dict[str, Any]:
    """Nested CV: outer folds provide the unbiased out-of-fold predictions.

    For every outer fold: inner CV selects C, the model is refit on the full
    outer-training set, and the held-out fold is scored once. When
    `hand_weights` / `tis_scores` are supplied, out-of-fold scores of the
    hand-set PIES and TIS baselines are collected on the SAME outer folds, so
    the pooled IoF AUCs support an honest, paired DeLong comparison.
    """
    y_nr = (1 - response_binary.astype(int)).values  # 1 = non-responder
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    rows: list[dict[str, Any]] = []
    coef_matrix: list[np.ndarray] = []
    dss_iof = np.full(len(feats), np.nan)
    pies_iof = np.full(len(feats), np.nan)
    tis_iof = np.full(len(feats), np.nan)

    for i, (tr, te) in enumerate(outer.split(feats, response_binary)):
        C = select_C_via_inner_cv(feats.iloc[tr], y_nr[tr])
        model = fit_dss(feats.iloc[tr], y_nr[tr], C)
        p_te = model.predict_proba(feats.iloc[te].values)[:, 1]
        auc = float(roc_auc_score(y_nr[te], p_te)) if len(np.unique(y_nr[te])) == 2 else np.nan
        coef_matrix.append(model.coef_.ravel())
        rows.append({
            "outer_fold": i + 1,
            "C_selected": C,
            "auc": auc,
            "n_test": int(len(te)),
            "n_responders_test": int(response_binary.iloc[te].sum()),
        })
        dss_iof[te]  = p_te
        if hand_weights is not None:
            pies_iof[te] = cohort_scores_from_module_features(
                feats.iloc[te], hand_weights
            ).values
        if tis_scores is not None:
            tis_iof[te] = -tis_scores.iloc[te].values   # higher TIS → responder
        logger.info(f"  outer fold {i+1}: C={C:g}, held-out AUC={auc:.3f}")

    folds = np.array([r["auc"] for r in rows], dtype=float)
    coef = np.vstack(coef_matrix)
    weights = pd.DataFrame({
        "module":  MODULE_ORDER,
        "coef_median": np.median(coef, axis=0),
        "coef_mean":   coef.mean(axis=0),
        "coef_min":    coef.min(axis=0),
        "coef_max":    coef.max(axis=0),
    }).sort_values("coef_median", ascending=False)

    return {
        "folds": rows,
        "auc_mean": float(np.nanmean(folds)),
        "auc_std": float(np.nanstd(folds)),
        "auc_folds": [float(a) for a in folds],
        "weights": weights,
        "out_of_fold": {
            "y_nr": y_nr,
            "PIES-DSS": dss_iof,
            "PIES (hand)": None if hand_weights is None else pies_iof,
            "TIS": None if tis_scores is None else tis_iof,
        },
    }


# ── Bootstrap CI + permutation null on held-out cohorts ──────────────────────

def bootstrap_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    obs = float(roc_auc_score(y, score))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bi = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        try:
            boots[i] = roc_auc_score(y[bi], score[bi])
        except ValueError:
            boots[i] = np.nan
    return {
        "auc": obs,
        "ci_lo": float(np.nanquantile(boots, 0.025)),
        "ci_hi": float(np.nanquantile(boots, 0.975)),
        "n": int(len(y)),
        "n_responders": int(y.sum()),
    }


def permutation_p(y: np.ndarray, score: np.ndarray, n_perm: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    obs = float(roc_auc_score(y, score))
    null = np.array([
        roc_auc_score(rng.permutation(y), score)
        if len(np.unique(y)) == 2 else 0.5
        for _ in range(n_perm)
    ])
    p_up   = float((null >= obs).mean())
    p_down = float((null <= obs).mean())
    return {
        "perm_p_two_sided": min(2.0 * min(p_up, p_down), 1.0),
        "perm_p_up": p_up,
        "perm_p_down": p_down,
        "null_mean": float(null.mean()),
        "n_perm": n_perm,
    }


def delong_paired(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    """DeLong covariance-based two-sided p for paired AUC comparison (s1 vs s2).

    Standard DeLong method (DeLong et al. 1988): placement values of positives
    among negatives and vice-versa, and a covariance estimate across cohorts.
    Reported as a supportive (not confirmatory) paired test.
    """
    s1p, s2p = s1[y == 1], s2[y == 1]
    s1n, s2n = s1[y == 0], s2[y == 0]
    m, n = len(s1p), len(s1n)
    if m < 2 or n < 2:
        return np.nan

    def placement(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        r = np.sort(ref)
        lo = np.searchsorted(r, x, side="left")
        hi = np.searchsorted(r, x, side="right")
        return (lo + 0.5 * (hi - lo)) / len(r)

    v1 = placement(s1p, s1n).mean()   # AUC of model 1
    v2 = placement(s2p, s2n).mean()   # AUC of model 2

    x1, x2 = placement(s1p, s1n), placement(s2p, s2n)
    z1, z2 = placement(s1n, s1p), placement(s2n, s2p)

    cov10 = np.cov(x1, x2, ddof=1)[0, 1] if m > 1 else 0.0
    cov01 = np.cov(z1, z2, ddof=1)[0, 1] if n > 1 else 0.0
    var1 = x1.var(ddof=1) / m + z1.var(ddof=1) / n
    var2 = x2.var(ddof=1) / m + z2.var(ddof=1) / n
    cov  = cov10 / m + cov01 / n
    zval = (v1 - v2) / np.sqrt(max(var1 + var2 - 2 * cov, 1e-12))
    return float(2.0 * norm.sf(abs(zval)))


# ── Cohort assembly ───────────────────────────────────────────────────────────

def load_cohort(name: str, cfg: dict):
    """Load raw ICB cohort and harmonise response labels.

    Returns (rna dataframe, response_binary series).
    """
    expr = pd.read_csv(f"data/immunotherapy/{name}_expr.csv", index_col=0).T
    clin = pd.read_csv(f"data/immunotherapy/{name}_clinical.csv", index_col=0)
    if name == "IMvigor210":
        expr = np.log2(expr + 1)
        rcol = "binaryResponse"
    else:
        rcol = "response"
    resp = harmonize_response(clin[rcol], cfg)
    merged = pd.concat([expr, resp.rename("y")], axis=1).dropna(subset=["y"])
    return merged.drop(columns=["y"]), merged["y"].astype(int)


# ── Master entry point ────────────────────────────────────────────────────────

def run(cfg: dict, n_perm: int = 1000, seed: int = 42) -> None:
    gene_panel = load_gene_panel(cfg)
    panel_genes = gene_panel["gene"].tolist()

    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()

    z_thr = cfg.get("concordance", {}).get("outlier_z_threshold", 1.645)
    hand_weights = cfg["pies"]["module_weights"]

    # ── IMvigor210 development cohort, nested CV ─────────────────────────────
    logger.info("=== PIES-DSS: nested CV on IMvigor210 (development cohort) ===")
    rna_dev, y_dev = load_cohort("IMvigor210", cfg)

    outliers_dev = madz_outliers(rna_dev, panel_genes, z_threshold=z_thr)
    feats_dev = module_features(outliers_dev, gene_panel, pt_frac)
    tis_dev = tis_score(rna_dev).reindex(feats_dev.index).bfill()
    logger.info(f"  n={len(feats_dev)}, responders={int(y_dev.sum())}, "
                f"modules={feats_dev.shape[1]}")

    nc = nested_cv_dss(feats_dev, y_dev, seed=seed,
                       hand_weights=hand_weights, tis_scores=tis_dev)
    logger.info(f"  nested-CV AUC = {nc['auc_mean']:.3f} ± {nc['auc_std']:.3f} "
                f"(folds: {[round(a, 3) for a in nc['auc_folds']]})")
    save_csv(pd.DataFrame(nc["folds"]),
             results_path(cfg, "aim4_pies_dss_nested_cv.csv"), index=False)

    weights = nc["weights"]
    logger.info("  learned module weights (per-outlier-event log-odds):\n"
                + weights.to_string(index=False))
    save_csv(weights, results_path(cfg, "aim4_pies_dss_weights.csv"), index=False)

    # Final model on ALL development data (for held-out application)
    y_nr_dev = (1 - y_dev.astype(int)).values
    C_final = select_C_via_inner_cv(feats_dev, y_nr_dev, seed=seed)
    final_model = fit_dss(feats_dev, y_nr_dev, C_final)
    logger.info(f"  refit on full development set: C={C_final:g}")

    # Learned weights as a plain series (for the shared scoring path)
    w = final_model.coef_.ravel()

    # ── Held-out cohorts: Hugo_2016, Riaz_2017 ───────────────────────────────
    logger.info("=== Held-out validation: Hugo_2016 / Riaz_2017 ===")
    heldout_rows: list[dict[str, Any]] = []
    heldout_scores: list[pd.DataFrame] = []
    for cohort in ("Hugo_2016", "Riaz_2017"):
        rna, y = load_cohort(cohort, cfg)
        outliers = madz_outliers(rna, panel_genes, z_threshold=z_thr)
        feats = module_features(outliers, gene_panel, pt_frac)
        p_dss = final_model.predict_proba(feats.values)[:, 1]
        heldout_scores.append(pd.DataFrame({
            "cohort": cohort,
            "score_nonresponse": p_dss,
        }, index=feats.index))
        y_nr = (1 - y.astype(int)).values

        res = bootstrap_ci(y_nr, p_dss)
        res.update(permutation_p(y_nr, p_dss, n_perm=n_perm, seed=seed))
        res["cohort"] = cohort
        res["model"] = "PIES-DSS"
        heldout_rows.append(res)
        logger.info(f"  {cohort}: AUC={res['auc']:.3f} "
                    f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}] "
                    f"perm p={res['perm_p_two_sided']:.3f}")

    save_csv(pd.DataFrame(heldout_rows),
             results_path(cfg, "aim4_pies_dss_heldout.csv"), index=False)
    save_csv(pd.concat(heldout_scores),
             results_path(cfg, "aim4_pies_dss_heldout_scores.csv"), index=True,
             index_label="sample")

    # ── Paired comparison on OUT-OF-FOLD development predictions ────────────
    # All three models are scored on the same held-out outer folds, so the
    # pooled AUCs are directly comparable (no training-set optimism).
    logger.info("=== Paired comparison (IMvigor210; pooled out-of-fold scores) ===")
    iof = nc["out_of_fold"]
    y_nr_full = iof["y_nr"]
    comp: list[dict[str, Any]] = []
    for label in ("PIES-DSS", "PIES (hand)", "TIS"):
        score = iof[label]
        if score is None:
            continue
        b = bootstrap_ci(y_nr_full, np.asarray(score))
        b.update(permutation_p(y_nr_full, np.asarray(score), n_perm=n_perm, seed=seed + 1))
        b["cohort"], b["model"] = "IMvigor210 (out-of-fold)", label
        comp.append(b)
        logger.info(f"  {label}: AUC={b['auc']:.3f} [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}] "
                    f"perm p={b['perm_p_two_sided']:.3f}")

    dss_s = iof["PIES-DSS"]
    if iof["PIES (hand)"] is not None:
        comp.append({
            "cohort": "IMvigor210 (out-of-fold)", "model": "DeLong p: DSS vs PIES (hand)",
            "auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "perm_p_two_sided": delong_paired(y_nr_full, dss_s, iof["PIES (hand)"]),
            "n": len(y_nr_full), "n_responders": int(y_nr_full.sum()), "n_perm": np.nan,
        })
    if iof["TIS"] is not None:
        comp.append({
            "cohort": "IMvigor210 (out-of-fold)", "model": "DeLong p: DSS vs TIS",
            "auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "perm_p_two_sided": delong_paired(y_nr_full, dss_s, iof["TIS"]),
            "n": len(y_nr_full), "n_responders": int(y_nr_full.sum()), "n_perm": np.nan,
        })
    save_csv(pd.DataFrame(comp),
             results_path(cfg, "aim4_pies_dss_comparison.csv"), index=False)

    # Per-sample out-of-fold scores (for figures / external re-scoring).
    iof_df = pd.DataFrame({k: v for k, v in iof.items() if k != "y_nr"},
                          index=feats_dev.index)
    iof_df["nonresponder"] = y_nr_full.astype(int)
    save_csv(iof_df, results_path(cfg, "aim4_pies_dss_iof_scores.csv"),
             index=True, index_label="sample")

    logger.info("=== PIES-DSS Complete ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PIES-DSS: nested-CV learned module weights")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.config, n_perm=args.n_perm, seed=args.seed)

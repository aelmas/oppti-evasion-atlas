#!/usr/bin/env python
"""
Step 11: Honest PIES validation with cross-validation
======================================================

Background (peer-review blocker)
--------------------------------
The PIES design (direction rule, default mode='rna') was selected by observing
IMvigor210 AUC move from 0.518 → 0.643 (commit 600dc3f). This is textbook
test-set leakage. The headline AUC=0.643 is therefore IN-SAMPLE on IMvigor210.

This script reports the honest counterpart:

  (a) IMvigor210 5-fold stratified CV AUC — an unbiased out-of-sample estimate
      under the assumption that the *design choice* (module weights + mode)
      was the only thing tuned, not the per-cohort hyperparameters.

  (b) Permutation null on IMvigor210 — randomize response labels 1,000× and
      score PIES vs random; report the empirical p-value of the observed AUC.

  (c) Hugo_2016 and Riaz_2017 are kept as truly held-out test cohorts; the
      original validation script (07_validate_icb_cohorts.py) already reports
      their in-sample AUC against the current PIES design. Their results are
      restated here for the disclosure table.

Outputs:
  results/aim4_honest_validation.csv

Usage:  python scripts/11_honest_pies_validation.py
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from oppti_immune.pies import compute_rna_pies, compute_pies_all_samples, compute_rna_outlier_binary
from oppti_immune.utils import (
    harmonize_response, load_config, load_gene_panel,
    results_path, save_csv, get_logger,
)

logger = get_logger("11_honest_validation")


def _load_cohort(name: str, cfg: dict):
    expr = pd.read_csv(f"data/immunotherapy/{name}_expr.csv", index_col=0).T
    clin = pd.read_csv(f"data/immunotherapy/{name}_clinical.csv", index_col=0)
    if name == "IMvigor210":
        expr = np.log2(expr + 1)
        rcol = "binaryResponse"
    else:
        rcol = "response"
    clin["resp_binary"] = harmonize_response(clin[rcol], cfg)
    return expr, clin


def cv_pies_within_cohort(
    name: str, cfg: dict, gene_panel: pd.DataFrame,
    pt_frac: pd.Series, cd8: pd.Series,
    n_splits: int = 5, seed: int = 42,
) -> dict:
    """5-fold CV AUC: in each fold, compute the PIES Z-thresholds using only the
    training samples (median+MAD), apply them to held-out samples, then score AUC.

    This is the strongest CV we can do without changing the PIES design itself
    — the module weights are still fixed across folds (they were chosen a priori
    using CPTAC data, not this cohort), so the only thing being held out is the
    within-cohort Z-normalization that defines outlier calls.
    """
    expr, clin = _load_cohort(name, cfg)
    merged = pd.concat([expr, clin[["resp_binary"]]], axis=1).dropna(subset=["resp_binary"])
    y = merged["resp_binary"].astype(int).values
    X = merged.drop(columns=["resp_binary"])

    if len(np.unique(y)) < 2:
        return {"cohort": name, "cv_auc_mean": np.nan, "cv_auc_std": np.nan, "n": len(y)}

    z_thr = cfg.get("concordance", {}).get("outlier_z_threshold", 1.645)
    panel_genes = gene_panel["gene"].tolist()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]

        # Compute MAD-Z thresholds on TRAINING samples only
        available = [g for g in panel_genes if g in X_train.columns]
        train_sub = X_train[available].astype(float)
        med = train_sub.median()
        mad = (train_sub - med).abs().median().replace(0, 1e-6)

        # Apply same shift+scale to TEST samples, then threshold
        test_sub = X_test[available].astype(float)
        test_z = (test_sub - med) / mad
        test_outliers = (test_z > z_thr).astype(int)

        # Compute PIES on test outliers using the fixed (a-priori) module weights
        pies_test = compute_pies_all_samples(
            test_outliers, gene_panel, pt_frac, cd8, cfg,
        )
        y_test = y[test_idx]
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, -pies_test.values)
            fold_aucs.append(auc)
        else:
            fold_aucs.append(np.nan)

    fold_aucs = np.array(fold_aucs)
    return {
        "cohort":       name,
        "cv_auc_mean":  float(np.nanmean(fold_aucs)),
        "cv_auc_std":   float(np.nanstd(fold_aucs)),
        "cv_auc_folds": ";".join(f"{x:.3f}" if not np.isnan(x) else "NA" for x in fold_aucs),
        "n":            int(len(y)),
        "n_responders": int(y.sum()),
    }


def permutation_null_auc(
    name: str, cfg: dict, gene_panel: pd.DataFrame,
    pt_frac: pd.Series, cd8: pd.Series,
    n_perm: int = 1000, seed: int = 42,
) -> dict:
    """Empirical AUC null: randomize response labels n_perm× and report
    P(null_auc >= observed_auc).
    """
    rng = np.random.default_rng(seed)
    expr, clin = _load_cohort(name, cfg)
    pies, _ = compute_rna_pies(expr, gene_panel, pt_frac, cd8, cfg)
    merged = pd.concat([pies, clin[["resp_binary"]]], axis=1).dropna(subset=["resp_binary"])
    y = merged["resp_binary"].astype(int).values
    s = -merged["PIES"].values  # higher score → more evasion → predict non-responder
    if len(np.unique(y)) < 2:
        return {"cohort": name, "observed_auc": np.nan, "perm_p": np.nan, "n_perm": n_perm}

    obs_auc = roc_auc_score(y, s)
    null_aucs = np.empty(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        null_aucs[i] = roc_auc_score(y_shuf, s)
    # one-sided p
    p_up   = float((null_aucs >= obs_auc).mean())
    p_down = float((null_aucs <= obs_auc).mean())
    return {
        "cohort":         name,
        "observed_auc":   float(obs_auc),
        "perm_p_one_sided_up":   p_up,
        "perm_p_one_sided_down": p_down,
        "null_auc_mean":  float(null_aucs.mean()),
        "null_auc_q025":  float(np.quantile(null_aucs, 0.025)),
        "null_auc_q975":  float(np.quantile(null_aucs, 0.975)),
        "n_perm":         int(n_perm),
    }


def auc_bootstrap_ci(name: str, cfg: dict, gene_panel: pd.DataFrame,
                     pt_frac: pd.Series, cd8: pd.Series,
                     n_boot: int = 2000, seed: int = 42) -> dict:
    """Stratified bootstrap 95% CI for in-sample AUC."""
    rng = np.random.default_rng(seed)
    expr, clin = _load_cohort(name, cfg)
    pies, _ = compute_rna_pies(expr, gene_panel, pt_frac, cd8, cfg)
    merged = pd.concat([pies, clin[["resp_binary"]]], axis=1).dropna(subset=["resp_binary"])
    y = merged["resp_binary"].astype(int).values
    s = -merged["PIES"].values
    if len(np.unique(y)) < 2:
        return {"cohort": name, "auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    obs = roc_auc_score(y, s)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        try:
            boots[i] = roc_auc_score(y[bi], s[bi])
        except ValueError:
            boots[i] = np.nan
    return {
        "cohort": name,
        "in_sample_auc": float(obs),
        "ci_lo": float(np.nanquantile(boots, 0.025)),
        "ci_hi": float(np.nanquantile(boots, 0.975)),
        "n_boot": int(n_boot),
    }


# ── DeLong AUC variance (analytic CI cross-check) ────────────────────────────

def _delong_auc_var(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (AUC, variance) via DeLong's fast placement-value estimator.

    Provides an analytic CI that does not depend on bootstrap resampling, which
    is more reliable than a stratified percentile bootstrap at small n_resp.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan

    # Placement values (mid-rank handling of ties via Mann-Whitney structure)
    def _structural(x, ys):
        # V_x = fraction of ys that x "beats" (1 if x>y, 0.5 if tie)
        out = np.empty(len(x))
        ys_sorted = np.sort(ys)
        for i, xi in enumerate(x):
            lo = np.searchsorted(ys_sorted, xi, side="left")
            hi = np.searchsorted(ys_sorted, xi, side="right")
            out[i] = (lo + 0.5 * (hi - lo)) / len(ys)
        return out

    v10 = _structural(pos, neg)      # placement of each positive among negatives
    v01 = 1.0 - _structural(neg, pos)  # placement of each negative among positives
    auc = v10.mean()
    s10 = v10.var(ddof=1) if m > 1 else 0.0
    s01 = v01.var(ddof=1) if n > 1 else 0.0
    var = s10 / m + s01 / n
    return float(auc), float(var)


def delong_ci(name: str, cfg: dict, gene_panel, pt_frac, cd8) -> dict:
    expr, clin = _load_cohort(name, cfg)
    pies, _ = compute_rna_pies(expr, gene_panel, pt_frac, cd8, cfg)
    merged = pd.concat([pies, clin[["resp_binary"]]], axis=1).dropna(subset=["resp_binary"])
    y = merged["resp_binary"].astype(int).values
    s = -merged["PIES"].values
    auc, var = _delong_auc_var(y, s)
    se = np.sqrt(var) if var == var else np.nan
    return {
        "cohort": name,
        "delong_auc": auc,
        "delong_se": se,
        "delong_ci_lo": float(auc - 1.96 * se) if se == se else np.nan,
        "delong_ci_hi": float(auc + 1.96 * se) if se == se else np.nan,
    }


def inverse_variance_meta_auc(rows: list[dict]) -> dict:
    """Fixed-effect inverse-variance pooled AUC across cohorts, using DeLong SEs.

    Also reports Cochran's Q heterogeneity and I^2.
    """
    aucs = np.array([r["delong_auc"] for r in rows], dtype=float)
    ses  = np.array([r["delong_se"] for r in rows], dtype=float)
    ok = np.isfinite(aucs) & np.isfinite(ses) & (ses > 0)
    aucs, ses = aucs[ok], ses[ok]
    if len(aucs) < 2:
        return {"meta_auc": np.nan}
    w = 1.0 / ses**2
    meta = float((w * aucs).sum() / w.sum())
    meta_se = float(np.sqrt(1.0 / w.sum()))
    Q = float((w * (aucs - meta) ** 2).sum())
    dfree = len(aucs) - 1
    I2 = float(max(0.0, (Q - dfree) / Q) * 100) if Q > 0 else 0.0
    return {
        "meta_auc": meta,
        "meta_ci_lo": meta - 1.96 * meta_se,
        "meta_ci_hi": meta + 1.96 * meta_se,
        "cochran_Q": Q,
        "Q_df": dfree,
        "I2_pct": I2,
        "n_cohorts": int(len(aucs)),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)

    gene_panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    tme = pd.read_csv(results_path(cfg, "aim3_protein_immune_correlation.csv"))
    cd8 = tme[tme["immune_metric"] == "CD8+ T-cells"].set_index("gene")["rho"]

    rows = []
    for cohort in ("IMvigor210", "Hugo_2016", "Riaz_2017"):
        logger.info(f"\n=== {cohort} ===")

        ci = auc_bootstrap_ci(cohort, cfg, gene_panel, pt_frac, cd8)
        logger.info(f"  in-sample AUC = {ci['in_sample_auc']:.3f}  "
                    f"95% CI [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]")

        cv = cv_pies_within_cohort(cohort, cfg, gene_panel, pt_frac, cd8)
        logger.info(f"  CV-5-fold AUC = {cv['cv_auc_mean']:.3f} ± {cv['cv_auc_std']:.3f}  "
                    f"(folds: {cv['cv_auc_folds']})")

        nl = permutation_null_auc(cohort, cfg, gene_panel, pt_frac, cd8, n_perm=args.n_perm)
        # Two-sided empirical p = min(2*min(up,down), 1)
        two_sided = min(2.0 * min(nl["perm_p_one_sided_up"], nl["perm_p_one_sided_down"]), 1.0)
        nl["perm_p_two_sided"] = float(two_sided)
        logger.info(f"  label-perm null AUC: mean={nl['null_auc_mean']:.3f}  "
                    f"95% range [{nl['null_auc_q025']:.3f}, {nl['null_auc_q975']:.3f}]  "
                    f"p_up={nl['perm_p_one_sided_up']:.3f} p_down={nl['perm_p_one_sided_down']:.3f} "
                    f"p_two_sided={two_sided:.3f}")

        dl = delong_ci(cohort, cfg, gene_panel, pt_frac, cd8)
        logger.info(f"  DeLong AUC = {dl['delong_auc']:.3f}  95% CI "
                    f"[{dl['delong_ci_lo']:.3f}, {dl['delong_ci_hi']:.3f}]")

        rows.append({**ci, **{k: v for k, v in cv.items() if k != "cohort"},
                     **{k: v for k, v in nl.items() if k != "cohort"},
                     **{k: v for k, v in dl.items() if k != "cohort"}})

    out = pd.DataFrame(rows)
    save_csv(out, results_path(cfg, "aim4_honest_validation.csv"), index=False)

    # Inverse-variance fixed-effect meta-AUC (DeLong SEs)
    meta = inverse_variance_meta_auc(rows)
    logger.info("\n=== Inverse-variance meta-AUC (fixed effect, DeLong SEs) ===")
    logger.info(f"  pooled AUC = {meta.get('meta_auc', float('nan')):.3f} "
                f"[{meta.get('meta_ci_lo', float('nan')):.3f}, {meta.get('meta_ci_hi', float('nan')):.3f}]  "
                f"Cochran Q={meta.get('cochran_Q', float('nan')):.2f} (df={meta.get('Q_df','?')})  "
                f"I²={meta.get('I2_pct', float('nan')):.1f}%")
    save_csv(pd.DataFrame([meta]), results_path(cfg, "aim4_meta_auc.csv"), index=False)

    logger.info(f"\nSaved honest validation table to results/aim4_honest_validation.csv")
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()

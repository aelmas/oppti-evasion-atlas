#!/usr/bin/env python
"""
Step 18 — Repeated nested CV + full-refit permutation null
==========================================================
Robustness upgrade over the single 5x5 nested-CV split (reviewer point 5):

1. **20 repeats** of the complete nested CV (outer 5-fold stratified splits
   re-drawn per repeat; inner C selection inside each outer fold). For each
   repeat: pooled out-of-fold (IoF) AUC with bootstrap CI, mean-of-folds AUC,
   plus paired DeLong tests against the fold-independent hand-set PIES score
   and the public housekeeper-normalized TIS (both fixed per-sample vectors).

2. **Full-refit permutation null** (200 permutations): each random label
   permutation re-runs the ENTIRE nested CV (fold splits re-drawn and the
   model refit on permuted labels), and the pooled IoF AUC is computed —
   the null that reviewers asked for, not a label-permutation on frozen
   scores. Per-repeat two-sided p values are read against this null.

Outputs
  results/aim8_repeated_cv.csv          : per-repeat row
  results/aim8_repeated_cv_summary.csv  : aggregates (mean/SD/win-rates/etc.)
  results/aim8_perm_null.csv            : 200 full-refit null AUCs

Usage:  python scripts/18_repeated_cv.py [--n-repeats 20] [--n-perm 200]
"""

import argparse
import logging

import numpy as np
import pandas as pd

from oppti_immune.pies_dss import (
    bootstrap_ci,
    cohort_scores_from_module_features,
    delong_paired,
    load_cohort,
    madz_outliers,
    module_features,
    nested_cv_dss,
    permutation_p,
)
from oppti_immune.tis import tis_score
from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("18_repeated_cv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gene_panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    z_thr = cfg.get("concordance", {}).get("outlier_z_threshold", 1.645)

    rna, y = load_cohort("IMvigor210", cfg)
    outliers = madz_outliers(rna, gene_panel["gene"].tolist(), z_threshold=z_thr)
    feats = module_features(outliers, gene_panel, pt_frac)

    # fold-independent baselines (fixed per-sample vectors, both oriented to
    # NON-response — matching the nested-CV IoF column convention exactly:
    # hand-PIES = feats @ hand_weights (module weights already encode the
    # non-response direction); TIS stored as -TIS).
    hand_fixed = cohort_scores_from_module_features(feats, cfg["pies"]["module_weights"])
    tis_fixed_raw = tis_score(rna.reindex(feats.index)).bfill()
    if tis_fixed_raw.isna().any():
        raise ValueError("TIS missing for some samples")
    tis_fixed = -tis_fixed_raw

    y_nr = (1 - y.astype(int)).values

    from sklearn.metrics import roc_auc_score as roc

    def pooled_auc(score: np.ndarray, yy: np.ndarray) -> float:
        return float(roc(yy, score))

    hand_auc = pooled_auc(hand_fixed.values, y_nr)
    tis_auc = pooled_auc(tis_fixed.values, y_nr)
    logger.info(f"fold-independent baselines: hand PIES IoF-pooled AUC={hand_auc:.3f}, "
                f"TIS(hk) pooled AUC={tis_auc:.3f}")

    # ── 1. repeats ───────────────────────────────────────────────────────────
    logging.getLogger("pies_dss").setLevel(logging.WARNING)  # silence per-fold logs
    rows = []
    for r in range(args.n_repeats):
        seed = args.seed_start + r
        nc = nested_cv_dss(feats, y, seed=seed,
                           hand_weights=cfg["pies"]["module_weights"], tis_scores=tis_fixed)
        iof = nc["out_of_fold"]
        dss = np.asarray(iof["PIES-DSS"], dtype=float)
        b = bootstrap_ci(y_nr, dss)
        row = {
            "repeat": r + 1, "seed": seed,
            "pooled_auc": b["auc"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
            "fold_mean": nc["auc_mean"], "fold_std": nc["auc_std"],
            "delong_vs_hand": float(delong_paired(y_nr, dss, hand_fixed.values)),
            "delong_vs_tis": float(delong_paired(y_nr, dss, tis_fixed.values)),
            "perm_p_labelperm": permutation_p(y_nr, dss, n_perm=1000,
                                              seed=seed)["perm_p_two_sided"],
        }
        rows.append(row)
        logger.info(f"  repeat {row['repeat']:>2d}: pooled AUC={row['pooled_auc']:.3f} "
                    f"[{row['ci_lo']:.3f},{row['ci_hi']:.3f}] "
                    f"DeLong hand={row['delong_vs_hand']:.2g} tis={row['delong_vs_tis']:.2g}")
    rep = pd.DataFrame(rows)
    aucs = rep["pooled_auc"].values

    # ── 2. full-refit permutation null ───────────────────────────────────────
    rng = np.random.default_rng(0)
    null_rows = []
    for p in range(args.n_perm):
        y_perm = pd.Series(rng.permutation(y.values), index=y.index)
        nc = nested_cv_dss(feats, y_perm, seed=10_000 + p,
                           hand_weights=cfg["pies"]["module_weights"], tis_scores=tis_fixed)
        null_auc = float(roc(nc["out_of_fold"]["y_nr"],
                             np.asarray(nc["out_of_fold"]["PIES-DSS"], dtype=float)))
        null_rows.append({"perm": p + 1, "pooled_auc": null_auc})
        if (p + 1) % 50 == 0:
            logger.info(f"  perm {p+1}/{args.n_perm}")
    nulls = pd.DataFrame(null_rows)

    # ── 3. summaries ─────────────────────────────────────────────────────────
    null_vals = nulls["pooled_auc"].values
    rep["perm_p_fullrefit"] = [
        float(np.mean(null_vals >= a)) for a in aucs
    ]
    summary = {
        "observed_mean_pooled_auc": float(np.mean(aucs)),
        "observed_sd_pooled_auc": float(np.std(aucs, ddof=1)),
        "observed_min": float(np.min(aucs)), "observed_max": float(np.max(aucs)),
        "observed_mean_fold_mean": float(rep["fold_mean"].mean()),
        "observed_sd_fold_mean": float(rep["fold_mean"].std(ddof=1)),
        "frac_repeats_auc_gt_0.5": float(np.mean(aucs > 0.5)),
        "frac_repeats_auc_gt_hand": float(np.mean(aucs > hand_auc)),
        "frac_repeats_auc_gt_tis": float(np.mean(aucs > tis_auc)),
        "hand_pooled_auc": hand_auc, "tis_hk_pooled_auc": tis_auc,
        "median_delong_vs_hand": float(rep["delong_vs_hand"].median()),
        "median_delong_vs_tis": float(rep["delong_vs_tis"].median()),
        "median_perm_p_labelperm": float(rep["perm_p_labelperm"].median()),
        "median_perm_p_fullrefit": float(rep["perm_p_fullrefit"].median()),
        "null_mean": float(np.mean(null_vals)), "null_sd": float(np.std(null_vals)),
        "null_max": float(np.max(null_vals)), "n_repeats": args.n_repeats,
        "n_perm_fullrefit": args.n_perm,
    }
    summ = pd.DataFrame(list(summary.items()), columns=["stat", "value"])

    save_csv(rep, results_path(cfg, "aim8_repeated_cv.csv"), index=False)
    save_csv(summ, results_path(cfg, "aim8_repeated_cv_summary.csv"), index=False)
    save_csv(nulls, results_path(cfg, "aim8_perm_null.csv"), index=False)
    logger.info("\n" + summ.to_string(index=False))


if __name__ == "__main__":
    main()

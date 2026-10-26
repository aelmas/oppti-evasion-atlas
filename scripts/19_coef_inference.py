#!/usr/bin/env python
"""
Step 19 — Inference on the learned module coefficients
======================================================
Reviewer follow-up to Results §5 ("the learned weights are interpretable"):

1. **Bootstrap CIs on the learned module coefficients.** Nonparametric
   percentile bootstrap (2,000 resamples of the IMvigor210 development
   cohort) of the frozen-model logistic refit at the CV-selected C = 3.
   Reported on BOTH scales:
     raw   : log-odds per unit module feature (per pt_frac-weighted outlier
             event) — the scale used in the manuscript weight table; and
     std   : standardized "per-SD" scale (raw coef × SD of the feature),
             comparable across modules.

2. **Spearman correlation matrix of the eight module features** over the
   298 development samples, documenting the (low) collinearity among the
   modules that share outlier events.

Outputs
  results/aim9_module_coef_bootstrap.csv : per-module coef + percentile CIs
  results/aim9_module_coef_summary.json  : max |coef over folds| sanity, etc.
  results/aim9_module_corr_spearman.csv  : 8×8 Spearman correlation matrix

Usage:  python scripts/19_coef_inference.py [--n-boot 2000] [--C 3.0]
"""

import argparse
import json

import numpy as np
import pandas as pd

from oppti_immune.pies_dss import (
    MODULE_ORDER,
    fit_dss,
    load_cohort,
    madz_outliers,
    module_features,
)
from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("19_coef_inference")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--C", type=float, default=3.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gene_panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    z_thr = cfg.get("concordance", {}).get("outlier_z_threshold", 1.645)

    rna, y = load_cohort("IMvigor210", cfg)
    outliers = madz_outliers(rna, gene_panel["gene"].tolist(), z_threshold=z_thr)
    feats = module_features(outliers, gene_panel, pt_frac)
    y_nr = (1 - y.astype(int)).values
    n, p = feats.shape
    X = feats[MODULE_ORDER].values
    feat_sd = feats[MODULE_ORDER].std(ddof=0).values

    # ── point estimate: frozen-model refit on the full cohort ──
    model = fit_dss(feats[MODULE_ORDER], y_nr, args.C)
    coef = model.coef_.ravel()
    logger.info("point estimates (raw): "
                + ", ".join(f"{m}={c:+.4f}" for m, c in zip(MODULE_ORDER, coef)))

    # ── percentile bootstrap ──
    rng = np.random.default_rng(0)
    draws = []
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_nr[idx])) < 2:
            continue
        m = fit_dss(feats[MODULE_ORDER].iloc[idx], y_nr[idx], args.C)
        draws.append(m.coef_.ravel())
    draws = np.asarray(draws)
    lo_b, hi_b = np.percentile(draws, [2.5, 97.5], axis=0)

    rows = []
    for j, m in enumerate(MODULE_ORDER):
        rows.append({
            "module": m,
            "coef_raw": float(coef[j]),
            "ci_lo_raw": float(lo_b[j]), "ci_hi_raw": float(hi_b[j]),
            "feat_sd": float(feat_sd[j]),
            "coef_std": float(coef[j] * feat_sd[j]),
            "ci_lo_std": float(lo_b[j] * feat_sd[j]),
            "ci_hi_std": float(hi_b[j] * feat_sd[j]),
            "n_bootstrap_valid": int(len(draws)),
        })
    coefs = pd.DataFrame(rows)
    save_csv(coefs, results_path(cfg, "aim9_module_coef_bootstrap.csv"), index=False)

    summary = {
        "n_boot_requested": args.n_boot,
        "n_boot_valid": int(len(draws)),
        "C": args.C,
        "n_samples": n,
        "modules": p,
        "includes_zero": {m: bool(lo_b[j] <= 0 <= hi_b[j])
                          for j, m in enumerate(MODULE_ORDER)},
    }
    with open(results_path(cfg, "aim9_module_coef_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── module-feature correlation (Spearman) ──
    corr = feats[MODULE_ORDER].corr(method="spearman")
    save_csv(corr, results_path(cfg, "aim9_module_corr_spearman.csv"), index=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    off = corr.values[mask]
    logger.info("module corr: max |rho| = %.3f, mean |rho| = %.3f",
                np.abs(off).max(), np.abs(off).mean())

    logger.info("\n" + coefs.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Step 13: Synthetic-recovery benchmark for the Python OPPTI implementation
=========================================================================

The published OPPTI R package returned all-missing scores on our preprocessed
(MAD-normalized) input under multiple API configurations, so a head-to-head
numerical comparison was not possible. To validate that the Python
re-implementation behaves as an outlier detector, we run a synthetic-recovery
benchmark:

  1. Build a synthetic cohort with a realistic correlation structure
     (latent factors + noise), so that each "protein" is predictable from
     its neighbours — the regime OPPTI is designed for.
  2. Inject known single-sample outliers into random (sample, protein) cells
     by adding a spike of magnitude `spike` (in SD units).
  3. Run OPPTI and measure recovery: ROC-AUC of the continuous Δ score against
     the injected-outlier ground truth, plus precision/recall of the Q95 call.

A correct outlier detector should achieve high AUC for spikes well above the
Q95 threshold and degrade gracefully as the spike shrinks.

Usage:  python scripts/13_oppti_benchmark.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from oppti_immune.oppti_core import run_oppti_cohort
from oppti_immune.utils import get_logger, results_path, save_csv, load_config

logger = get_logger("13_oppti_benchmark")


def make_synthetic_cohort(n_samples=150, n_proteins=120, n_factors=8,
                          noise=0.4, seed=0):
    """Latent-factor expression matrix: each protein is a sparse linear mix of
    shared factors plus noise, giving a predictable neighbour structure."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n_samples, n_factors))            # sample factor scores
    L = rng.standard_normal((n_factors, n_proteins)) * (rng.random((n_factors, n_proteins)) < 0.5)
    X = F @ L + noise * rng.standard_normal((n_samples, n_proteins))
    # standardize per protein (matches MAD-ish preprocessing scale)
    X = (X - X.mean(0)) / X.std(0)
    cols = [f"P{j}" for j in range(n_proteins)]
    return pd.DataFrame(X, columns=cols), rng


def inject_outliers(df, rng, n_outliers=300, spike=4.0):
    """Add a +spike (SD units) bump to n_outliers random (sample, protein) cells."""
    X = df.values.copy()
    ns, npr = X.shape
    truth = np.zeros_like(X, dtype=int)
    idx = rng.choice(ns * npr, size=n_outliers, replace=False)
    rows, cols = np.unravel_index(idx, X.shape)
    X[rows, cols] += spike
    truth[rows, cols] = 1
    return pd.DataFrame(X, columns=df.columns, index=df.index), truth


def main():
    cfg = load_config()
    rows = []
    base, rng = make_synthetic_cohort(seed=0)

    for spike in (6.0, 4.0, 3.0, 2.0, 1.0):
        Xo, truth = inject_outliers(base, rng, n_outliers=300, spike=spike)
        scores, binary, _perm = run_oppti_cohort(
            Xo, target_proteins=list(Xo.columns), k=5, n_perm=0,
            outlier_pctile=95, seed=42,
        )
        s = scores.values.ravel()
        t = truth.ravel()
        ok = ~np.isnan(s)
        auc = roc_auc_score(t[ok], s[ok])
        b = binary.values.ravel()[ok]
        prec = precision_score(t[ok], b, zero_division=0)
        rec  = recall_score(t[ok], b, zero_division=0)
        rows.append({
            "spike_SD": spike,
            "recovery_auc": round(float(auc), 3),
            "Q95_precision": round(float(prec), 3),
            "Q95_recall": round(float(rec), 3),
        })
        logger.info(f"spike={spike} SD → recovery AUC={auc:.3f}, "
                    f"Q95 precision={prec:.3f}, recall={rec:.3f}")

    df = pd.DataFrame(rows)
    save_csv(df, results_path(cfg, "oppti_synthetic_benchmark.csv"), index=False)
    logger.info("\n" + df.to_string(index=False))
    logger.info("\nInterpretation: a correct detector gives near-1.0 recovery AUC "
                "for large spikes and degrades smoothly toward 0.5 as spikes "
                "approach the noise floor.")


if __name__ == "__main__":
    main()

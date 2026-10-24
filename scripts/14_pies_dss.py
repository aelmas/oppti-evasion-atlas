#!/usr/bin/env python
"""
Step 14: PIES-DSS — nested-CV learned module weights (Decision Support System)
=============================================================================
Replaces the hand-set PIES module weights with L2 logistic-regression weights
fitted inside nested cross-validation on the IMvigor210 development cohort,
then applies them to the held-out Hugo_2016 / Riaz_2017 cohorts.

Outputs:
  results/aim4_pies_dss_nested_cv.csv
  results/aim4_pies_dss_weights.csv
  results/aim4_pies_dss_heldout.csv
  results/aim4_pies_dss_comparison.csv

Usage:  python scripts/14_pies_dss.py [--n-perm 1000]
"""

import argparse

from oppti_immune.pies_dss import run
from oppti_immune.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg, n_perm=args.n_perm, seed=args.seed)


if __name__ == "__main__":
    main()

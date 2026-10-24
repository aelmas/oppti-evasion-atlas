#!/usr/bin/env python
"""
Step 2: Preprocessing & Normalization
=====================================
Loads raw CPTAC data, filters tumor samples, removes high-missingness proteins,
MAD-normalizes, subsets to the IEP panel, and assembles pan-cancer matrices.

Outputs:
  data/processed/pan_prot_iep.parquet
  data/processed/pan_rna_iep.parquet
  data/processed/pan_prot_full.parquet
  data/processed/pan_rna_full.parquet
  results/preprocessing_summary.csv

Usage:  python scripts/02_preprocess.py
"""

import argparse

from oppti_immune.preprocessing import run
from oppti_immune.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()

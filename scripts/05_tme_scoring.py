#!/usr/bin/env python
"""
Step 5: TME Immune Signature Scoring
=====================================
Estimates immune cell abundance from MAD-normalized bulk RNA using
curated gene signatures (mean Z-score aggregation), then correlates
each IEP outlier protein with immune-cell infiltration.

This is a Python replacement for 05_tme_deconvolution.R (which requires
raw TPM for EPIC; our RNA data is already MAD-normalized).

Outputs:
  results/aim3_tme_signature_scores.csv
  results/aim3_protein_immune_correlation.csv

Usage:  python scripts/05_tme_scoring.py [--config config/config.yaml]
"""

import argparse

import pandas as pd

from oppti_immune.tme_scoring import run as run_tme
from oppti_immune.utils import (
    data_path, load_config, results_path, get_logger,
)

logger = get_logger("05_tme_scoring")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Load pan-cancer full RNA (MAD-normalized, samples × genes)
    rna_path = data_path(cfg, "processed/pan_rna_full.parquet")
    logger.info(f"Loading RNA from {rna_path} ...")
    rna = pd.read_parquet(rna_path)
    if "sample_id" in rna.columns:
        rna = rna.set_index("sample_id")
    logger.info(f"RNA matrix: {rna.shape[0]} samples × {rna.shape[1]} genes")

    # Load OPPTI binary outlier matrix
    ob_path = results_path(cfg, "oppti_binary_matrix.parquet")
    logger.info(f"Loading outlier binary matrix from {ob_path} ...")
    outlier_binary = pd.read_parquet(ob_path)
    for col in ("sample", "sample_id"):
        if col in outlier_binary.columns:
            outlier_binary = outlier_binary.set_index(col)
            break
    outlier_binary = outlier_binary.fillna(0).astype(int)
    logger.info(f"Outlier binary: {outlier_binary.shape[0]} samples × {outlier_binary.shape[1]} proteins")

    # Run TME scoring + correlation
    run_tme(cfg, rna, outlier_binary)

    logger.info("Step 5 complete.")


if __name__ == "__main__":
    main()

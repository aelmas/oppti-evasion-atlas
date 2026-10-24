#!/usr/bin/env python
"""
Step 6: Build the Protein-level Immune Evasion Score (PIES)
===========================================================
Computes PIES for all CPTAC samples and trains RNA→Protein imputation models
for use with RNA-only validation cohorts.

Outputs:
  results/aim4_PIES_scores_cptac.csv
  results/oppti_protein_p95_cutoffs.csv
  models/rna_to_protein_imputation.pkl

Usage:  python scripts/06_build_pies.py
"""

import argparse
import pickle

import pandas as pd

from oppti_immune.pies import (
    compute_pies_all_samples,
    train_imputation_models,
)
from oppti_immune.utils import (
    data_path, load_config, load_gene_panel, models_path,
    results_path, save_csv, get_logger,
)

logger = get_logger("06_build_pies")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gene_panel = load_gene_panel(cfg)
    panel_genes = gene_panel["gene"].tolist()

    # Load OPPTI binary matrix
    ob_path = results_path(cfg, "oppti_binary_matrix.parquet")
    outlier_binary = pd.read_parquet(ob_path)
    for col in ("sample", "sample_id"):
        if col in outlier_binary.columns:
            outlier_binary = outlier_binary.set_index(col)
            break
    outlier_binary = outlier_binary.fillna(0)

    # Concordance summary → post-transcriptional fraction per gene
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_fraction = pt.groupby("gene")["pct_post_transcriptional"].mean()

    # CD8 correlation (from Step 5)
    tme_path = results_path(cfg, "aim3_protein_immune_correlation.csv")
    if tme_path.exists():
        tme = pd.read_csv(tme_path)
        cd8 = tme[tme["immune_metric"] == "CD8+ T-cells"].set_index("gene")["rho"]
    else:
        logger.warning("TME correlation file missing — using zero CD8 correlations.")
        cd8 = pd.Series(dtype=float)

    # Compute PIES
    pies = compute_pies_all_samples(outlier_binary, gene_panel, pt_fraction, cd8, cfg)
    save_csv(pies.to_frame(), results_path(cfg, "aim4_PIES_scores_cptac.csv"))
    logger.info(f"PIES computed for {len(pies)} samples. "
                f"Range: [{pies.min():.2f}, {pies.max():.2f}]")

    # Train RNA→Protein imputation models
    logger.info("Training RNA→Protein imputation models on CPTAC...")
    pan_prot = pd.read_parquet(data_path(cfg, "processed/pan_prot_iep.parquet"))
    if "sample_id" in pan_prot.columns:
        pan_prot = pan_prot.set_index("sample_id")
    pan_rna  = pd.read_parquet(data_path(cfg, "processed/pan_rna_full.parquet"))
    if "sample_id" in pan_rna.columns:
        pan_rna = pan_rna.set_index("sample_id")

    models = train_imputation_models(pan_prot, pan_rna, panel_genes, cfg)
    model_out = models_path(cfg, "rna_to_protein_imputation.pkl")
    with open(model_out, "wb") as f:
        pickle.dump(models, f)
    logger.info(f"Saved {len(models)} imputation models → {model_out}")

    # Save p95 cutoffs (per-gene, for outlier thresholding in ICB cohorts)
    prot_no_label = pan_prot.drop(columns=["cancer_type"], errors="ignore")
    cutoffs = prot_no_label.quantile(cfg["oppti"]["outlier_percentile"] / 100.0)
    save_csv(cutoffs.to_frame("p95"),
             results_path(cfg, "oppti_protein_p95_cutoffs.csv"))

    logger.info("Step 6 complete.")


if __name__ == "__main__":
    main()

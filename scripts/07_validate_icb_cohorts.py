#!/usr/bin/env python
"""
Step 7: Validate PIES in Immunotherapy Cohorts
==============================================
For each RNA-only ICB cohort:
  1. Impute protein levels using CPTAC-trained ElasticNet models
  2. Compute PIES
  3. Evaluate response prediction (AUC) and OS stratification (Kaplan–Meier)
  4. Compare to TIS score baseline

Outputs:
  results/aim4_AUC_by_cohort.csv
  figures/aim4_KM_<cohort>.pdf

Usage:  python scripts/07_validate_icb_cohorts.py
"""

import argparse
import pickle

import pandas as pd

from oppti_immune.pies import validate_in_cohort
from oppti_immune.utils import (
    load_config, load_gene_panel, models_path, results_path, save_csv, get_logger,
)

logger = get_logger("07_validate_icb")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gene_panel = load_gene_panel(cfg)

    # Load imputation models + cutoffs
    with open(models_path(cfg, "rna_to_protein_imputation.pkl"), "rb") as f:
        models = pickle.load(f)
    cutoffs_df = pd.read_csv(results_path(cfg, "oppti_protein_p95_cutoffs.csv"), index_col=0)
    cutoffs = cutoffs_df.iloc[:, 0]  # first column contains the p95 values

    # Load concordance + TME data for PIES weighting
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_fraction = pt.groupby("gene")["pct_post_transcriptional"].mean()

    tme_path = results_path(cfg, "aim3_protein_immune_correlation.csv")
    if tme_path.exists():
        tme = pd.read_csv(tme_path)
        cd8 = tme[tme["immune_metric"] == "CD8+ T-cells"].set_index("gene")["rho"]
    else:
        cd8 = pd.Series(dtype=float)

    # Validate in each cohort
    all_results = []
    for name, cohort_cfg in cfg.get("immunotherapy_cohorts", {}).items():
        logger.info(f"\n▶ Validating in {name} "
                    f"({cohort_cfg.get('cancer')}, {cohort_cfg.get('treatment')})...")
        res = validate_in_cohort(
            name, cohort_cfg, models, cutoffs,
            gene_panel, pt_fraction, cd8, cfg,
        )
        if res:
            all_results.append(res)

    df = pd.DataFrame(all_results)
    save_csv(df, results_path(cfg, "aim4_AUC_by_cohort.csv"), index=False)

    if not df.empty:
        logger.info("\n══════════════════════════════════════════════════════════════")
        logger.info(" ICB Cohort Validation Summary")
        logger.info("══════════════════════════════════════════════════════════════")
        logger.info(f"\n{df.to_string()}")


if __name__ == "__main__":
    main()

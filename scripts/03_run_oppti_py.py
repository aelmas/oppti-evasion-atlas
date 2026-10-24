#!/usr/bin/env python
"""
Step 3 (Python): Run OPPTI on all cancer types using the native Python implementation.

Generates the same outputs as scripts/03_run_oppti.R but runs much faster
because it uses the closed-form LOO ridge regression (no slow R package).

Outputs (compatible with downstream Python steps):
  results/oppti/<cancer>_scores.csv
  results/oppti/<cancer>_outlier_binary.csv
  results/oppti_binary_matrix.parquet
  results/oppti_scores_matrix.parquet
  results/aim1_pan_cancer_oppti_summary.csv
  results/aim1_recurrent_IEP_outliers.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

from oppti_immune.oppti_core import run_pan_cancer
from oppti_immune.utils import (
    data_path, get_logger, load_config, load_gene_panel,
    results_path, save_csv, save_parquet,
)

logger = get_logger("03_run_oppti_py")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run OPPTI pan-cancer analysis (Python)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-perm", type=int, default=None,
                    help="Override n_permutations from config (use 0 to skip)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = load_gene_panel(cfg)
    panel_genes = panel["gene"].tolist()

    n_perm = args.n_perm if args.n_perm is not None else cfg["oppti"]["n_permutations"]
    k      = cfg["oppti"]["k_neighbors"]
    pctile = cfg["oppti"]["outlier_percentile"]
    seed   = cfg.get("seed", 42)

    logger.info(f"OPPTI config: k={k}, n_perm={n_perm}, outlier_pctile={pctile}")

    # Load pan-cancer IEP proteomics (samples × proteins + cancer_type)
    pan_prot_path = data_path(cfg, "processed/pan_prot_iep.parquet")
    if not pan_prot_path.exists():
        logger.error(f"IEP proteomics not found: {pan_prot_path}. Run 02_preprocess.py first.")
        sys.exit(1)

    pan_prot = pd.read_parquet(pan_prot_path)
    if "sample_id" in pan_prot.columns:
        pan_prot = pan_prot.set_index("sample_id")

    logger.info(f"Loaded pan-cancer IEP: {pan_prot.shape[0]} samples × "
                f"{pan_prot.shape[1]-1} proteins + cancer_type")

    # Run pan-cancer OPPTI
    pan_scores, pan_binary, summary, pan_pvals = run_pan_cancer(
        pan_prot, panel_genes,
        k=k, n_perm=n_perm,
        outlier_pctile=pctile, seed=seed,
    )

    # Per-cancer CSV exports (matches R script outputs)
    oppti_dir = results_path(cfg, "oppti")
    oppti_dir.mkdir(parents=True, exist_ok=True)

    cancer_labels = pan_prot["cancer_type"]
    for ct in cancer_labels.unique():
        ct_mask   = cancer_labels == ct
        ct_scores = pan_scores[ct_mask].T   # proteins × samples (R convention)
        ct_binary = pan_binary[ct_mask].T

        ct_scores.to_csv(oppti_dir / f"{ct}_scores.csv")
        ct_binary.to_csv(oppti_dir / f"{ct}_outlier_binary.csv")
        logger.info(f"  {ct}: {ct_scores.shape[0]} proteins, "
                    f"{ct_binary.values.sum()} outlier calls")

    # Pan-cancer binary & scores matrices
    save_parquet(pan_binary.reset_index(names="sample"),
                 results_path(cfg, "oppti_binary_matrix.parquet"))
    save_parquet(pan_scores.reset_index(names="sample"),
                 results_path(cfg, "oppti_scores_matrix.parquet"))

    # Save permutation p-values if computed
    perm_pvals = pan_pvals
    if perm_pvals is not None:
        save_parquet(perm_pvals.reset_index(names="sample"),
                     results_path(cfg, "oppti_perm_pvals.parquet"))
        # Per-call BH FDR across all (sample × protein) tested values
        from statsmodels.stats.multitest import multipletests
        flat = perm_pvals.values.ravel()
        ok = ~np.isnan(flat)
        fdr = np.full_like(flat, np.nan)
        if ok.sum() > 0:
            _, fdr[ok], _, _ = multipletests(flat[ok], method="fdr_bh")
        fdr_df = pd.DataFrame(fdr.reshape(perm_pvals.shape),
                              index=perm_pvals.index, columns=perm_pvals.columns)
        save_parquet(fdr_df.reset_index(names="sample"),
                     results_path(cfg, "oppti_perm_fdr.parquet"))

        n_calls   = int((pan_binary.values == 1).sum())
        n_fdr_sig = int(((pan_binary.values == 1) & (fdr_df.values < 0.05)).sum())
        logger.info(f"Permutation FDR: {n_fdr_sig}/{n_calls} calls pass BH-FDR < 0.05 "
                    f"({n_fdr_sig/max(n_calls,1)*100:.1f}%)")

    # Summary (AIM 1)
    save_csv(summary, results_path(cfg, "aim1_pan_cancer_oppti_summary.csv"), index=False)

    # Recurrent IEP outliers: proteins outlier in ≥ min_cancer_types with ≥ min_outlier_frac
    min_frac  = cfg["pies"]["min_outlier_fraction"] * 100     # convert to %
    min_ct    = cfg["pies"]["min_cancer_types"]

    recurrent = (
        summary[summary["pct_outlier"] >= min_frac]
        .groupby("protein")
        .agg(
            n_cancer_types=("cancer_type", "nunique"),
            mean_pct=("pct_outlier", "mean"),
            cancers=("cancer_type", lambda x: ",".join(sorted(x.unique()))),
        )
        .reset_index()
        .query(f"n_cancer_types >= {min_ct}")
        .sort_values(["n_cancer_types", "mean_pct"], ascending=False)
    )

    # Add module information
    module_map = panel.set_index("gene")["module"]
    recurrent["modules"] = recurrent["protein"].map(
        lambda g: module_map.get(g, "unknown")
    )

    save_csv(recurrent, results_path(cfg, "aim1_recurrent_IEP_outliers.csv"), index=False)

    logger.info("\n══════════════════════════════════════════════════════════════")
    logger.info(f" OPPTI complete. {len(recurrent)} recurrent IEP outlier proteins.")
    logger.info("══════════════════════════════════════════════════════════════")
    if not recurrent.empty:
        logger.info(f"\n{recurrent.head(20).to_string(index=False)}")

    total_calls = int(pan_binary.values.sum())
    logger.info(f"\nTotal pan-cancer outlier calls: {total_calls:,}")
    logger.info(f"Samples with ≥1 outlier: "
                f"{(pan_binary.sum(axis=1) > 0).sum():,}/{len(pan_binary):,}")


if __name__ == "__main__":
    main()

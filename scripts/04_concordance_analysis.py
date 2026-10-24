#!/usr/bin/env python
"""
Step 4: RNA–Protein Concordance Analysis
========================================
Classifies every outlier call as RNA-concordant or post-transcriptional.
Tests candidate mechanisms: CNV amplification, phospho-stabilization.

Outputs:
  results/aim2_concordance_classification.parquet
  results/aim2_post_transcriptional_fraction.csv
  results/aim2_cnv_mechanism.csv
  results/aim2_phospho_stabilization.csv

Usage:  python scripts/04_concordance_analysis.py
"""

import argparse

import pandas as pd

from oppti_immune.concordance import (
    compute_post_transcriptional_fraction,
    classify_using_oppti_calls,
    test_cnv_mechanism,
    test_phospho_stabilization,
)
from oppti_immune.utils import (
    data_path, load_config, load_gene_panel, results_path, save_csv, save_parquet, get_logger,
)

logger = get_logger("04_concordance")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = load_gene_panel(cfg)
    panel_genes = panel["gene"].tolist()

    # Load pan-cancer matrices; set sample_id as index (saved as column for R compat)
    pan_prot = pd.read_parquet(data_path(cfg, "processed/pan_prot_iep.parquet"))
    if "sample_id" in pan_prot.columns:
        pan_prot = pan_prot.set_index("sample_id")

    pan_rna  = pd.read_parquet(data_path(cfg, "processed/pan_rna_iep.parquet"))
    if "sample_id" in pan_rna.columns:
        pan_rna = pan_rna.set_index("sample_id")

    # Propagate cancer_type to RNA (using proteomics labels for matched samples)
    if "cancer_type" in pan_prot.columns and "cancer_type" not in pan_rna.columns:
        pan_rna = pan_rna.join(pan_prot[["cancer_type"]], how="left")

    # 1) Per-sample classification — restricted to the ACTUAL OPPTI Q95 call
    #    set (peer-review fix: the headline "X% of outliers are
    #    post-transcriptional" must use OPPTI's own call population as the
    #    denominator, not an independently recomputed protein Z-threshold).
    logger.info("Classifying concordance restricted to OPPTI Q95 outlier calls...")
    ob = pd.read_parquet(results_path(cfg, "oppti_binary_matrix.parquet"))
    if "sample" in ob.columns:
        ob = ob.set_index("sample")
    if "cancer_type" in ob.columns:
        ob = ob.drop(columns=["cancer_type"])
    ob   = ob.reindex(pan_prot.index).fillna(0).astype(int)
    genes_oppti = ob.columns.tolist()

    perm_p_path = results_path(cfg, "oppti_perm_pvals.parquet")
    if perm_p_path.exists():
        pp = pd.read_parquet(perm_p_path)
        if "sample" in pp.columns:
            pp = pp.set_index("sample")
        pp = pp.reindex(index=ob.index, columns=genes_oppti)
        perm_sig = (pp < 0.05) & (ob == 1)
        logger.info(f"OPPTI Q95 calls: {int(ob.values.sum()):,} | "
                    f"permutation-significant (p<0.05): {int(perm_sig.values.sum()):,}")
    else:
        perm_sig = None

    conc = classify_using_oppti_calls(
        pan_prot[genes_oppti], pan_rna.reindex(columns=genes_oppti),
        ob, cfg["concordance"]["outlier_z_threshold"],
        perm_mask=None,
    )
    save_parquet(conc, results_path(cfg, "aim2_concordance_classification.parquet"))

    # 1b) Same classification restricted to permutation-significant calls only
    #     (robustness subset for the manuscript).
    if perm_sig is not None:
        conc_sig = classify_using_oppti_calls(
            pan_prot[genes_oppti], pan_rna.reindex(columns=genes_oppti),
            ob, cfg["concordance"]["outlier_z_threshold"],
            perm_mask=perm_sig,
        )
        save_parquet(conc_sig, results_path(cfg, "aim2_concordance_perm_significant.parquet"))
        pt_sig = conc_sig["concordance"].value_counts()
        pt_n = pt_sig.get("post_transcriptional", 0)
        tot  = pt_n + pt_sig.get("concordant", 0)
        logger.info(f"Perm-significant subset: {pt_n:,}/{tot:,} ({pt_n/max(tot,1)*100:.1f}%) "
                    f"post-transcriptional.")

    # 2) Per-gene post-transcriptional fraction (canonical PIES pt_frac input)
    logger.info("Computing post-transcriptional fraction per gene...")
    pt = compute_post_transcriptional_fraction(conc)
    save_csv(pt, results_path(cfg, "aim2_post_transcriptional_fraction.csv"), index=False)

    overall_pt = (conc["concordance"] == "post_transcriptional").sum()
    overall_total = conc["concordance"].isin(["concordant", "post_transcriptional"]).sum()
    logger.info(
        f"Headline: {overall_pt:,}/{overall_total:,} "
        f"({overall_pt/max(overall_total,1)*100:.1f}%) of IEP outliers are "
        f"post-transcriptional (RNA-invisible)."
    )

    # 3) CNV mechanism
    pt_genes = pt[pt["pct_post_transcriptional"] > 50]["gene"].unique().tolist()
    logger.info(f"Testing CNV mechanism for {len(pt_genes)} genes...")
    cnv_df = test_cnv_mechanism(pt_genes, cfg)
    save_csv(cnv_df, results_path(cfg, "aim2_cnv_mechanism.csv"), index=False)

    # 4) Phospho-stabilization mechanism
    outlier_path = results_path(cfg, "oppti_binary_matrix.parquet")
    if outlier_path.exists():
        logger.info("Testing phospho-stabilization mechanism...")
        ob = pd.read_parquet(outlier_path)
        if "sample" in ob.columns:
            ob = ob.set_index("sample")
        phos_df = test_phospho_stabilization(ob, cfg)
        save_csv(phos_df, results_path(cfg, "aim2_phospho_stabilization.csv"), index=False)
    else:
        logger.warning("OPPTI binary matrix not found; skipping phospho analysis.")

    logger.info("Step 4 complete.")


if __name__ == "__main__":
    main()

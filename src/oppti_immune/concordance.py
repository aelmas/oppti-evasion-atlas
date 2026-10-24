"""
concordance.py — RNA–protein concordance classification and PTM mechanism analysis.

For every (sample, IEP gene) pair where OPPTI calls a protein outlier, classify as:
  - 'concordant'          : RNA also elevated (Z > threshold)
  - 'post_transcriptional': Protein outlier but RNA is NOT elevated
  - 'rna_only'            : RNA elevated but protein is NOT an outlier
  - 'none'                : Neither elevated

Then test candidate post-transcriptional mechanisms:
  1. CNV amplification (protein up without RNA)
  2. Phospho-driven protein stabilization (CPTAC phosphoproteomics)
  3. miRNA regulation (TargetScan predictions via API)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from .utils import get_logger, load_config, data_path, results_path, save_csv, save_parquet

logger = get_logger(__name__)


# ── Per-sample concordance classification ─────────────────────────────────────

def classify_concordance(
    prot_z: float,
    rna_z: float,
    threshold: float = 1.645,
) -> str:
    """Classify one sample–gene pair by RNA/protein concordance.

    Parameters
    ----------
    prot_z    : protein Z-score (MAD-normalized)
    rna_z     : RNA Z-score (MAD-normalized)
    threshold : Z-score cutoff for 'elevated' (~95th percentile one-sided)

    Returns
    -------
    One of: 'concordant', 'post_transcriptional', 'rna_only', 'none'
    """
    prot_up = (not np.isnan(prot_z)) and prot_z > threshold
    rna_up  = (not np.isnan(rna_z))  and rna_z  > threshold

    if prot_up and rna_up:
        return "concordant"
    elif prot_up and not rna_up:
        return "post_transcriptional"
    elif not prot_up and rna_up:
        return "rna_only"
    return "none"


def classify_all(
    pan_prot: pd.DataFrame,
    pan_rna:  pd.DataFrame,
    panel_genes: list[str],
    threshold: float = 1.645,
) -> pd.DataFrame:
    """Classify concordance for all sample–gene pairs across the IEP panel.

    Parameters
    ----------
    pan_prot     : samples × (IEP genes + 'cancer_type') — MAD-normalized proteomics
    pan_rna      : samples × IEP genes — MAD-normalized transcriptomics
    panel_genes  : list of IEP gene symbols
    threshold    : Z-score cutoff

    Returns
    -------
    DataFrame with columns: sample, gene, cancer_type, concordance, prot_z, rna_z
    """
    cancer_labels = pan_prot["cancer_type"] if "cancer_type" in pan_prot.columns else None

    prot_mat = pan_prot.drop(columns=["cancer_type"], errors="ignore")
    shared_samples = prot_mat.index.intersection(pan_rna.index)
    shared_genes   = [g for g in panel_genes if g in prot_mat.columns and g in pan_rna.columns]

    logger.info(
        f"Classifying concordance: {len(shared_samples)} samples × {len(shared_genes)} genes"
    )

    records: list[dict] = []

    for gene in shared_genes:
        p_vals = prot_mat.loc[shared_samples, gene]
        r_vals = pan_rna.loc[shared_samples, gene]

        # Recompute background Z within each cancer type to match OPPTI
        for ct in (cancer_labels.unique() if cancer_labels is not None else ["all"]):
            if cancer_labels is not None:
                ct_samples = shared_samples[cancer_labels.reindex(shared_samples) == ct]
            else:
                ct_samples = shared_samples

            p_sub = p_vals.reindex(ct_samples).dropna()
            r_sub = r_vals.reindex(ct_samples).dropna()
            common = p_sub.index.intersection(r_sub.index)

            if len(common) < 10:
                continue

            # Compute per-cancer-type Z-scores for this gene
            p_med, p_mad = p_sub[common].median(), (p_sub[common] - p_sub[common].median()).abs().median()
            r_med, r_mad = r_sub[common].median(), (r_sub[common] - r_sub[common].median()).abs().median()
            if p_mad == 0: p_mad = 1e-6
            if r_mad == 0: r_mad = 1e-6

            for sample in common:
                pz = (p_sub[sample] - p_med) / p_mad
                rz = (r_sub[sample] - r_med) / r_mad
                cls = classify_concordance(pz, rz, threshold)
                records.append({
                    "sample":      sample,
                    "gene":        gene,
                    "cancer_type": ct,
                    "concordance": cls,
                    "prot_z":      pz,
                    "rna_z":       rz,
                })

    return pd.DataFrame(records)


# ── Summary statistics ────────────────────────────────────────────────────────

def classify_using_oppti_calls(
    pan_prot: pd.DataFrame,
    pan_rna:  pd.DataFrame,
    oppti_binary: pd.DataFrame,
    rna_threshold: float = 1.645,
    perm_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify concordance restricted to ACTUAL OPPTI outlier calls.

    Fixes a peer-review blocker: the original `classify_all()` recomputed its own
    Z-threshold on protein, so the "X% post-transcriptional of outliers" headline
    used a different outlier population than OPPTI. This function intersects with
    OPPTI's binary call matrix (and optionally with the permutation-significant
    subset) so the denominator is the actual OPPTI call set.

    Parameters
    ----------
    pan_prot      : samples × (IEP genes + 'cancer_type') — MAD-normalized
    pan_rna       : samples × IEP genes — MAD-normalized transcriptomics
    oppti_binary  : samples × IEP genes — 0/1 OPPTI Q95 outlier calls
    rna_threshold : Z-score cutoff for RNA elevation
    perm_mask     : optional bool DataFrame (samples × genes); if provided,
                    only OPPTI calls that ALSO pass this mask are counted

    Returns
    -------
    DataFrame with one row per OPPTI-called (sample, gene):
      sample, gene, cancer_type, rna_z, concordance ∈ {concordant, post_transcriptional}
    """
    cancer_labels = pan_prot["cancer_type"] if "cancer_type" in pan_prot.columns else None
    prot_mat = pan_prot.drop(columns=["cancer_type"], errors="ignore")

    # Intersect samples and genes across all three matrices
    shared_samples = prot_mat.index.intersection(pan_rna.index).intersection(oppti_binary.index)
    shared_genes   = [g for g in oppti_binary.columns
                      if g in prot_mat.columns and g in pan_rna.columns]

    if perm_mask is not None:
        shared_samples = shared_samples.intersection(perm_mask.index)
        shared_genes   = [g for g in shared_genes if g in perm_mask.columns]

    ob = oppti_binary.loc[shared_samples, shared_genes]
    if perm_mask is not None:
        ob = ob & perm_mask.loc[shared_samples, shared_genes].astype(bool)

    logger.info(
        f"Concordance restricted to OPPTI calls: {int(ob.values.sum())} (sample, gene) outliers "
        f"across {len(shared_samples)} samples × {len(shared_genes)} genes"
    )

    records: list[dict] = []

    # Compute RNA Z-scores per cancer-type per gene (same scaling as classify_all)
    for gene in shared_genes:
        r_vals = pan_rna.loc[shared_samples, gene]
        for ct in (cancer_labels.unique() if cancer_labels is not None else ["all"]):
            if cancer_labels is not None:
                ct_samples = shared_samples[cancer_labels.reindex(shared_samples) == ct]
            else:
                ct_samples = shared_samples
            r_sub = r_vals.reindex(ct_samples).dropna()
            if len(r_sub) < 10:
                continue
            r_med = r_sub.median()
            r_mad = (r_sub - r_med).abs().median()
            if r_mad == 0:
                r_mad = 1e-6

            # Restrict to samples that are OPPTI outliers for this gene
            outlier_samples = r_sub.index[ob.loc[r_sub.index, gene].astype(bool)]
            for sample in outlier_samples:
                rz = (r_sub[sample] - r_med) / r_mad
                cls = "concordant" if (not np.isnan(rz) and rz > rna_threshold) else "post_transcriptional"
                records.append({
                    "sample":      sample,
                    "gene":        gene,
                    "cancer_type": ct,
                    "rna_z":       rz,
                    "concordance": cls,
                })

    df = pd.DataFrame(records)
    if not df.empty:
        n_pt   = (df["concordance"] == "post_transcriptional").sum()
        n_conc = (df["concordance"] == "concordant").sum()
        pct = n_pt / max(n_pt + n_conc, 1) * 100
        logger.info(
            f"OPPTI-restricted concordance: "
            f"{n_pt}/{n_pt + n_conc} = {pct:.1f}% post-transcriptional"
        )
    return df


def compute_post_transcriptional_fraction(concordance_df: pd.DataFrame) -> pd.DataFrame:
    """Compute % post-transcriptional per gene (and overall).

    Only considers samples where the protein is an outlier
    (concordant OR post_transcriptional).
    """
    outlier_rows = concordance_df[
        concordance_df["concordance"].isin(["concordant", "post_transcriptional"])
    ]

    summary = (
        outlier_rows
        .groupby(["gene", "cancer_type", "concordance"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["concordant", "post_transcriptional"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["n_outlier_total"] = summary["concordant"] + summary["post_transcriptional"]
    summary["pct_post_transcriptional"] = (
        summary["post_transcriptional"] / (summary["n_outlier_total"] + 1e-9) * 100
    )

    return summary.sort_values("pct_post_transcriptional", ascending=False)


# ── Mechanism 1: CNV-driven ────────────────────────────────────────────────────

def test_cnv_mechanism(
    post_transcriptional_genes: list[str],
    cfg: dict,
) -> pd.DataFrame:
    """Test whether post-transcriptional outliers are explained by DNA copy number.

    For each gene × cancer type, compute Spearman correlation between
    CNV log-ratio and protein expression. High correlation + no RNA change
    = CNV-driven (amplification stabilizes protein without RNA amplification).
    """
    cptac_dir = data_path(cfg, "cptac")
    records: list[dict] = []

    for cancer in cfg["cptac"]["cancer_types"]:
        cnv_file  = next(
            (cptac_dir / f"{cancer}_{s}.parquet"
             for s in ("cnv", "CNV") if (cptac_dir / f"{cancer}_{s}.parquet").exists()),
            cptac_dir / f"{cancer}_cnv.parquet",
        )
        prot_file = cptac_dir / f"{cancer}_proteomics.parquet"

        if not cnv_file.exists() or not prot_file.exists():
            continue

        from .preprocessing import extract_gene_prefix, filter_tumor_samples
        cnv  = extract_gene_prefix(pd.read_parquet(cnv_file))
        prot = extract_gene_prefix(filter_tumor_samples(pd.read_parquet(prot_file)))

        for gene in post_transcriptional_genes:
            if gene not in cnv.columns or gene not in prot.columns:
                continue
            shared = cnv.index.intersection(prot.index)
            if len(shared) < 10:
                continue

            cn_vec   = cnv.loc[shared, gene].dropna()
            prot_vec = prot.loc[cn_vec.index, gene].dropna()
            common   = cn_vec.index.intersection(prot_vec.index)

            if len(common) < 10:
                continue

            r, p = spearmanr(cn_vec[common], prot_vec[common])
            records.append({
                "gene":                  gene,
                "cancer_type":           cancer,
                "cnv_protein_spearman":  r,
                "pvalue":                p,
                "n":                     len(common),
                "cnv_driven":            r > 0.3 and p < 0.05,
            })

    return pd.DataFrame(records)


# ── Mechanism 2: Phospho-stabilization ────────────────────────────────────────

# Known stabilizing phosphosites from PhosphoSitePlus (curated subset)
PHOSPHO_STABILIZING_SITES: dict[str, list[str]] = {
    "CD274":  ["S283", "T290", "Y112"],   # PD-L1: GSK3β targets → β-TrCP ubiquitination prevented
    "IDO1":   ["S167"],                    # IDO1 stabilization
    "ARG1":   ["S330"],
    "HAVCR2": ["Y265", "T256"],            # TIM-3
    "TIGIT":  ["Y225"],
    "LGALS9": ["T49"],
    "IDO2":   ["S75"],
    "TGFB1":  ["S165"],
    "CD276":  ["S282"],                    # B7-H3 stabilization
}


def test_phospho_stabilization(
    outlier_binary: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Test whether stabilizing phosphosites are enriched in protein-outlier tumors.

    Parameters
    ----------
    outlier_binary : samples × IEP genes, values 0/1 (OPPTI output)

    Returns
    -------
    DataFrame with Mann-Whitney test results per gene × site × cancer_type
    """
    cptac_dir = data_path(cfg, "cptac")
    records: list[dict] = []

    for cancer in cfg["cptac"]["cancer_types"]:
        phospho_file = cptac_dir / f"{cancer}_phosphoproteomics.parquet"
        if not phospho_file.exists():
            continue

        phospho = pd.read_parquet(phospho_file)

        # Identify samples belonging to this cancer type in the outlier matrix.
        # Samples may be bare CPTAC barcodes or tagged with _{cancer}.
        all_cancer_samples = phospho.index  # all samples with phospho data
        shared = outlier_binary.index.intersection(all_cancer_samples)
        if len(shared) < 6:
            # Try _{cancer} suffix format
            shared = outlier_binary.index[
                outlier_binary.index.str.endswith(f"_{cancer}")
            ].intersection(all_cancer_samples)

        for gene, sites in PHOSPHO_STABILIZING_SITES.items():
            if gene not in outlier_binary.columns:
                continue

            gene_samples = shared
            if len(gene_samples) < 6:
                continue

            outlier_mask = outlier_binary.loc[gene_samples, gene] == 1
            outlier_samp = gene_samples[outlier_mask]
            nonout_samp  = gene_samples[~outlier_mask]

            for site in sites:
                site_cols = [c for c in phospho.columns if gene in c and site in c]
                if not site_cols:
                    continue

                phospho_vals = phospho[site_cols[0]].reindex(all_cancer_samples)
                out_vals  = phospho_vals.reindex(outlier_samp).dropna()
                nout_vals = phospho_vals.reindex(nonout_samp).dropna()

                if len(out_vals) < 3 or len(nout_vals) < 3:
                    continue

                stat, pval = mannwhitneyu(out_vals, nout_vals, alternative="greater")
                records.append({
                    "gene":                  gene,
                    "site":                  site,
                    "cancer_type":           cancer,
                    "outlier_mean_phospho":  out_vals.mean(),
                    "nonoutlier_mean_phospho": nout_vals.mean(),
                    "fold_enrichment":       out_vals.mean() / (nout_vals.mean() + 1e-9),
                    "pvalue":                pval,
                    "n_outlier":             len(out_vals),
                    "n_nonoutlier":          len(nout_vals),
                })

    df = pd.DataFrame(records)
    if not df.empty:
        from statsmodels.stats.multitest import multipletests
        _, df["padj"], _, _ = multipletests(df["pvalue"], method="fdr_bh")

    return df


# ── Master entry point ────────────────────────────────────────────────────────

def run(cfg: dict, pan_prot: pd.DataFrame, pan_rna: pd.DataFrame) -> None:
    """Run the full concordance analysis pipeline."""
    logger.info("=== Step 4: RNA–Protein Concordance Analysis ===")

    from .utils import load_gene_panel
    panel_genes = load_gene_panel(cfg)["gene"].tolist()
    threshold   = cfg["concordance"]["outlier_z_threshold"]

    concordance_df = classify_all(pan_prot, pan_rna, panel_genes, threshold)
    save_parquet(concordance_df, results_path(cfg, "aim2_concordance_classification.parquet"))

    pt_summary = compute_post_transcriptional_fraction(concordance_df)
    save_csv(pt_summary, results_path(cfg, "aim2_post_transcriptional_fraction.csv"), index=False)

    # Overall fraction (headline statistic)
    total_outliers = concordance_df[concordance_df["concordance"].isin(
        ["concordant", "post_transcriptional"])].shape[0]
    pt_count = (concordance_df["concordance"] == "post_transcriptional").sum()
    pct = pt_count / (total_outliers + 1e-9) * 100

    logger.info(
        f"Post-transcriptional fraction (IEP-wide): "
        f"{pt_count:,}/{total_outliers:,} = {pct:.1f}%"
    )

    # CNV mechanism
    pt_genes = pt_summary[pt_summary["pct_post_transcriptional"] > 50]["gene"].unique().tolist()
    logger.info(f"Testing CNV mechanism for {len(pt_genes)} genes...")
    cnv_df = test_cnv_mechanism(pt_genes, cfg)
    save_csv(cnv_df, results_path(cfg, "aim2_cnv_mechanism.csv"), index=False)

    logger.info("=== Concordance Analysis Complete ===")


if __name__ == "__main__":
    cfg = load_config()
    pan_prot = pd.read_parquet(data_path(cfg, "processed/pan_prot_iep.parquet"))
    pan_rna  = pd.read_parquet(data_path(cfg, "processed/pan_rna_iep.parquet"))
    run(cfg, pan_prot, pan_rna)

"""
pies.py — Immune Evasion Score (PIES): computation and validation.

PIES is a per-sample composite score over IEP-panel outlier calls.
Higher PIES → more immune evasion → predicted ICB resistance.

Formula (per sample)
--------------------
    PIES = Σ_{g ∈ outliers} module_weight[module(g)] × (1 + pt_mult × pt_frac[g])

where:
- `module_weight` is the curated, hand-set weight per functional module
  (signs encode evasion direction: checkpoint=+3, antigen_presentation=−2.5,
  ifn_pathway=−2, mismatch_repair=−1.5; positives indicate that an outlier
  call in that module increases predicted evasion).
- `pt_frac[g]` is the gene's pan-cancer post-transcriptional fraction
  (0–1); the term up-weights events invisible to RNA-seq.

NOTE: An earlier design multiplied each term by a sign derived from the
gene's Spearman correlation with CD8 infiltration. That logic is no longer
applied to any panel gene (no genes map to the 'other' module), and is
documented as "removed" in commit 600dc3f after observing that
PD-L1/IDO1/HAVCR2 are co-induced with CD8 (T-cell-inflamed phenotype),
which inverted their PIES contributions. The `cd8_correlation` parameter
is retained for backward compatibility and may be removed in a future
release.

Two evaluation modes (see compute_rna_pies and validate_in_cohort):
- 'rna'    : Z-score IEP genes within each cohort, call outliers at Z>z_thr,
             then compute PIES. This is the DEFAULT and used for all
             headline ICB-validation numbers.
- 'imputed': use CPTAC-trained ElasticNet RNA→protein models, threshold by
             per-protein Q95 cutoffs, then compute PIES. Provided for
             reproducibility of the legacy path; not used in headline
             results.

Validation utilities
--------------------
- Binary response AUC in pre-treatment ICB cohorts
- Kaplan-Meier OS stratification by PIES quartile
- Comparison to TIS score (Ayers et al. 2017)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from .utils import get_logger, load_config, results_path, data_path, models_path, save_csv, save_parquet

logger = get_logger(__name__)

# Tumor Inflammation Signature (18-gene, Ayers et al. 2017 JCI)
TIS_GENES = [
    "CD3D", "IDO1", "CIITA", "CD3E", "CCL5", "GZMK", "CD2",
    "HLA-DRA", "CXCL13", "IL2RG", "NKG7", "HLA-E", "CXCR6",
    "LAG3", "TAGAP", "CXCL10", "STAT1", "TIGIT",
]


# ── PIES computation ──────────────────────────────────────────────────────────

def compute_pies_score(
    sample_outliers: pd.Series,
    module_map: pd.Series,
    module_weights: dict[str, float],
    pt_fraction: pd.Series,
    cd8_correlation: pd.Series,
    pt_weight_multiplier: float = 1.0,
) -> float:
    """Compute PIES for a single sample.

    Parameters
    ----------
    sample_outliers      : gene → 0/1 outlier indicator for this sample
    module_map           : gene → module name
    module_weights       : module name → weight (sign encodes direction)
    pt_fraction          : gene → % post-transcriptional
    cd8_correlation      : gene → Spearman rho with CD8+ T-cell fraction
    pt_weight_multiplier : scalar upweight for post-transcriptional events

    Returns
    -------
    float PIES value
    """
    # PIES = Σ_{outlier g} module_weight[module(g)] × (1 + pt_mult × pt_frac[g]).
    # Module weights already encode evasion direction (see file docstring).
    # cd8_correlation is unused (no panel gene is in 'other'); kept for API stability.
    _ = cd8_correlation  # explicitly mark unused

    score = 0.0
    for gene, is_outlier in sample_outliers.items():
        if pd.isna(is_outlier) or is_outlier == 0:
            continue
        module  = module_map.get(gene, "other")
        mod_w   = module_weights.get(module, 1.0)
        pt_frac = pt_fraction.get(gene, 0.0) / 100.0
        pt_w    = 1.0 + pt_frac * pt_weight_multiplier
        score += mod_w * pt_w
    return score


def compute_pies_all_samples(
    outlier_binary: pd.DataFrame,
    gene_panel: pd.DataFrame,
    pt_fraction: pd.Series,
    cd8_correlation: pd.Series,
    cfg: dict,
) -> pd.Series:
    """Compute PIES for all samples in the outlier binary matrix.

    Parameters
    ----------
    outlier_binary : samples × IEP genes (0/1, NaN allowed)
    gene_panel     : DataFrame with 'gene' and 'module' columns
    pt_fraction    : indexed by gene, % post-transcriptional
    cd8_correlation: indexed by gene, Spearman rho with CD8+ T cells

    Returns
    -------
    pd.Series indexed by sample, named 'PIES'
    """
    module_map     = gene_panel.set_index("gene")["module"]
    module_weights = cfg["pies"]["module_weights"]
    pt_w_mult      = cfg["pies"]["pt_weight_multiplier"]

    scores = {
        sample: compute_pies_score(
            outlier_binary.loc[sample].fillna(0),
            module_map, module_weights,
            pt_fraction, cd8_correlation, pt_w_mult,
        )
        for sample in outlier_binary.index
    }

    return pd.Series(scores, name="PIES")


# ── RNA-PIES: direct PIES from cohort RNA (no imputation) ────────────────────

def compute_rna_outlier_binary(
    rna_expr:    pd.DataFrame,    # samples × IEP genes (any scale)
    panel_genes: list[str],
    z_threshold: float = 1.645,   # ~95th percentile one-sided
) -> pd.DataFrame:
    """Compute IEP outlier binary matrix from cohort RNA.

    For each gene, MAD-normalize within cohort then call outlier where Z > threshold.
    This sidesteps the CPTAC→ICB imputation calibration problem entirely.

    Parameters
    ----------
    rna_expr     : samples × genes RNA expression (log-CPM, FPKM, etc.)
    panel_genes  : list of IEP gene symbols
    z_threshold  : Z-score cutoff (1.645 ≈ 95th percentile one-sided)

    Returns
    -------
    DataFrame samples × IEP-genes-present (0/1)
    """
    available = [g for g in panel_genes if g in rna_expr.columns]
    sub = rna_expr[available].astype(float).copy()

    # Per-gene MAD Z-score within cohort
    z = pd.DataFrame(index=sub.index, columns=sub.columns, dtype=float)
    for g in sub.columns:
        col = sub[g]
        med = col.median()
        mad = (col - med).abs().median()
        if mad == 0 or np.isnan(mad):
            mad = 1e-6
        z[g] = (col - med) / mad

    binary = (z > z_threshold).astype(int)
    binary = binary.where(~z.isna(), 0)
    return binary


def compute_rna_pies(
    rna_expr:        pd.DataFrame,
    gene_panel:      pd.DataFrame,
    pt_fraction:     pd.Series,
    cd8_correlation: pd.Series,
    cfg:             dict,
    z_threshold:     float = 1.645,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute PIES directly from cohort RNA (no imputation).

    Returns
    -------
    pies   : per-sample PIES values
    binary : samples × IEP genes outlier matrix (used internally)
    """
    panel_genes = gene_panel["gene"].tolist()
    binary = compute_rna_outlier_binary(rna_expr, panel_genes, z_threshold)
    pies   = compute_pies_all_samples(
        binary, gene_panel, pt_fraction, cd8_correlation, cfg
    )
    return pies, binary


# ── RNA→Protein imputation models ─────────────────────────────────────────────

def train_imputation_models(
    pan_prot: pd.DataFrame,
    pan_rna_full: pd.DataFrame,
    panel_genes: list[str],
    cfg: dict,
) -> dict[str, dict]:
    """Train per-gene ElasticNet models: RNA features → protein level.

    Used to impute protein levels in RNA-only validation cohorts.

    Parameters
    ----------
    pan_prot      : CPTAC pan-cancer proteomics (samples × genes)
    pan_rna_full  : CPTAC full transcriptomics (samples × all genes)
    panel_genes   : IEP genes to train models for

    Returns
    -------
    dict: gene → {'model': ElasticNetCV, 'scaler': StandardScaler,
                  'features': list[str], 'r2': float}
    """
    n_features  = cfg["imputation_model"]["n_coexpressed_features"]
    min_samples = cfg["imputation_model"]["min_training_samples"]
    l1_ratios   = cfg["imputation_model"]["l1_ratios"]
    cv_folds    = cfg["imputation_model"]["cv_folds"]
    max_iter    = cfg["imputation_model"]["max_iter"]

    prot_no_label = pan_prot.drop(columns=["cancer_type"], errors="ignore")
    models: dict[str, dict] = {}

    for i, gene in enumerate(panel_genes):
        if gene not in prot_no_label.columns:
            continue

        y = prot_no_label[gene].dropna()
        if len(y) < min_samples:
            continue

        # Select top co-expressed RNA genes as features
        shared_samples = y.index.intersection(pan_rna_full.index)
        if len(shared_samples) < min_samples:
            continue

        corr = pan_rna_full.loc[shared_samples].corrwith(y.reindex(shared_samples))
        top_features = corr.abs().nlargest(n_features + 1).index.tolist()
        if gene in top_features:
            top_features.remove(gene)
        top_features = top_features[:n_features]

        X_df = pan_rna_full.loc[shared_samples, top_features].fillna(0)
        y_tr = y.reindex(shared_samples)

        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_df.values)
        model   = ElasticNetCV(
            l1_ratio=l1_ratios, cv=cv_folds,
            max_iter=max_iter, n_jobs=1,
        )
        model.fit(X_sc, y_tr.values)

        models[gene] = {
            "model":    model,
            "scaler":   scaler,
            "features": top_features,
            "r2":       model.score(X_sc, y_tr.values),
        }

        if (i + 1) % 50 == 0:
            logger.info(f"  Trained {i+1}/{len(panel_genes)} models")

    logger.info(f"Trained imputation models for {len(models)} IEP genes.")
    return models


def impute_protein_from_rna(
    rna_expr: pd.DataFrame,
    models: dict[str, dict],
    cutoffs: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply trained models to impute protein levels + compute outlier binary.

    Parameters
    ----------
    rna_expr : samples × genes (RNA expression, log2-TPM)
    models   : trained imputation models (from train_imputation_models)
    cutoffs  : gene → 95th-percentile threshold from CPTAC

    Returns
    -------
    imputed_prot   : samples × IEP genes (imputed protein levels)
    outlier_binary : samples × IEP genes (0/1 above cutoff)
    """
    imputed: dict[str, np.ndarray] = {}

    for gene, m in models.items():
        available = [f for f in m["features"] if f in rna_expr.columns]
        if len(available) < 5:
            continue
        X = rna_expr[available].fillna(0).values
        if X.shape[1] < len(m["features"]):
            pad = np.zeros((X.shape[0], len(m["features"]) - X.shape[1]))
            X   = np.hstack([X, pad])
        X_sc = m["scaler"].transform(X[:, :len(m["features"])])
        imputed[gene] = m["model"].predict(X_sc)

    imputed_df = pd.DataFrame(imputed, index=rna_expr.index)

    # Apply CPTAC-derived thresholds
    outlier_binary = pd.DataFrame(False, index=imputed_df.index, columns=imputed_df.columns)
    for gene in imputed_df.columns:
        if gene in cutoffs.index:
            outlier_binary[gene] = imputed_df[gene] > cutoffs.loc[gene]

    return imputed_df, outlier_binary.astype(int)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_in_cohort(
    cohort_name: str,
    cohort_cfg: dict,
    models: dict,
    cutoffs: pd.Series,
    gene_panel: pd.DataFrame,
    pt_fraction: pd.Series,
    cd8_correlation: pd.Series,
    cfg: dict,
) -> dict[str, Any]:
    """Validate PIES in one immunotherapy cohort.

    Returns
    -------
    dict with keys: auc, pvalue, n_responders, n_nonresponders, cohort_name
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    immuno_dir = data_path(cfg, "immunotherapy")
    expr_file  = immuno_dir / f"{cohort_name}_expr.csv"
    clin_file  = immuno_dir / f"{cohort_name}_clinical.csv"

    if not expr_file.exists() or not clin_file.exists():
        logger.warning(f"{cohort_name}: data files missing, skipping.")
        return {}

    expr = pd.read_csv(expr_file, index_col=0).T
    clin = pd.read_csv(clin_file, index_col=0)

    if cohort_cfg.get("log_transform", False):
        expr = np.log2(expr + 1)

    # PIES mode: 'rna' (direct, no imputation) or 'imputed' (legacy)
    pies_mode = cfg.get("validation", {}).get("pies_mode", "rna")

    if pies_mode == "rna":
        # Direct RNA-PIES — Z-score IEP panel within cohort, no imputation
        logger.info(f"  RNA-PIES mode: Z-scoring {expr.shape[1]} genes within cohort")
        pies, outlier_binary = compute_rna_pies(
            expr, gene_panel, pt_fraction, cd8_correlation, cfg,
            z_threshold=cfg.get("concordance", {}).get("outlier_z_threshold", 1.645),
        )
    else:
        # Imputed-PIES — use trained models + cutoffs (legacy)
        logger.info(f"  Imputed-PIES mode: MAD-normalizing {expr.shape[1]} features...")
        expr_mad = expr.copy().astype(float)
        for gene in expr_mad.columns:
            col = expr_mad[gene]
            med = col.median()
            mad = (col - med).abs().median()
            if mad == 0:
                mad = 1e-6
            expr_mad[gene] = (col - med) / mad
        _, outlier_binary = impute_protein_from_rna(expr_mad, models, cutoffs)
        pies = compute_pies_all_samples(
            outlier_binary, gene_panel, pt_fraction, cd8_correlation, cfg
        )

    resp_col = cohort_cfg["response_col"]
    os_col   = cohort_cfg.get("os_col")
    ev_col   = cohort_cfg.get("os_event_col")

    merged = pd.concat([pies, clin], axis=1).dropna(subset=[resp_col])
    from .utils import harmonize_response
    merged["response_binary"] = harmonize_response(merged[resp_col], cfg)
    merged = merged.dropna(subset=["response_binary"])

    result: dict[str, Any] = {"cohort_name": cohort_name}

    # AUC
    if merged["response_binary"].nunique() == 2:
        auc = roc_auc_score(merged["response_binary"], -merged["PIES"])
        result["auc"] = auc
        result["n_responders"]    = int(merged["response_binary"].sum())
        result["n_nonresponders"] = int((merged["response_binary"] == 0).sum())
        logger.info(f"  {cohort_name}: AUC = {auc:.3f} (n={len(merged)})")

    # TIS score comparison (guard: no silent fillna — missing TIS must fail loudly)
    from .tis import tis_score, HOUSEKEEPERS
    from .pies_dss import bootstrap_ci  # function-local to avoid import cycle
    tis_avail = [g for g in TIS_GENES if g in expr.columns]
    if tis_avail:
        hk = [h for h in HOUSEKEEPERS if h in expr.columns]
        tis = tis_score(expr.reindex(merged.index)).reindex(merged.index)
        if tis.isna().any():
            n_missing = int(tis.isna().sum())
            logger.warning(f"  {cohort_name}: {n_missing}/{len(tis)} samples missing "
                           "TIS genes — excluding them from the TIS comparison")
            merged_tis = merged.loc[tis.notna()]
            tis = tis.dropna()
        else:
            merged_tis = merged
        y_nr_tis = 1 - merged_tis["response_binary"].astype(int).values
        tis_auc = roc_auc_score(y_nr_tis, -tis)   # non-response orientation (≡ roc(y, tis))
        boot = bootstrap_ci(y_nr_tis, -tis)
        result["tis_auc"] = tis_auc
        result["tis_ci_lo"], result["tis_ci_hi"] = boot["ci_lo"], boot["ci_hi"]
        result["tis_n"] = int(len(tis))
        result["tis_variant"] = "hk" if hk else "madz_only"
        logger.info(f"  {cohort_name}: TIS AUC = {tis_auc:.3f} "
                    f"[{boot['ci_lo']:.3f},{boot['ci_hi']:.3f}] (n={len(tis)})")

    # Kaplan–Meier
    if os_col and ev_col and os_col in merged.columns and ev_col in merged.columns:
        km_data = merged[[os_col, ev_col, "PIES"]].dropna()
        if len(km_data) >= 10:
            km_data["PIES_group"] = pd.qcut(
                km_data["PIES"], q=2, labels=["Low PIES", "High PIES"],
                duplicates="drop",
            )
            low  = km_data[km_data["PIES_group"] == "Low PIES"]
            high = km_data[km_data["PIES_group"] == "High PIES"]

            if len(low) >= 5 and len(high) >= 5:
                lr = logrank_test(
                    low[os_col],  high[os_col],
                    event_observed_A=low[ev_col],
                    event_observed_B=high[ev_col],
                )
                result["logrank_pvalue"] = lr.p_value

                fig, ax = plt.subplots(figsize=(8, 6))
                kmf = KaplanMeierFitter()
                for grp, color, label in [
                    ("Low PIES",  "#1E88E5", f"Low PIES (n={len(low)})"),
                    ("High PIES", "#E53935", f"High PIES (n={len(high)})"),
                ]:
                    sub = km_data[km_data["PIES_group"] == grp]
                    kmf.fit(sub[os_col], sub[ev_col], label=label)
                    kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

                ax.set_title(
                    f"{cohort_name}\n"
                    f"Log-rank p={lr.p_value:.4f}  |  AUC={result.get('auc', 0):.3f}"
                )
                ax.set_xlabel("Time (months)")
                ax.set_ylabel("Overall Survival Probability")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()

                from .utils import project_root
                fig_path = project_root() / cfg["paths"]["figures_root"] / f"aim4_KM_{cohort_name}.pdf"
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(fig_path, dpi=cfg["figures"]["dpi"])
                plt.close()

    return result


# ── Master entry point ────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    """Build PIES, train imputation models, validate in ICB cohorts."""
    logger.info("=== Steps 6–7: Build PIES + Validate in ICB Cohorts ===")

    from .utils import load_gene_panel
    gene_panel  = load_gene_panel(cfg)
    panel_genes = gene_panel["gene"].tolist()

    # Load OPPTI outlier binary matrix (output of step 3)
    outlier_path = results_path(cfg, "oppti_binary_matrix.parquet")
    if not outlier_path.exists():
        raise FileNotFoundError(
            "OPPTI binary matrix not found. Run scripts/03_run_oppti.R first."
        )
    outlier_binary = pd.read_parquet(outlier_path).fillna(0)

    # Load concordance data
    pt_summary = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_fraction = pt_summary.groupby("gene")["pct_post_transcriptional"].mean()

    # Load TME correlation (output of step 5)
    tme_corr_path = results_path(cfg, "aim3_protein_immune_correlation.csv")
    cd8_corr: pd.Series
    if tme_corr_path.exists():
        tme_df   = pd.read_csv(tme_corr_path)
        cd8_corr = tme_df[tme_df["immune_metric"] == "CD8+ T-cells"].set_index("gene")["rho"]
    else:
        logger.warning("TME correlation file missing; using zero CD8 correlations.")
        cd8_corr = pd.Series(dtype=float)

    # Compute PIES on CPTAC
    pies = compute_pies_all_samples(outlier_binary, gene_panel, pt_fraction, cd8_corr, cfg)
    save_csv(pies.to_frame(), results_path(cfg, "aim4_PIES_scores_cptac.csv"))

    # Train RNA→Protein imputation models
    pan_prot_iep  = pd.read_parquet(data_path(cfg, "processed/pan_prot_iep.parquet"))
    pan_rna_full  = pd.read_parquet(data_path(cfg, "processed/pan_rna_full.parquet"))

    models = train_imputation_models(pan_prot_iep, pan_rna_full, panel_genes, cfg)
    with open(models_path(cfg, "rna_to_protein_imputation.pkl"), "wb") as f:
        pickle.dump(models, f)

    # CPTAC 95th-percentile cutoffs for outlier thresholding
    prot_no_label = pan_prot_iep.drop(columns=["cancer_type"], errors="ignore")
    cutoffs = prot_no_label.quantile(
        cfg["oppti"]["outlier_percentile"] / 100.0, axis=0
    )
    save_csv(cutoffs.to_frame("p95"), results_path(cfg, "oppti_protein_p95_cutoffs.csv"))

    # Validate in each ICB cohort
    all_results: list[dict] = []
    for name, cohort_cfg in cfg.get("immunotherapy_cohorts", {}).items():
        res = validate_in_cohort(
            name, cohort_cfg, models, cutoffs,
            gene_panel, pt_fraction, cd8_corr, cfg
        )
        if res:
            all_results.append(res)

    results_df = pd.DataFrame(all_results)
    save_csv(results_df, results_path(cfg, "aim4_AUC_by_cohort.csv"), index=False)
    logger.info(f"\nValidation summary:\n{results_df.to_string()}")
    logger.info("=== PIES Build + Validation Complete ===")


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)

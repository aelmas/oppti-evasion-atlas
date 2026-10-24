#!/usr/bin/env python
"""
Step 10: PIES sensitivity analyses (reproducible)
==================================================

Generates the sensitivity tables referenced in RESULTS.md:

  (A) Recurrence threshold sweep — # recurrent IEP proteins under different
      (min_outlier_fraction, min_cancer_types) cutoffs, for both Q95-only
      and permutation-significant call sets.

  (B) Module-weight scheme sweep — IMvigor210 / Hugo_2016 / Riaz_2017
      ICB AUC under different PIES module-weight schemes (curated, uniform,
      HLA-only, checkpoint-only, no IFN/MMR).

Outputs:
  results/sensitivity_recurrence.csv
  results/sensitivity_module_weights.csv

Usage:  python scripts/10_sensitivity_analysis.py
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from oppti_immune.pies import compute_rna_pies
from oppti_immune.utils import (
    data_path, harmonize_response, load_config, load_gene_panel,
    results_path, save_csv, get_logger,
)

logger = get_logger("10_sensitivity")


# ── (A) Recurrence-threshold sensitivity ─────────────────────────────────────

def recurrence_sweep(cfg: dict) -> pd.DataFrame:
    s_df = pd.read_csv(results_path(cfg, "aim1_pan_cancer_oppti_summary.csv"))
    rows = []
    for mf in (3, 5, 7, 10, 15, 20):
        for mc in (2, 3, 4, 5):
            n_q95 = (s_df[s_df["pct_outlier"] >= mf]
                     .groupby("protein").size().pipe(lambda s: (s >= mc).sum()))
            if "pct_outlier_perm_sig" in s_df.columns:
                n_perm = (s_df[s_df["pct_outlier_perm_sig"] >= mf]
                          .groupby("protein").size().pipe(lambda s: (s >= mc).sum()))
            else:
                n_perm = np.nan
            rows.append({
                "min_outlier_pct": mf,
                "min_cancer_types": mc,
                "n_recurrent_Q95": int(n_q95),
                "n_recurrent_perm_sig": int(n_perm) if not np.isnan(n_perm) else "NA",
            })
    return pd.DataFrame(rows)


# ── (B) Module-weight scheme sensitivity ──────────────────────────────────────

def _load_cohort(name: str, cfg: dict):
    expr = pd.read_csv(f"data/immunotherapy/{name}_expr.csv", index_col=0).T
    clin = pd.read_csv(f"data/immunotherapy/{name}_clinical.csv", index_col=0)
    if name == "IMvigor210":
        expr = np.log2(expr + 1)
        rcol = "binaryResponse"
    else:
        rcol = "response"
    clin["resp_binary"] = harmonize_response(clin[rcol], cfg)
    return expr, clin


def module_weight_sweep(cfg: dict) -> pd.DataFrame:
    gene_panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    tme = pd.read_csv(results_path(cfg, "aim3_protein_immune_correlation.csv"))
    cd8 = tme[tme["immune_metric"] == "CD8+ T-cells"].set_index("gene")["rho"]

    cohorts = {n: _load_cohort(n, cfg) for n in ("IMvigor210", "Hugo_2016", "Riaz_2017")}

    base_weights = cfg["pies"]["module_weights"]
    schemes = {
        "default_curated":  dict(base_weights),
        "uniform_plus1":    {m: 1.0 for m in base_weights},
        "HLA_only":         {**{m: 0.0 for m in base_weights}, "antigen_presentation": -3.0},
        "checkpoint_only":  {**{m: 0.0 for m in base_weights}, "checkpoint": 3.0},
        "no_ifn_mmr":       {**base_weights, "ifn_pathway": 0.0, "mismatch_repair": 0.0},
    }

    rows = []
    for scheme_name, weights in schemes.items():
        cfg_t = {**cfg, "pies": {**cfg["pies"], "module_weights": weights}}
        row = {"scheme": scheme_name}
        for c, (expr, clin) in cohorts.items():
            pies, _ = compute_rna_pies(expr, gene_panel, pt_frac, cd8, cfg_t)
            merged = pd.concat([pies, clin], axis=1).dropna(subset=["resp_binary"])
            row[c] = (roc_auc_score(merged["resp_binary"], -merged["PIES"])
                      if merged["resp_binary"].nunique() == 2 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ── (C) Random-weight null + leave-one-module-out ablation ────────────────────

def _auc_for_weights(weights, cohorts, gene_panel, pt_frac, cd8, cfg) -> dict:
    cfg_t = {**cfg, "pies": {**cfg["pies"], "module_weights": weights}}
    out = {}
    for c, (expr, clin) in cohorts.items():
        pies, _ = compute_rna_pies(expr, gene_panel, pt_frac, cd8, cfg_t)
        merged = pd.concat([pies, clin], axis=1).dropna(subset=["resp_binary"])
        out[c] = (roc_auc_score(merged["resp_binary"], -merged["PIES"])
                  if merged["resp_binary"].nunique() == 2 else np.nan)
    return out


def random_weight_null(cfg: dict, n_draws: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Draw n_draws random module-weight vectors and report the empirical
    percentile of the curated weights' AUC within the null, per cohort.

    Random weights: magnitude ~ Uniform(0.5, 3.0), sign ~ Bernoulli(0.5),
    matching the scale/structure of the curated scheme but with arbitrary
    sign and size. Tests whether the curated weighting beats chance weighting.
    """
    rng = np.random.default_rng(seed)
    gene_panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    tme = pd.read_csv(results_path(cfg, "aim3_protein_immune_correlation.csv"))
    cd8 = tme[tme["immune_metric"] == "CD8+ T-cells"].set_index("gene")["rho"]
    cohorts = {n: _load_cohort(n, cfg) for n in ("IMvigor210", "Hugo_2016", "Riaz_2017")}
    modules = list(cfg["pies"]["module_weights"].keys())

    curated = _auc_for_weights(cfg["pies"]["module_weights"], cohorts, gene_panel, pt_frac, cd8, cfg)

    null = {c: [] for c in cohorts}
    for _ in range(n_draws):
        w = {m: float(rng.uniform(0.5, 3.0) * (1 if rng.random() < 0.5 else -1)) for m in modules}
        a = _auc_for_weights(w, cohorts, gene_panel, pt_frac, cd8, cfg)
        for c in cohorts:
            null[c].append(a[c])

    rows = []
    for c in cohorts:
        arr = np.array(null[c], dtype=float)
        arr = arr[np.isfinite(arr)]
        cur = curated[c]
        # percentile of curated within null (two-sided distance from 0.5)
        pct = float((np.abs(arr - 0.5) <= np.abs(cur - 0.5)).mean()) * 100  # how ordinary curated's |AUC-0.5| is
        beat = float((np.abs(arr - 0.5) >= np.abs(cur - 0.5)).mean())       # frac of random ≥ curated discrimination
        rows.append({
            "cohort": c,
            "curated_auc": round(cur, 3),
            "null_auc_mean": round(arr.mean(), 3),
            "null_auc_q025": round(np.quantile(arr, 0.025), 3),
            "null_auc_q975": round(np.quantile(arr, 0.975), 3),
            "frac_random_geq_curated_discrimination": round(beat, 3),
            "n_draws": len(arr),
        })
    return pd.DataFrame(rows)


def leave_one_module_out(cfg: dict) -> pd.DataFrame:
    """Zero each module weight in turn; report per-cohort AUC delta vs curated."""
    gene_panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    tme = pd.read_csv(results_path(cfg, "aim3_protein_immune_correlation.csv"))
    cd8 = tme[tme["immune_metric"] == "CD8+ T-cells"].set_index("gene")["rho"]
    cohorts = {n: _load_cohort(n, cfg) for n in ("IMvigor210", "Hugo_2016", "Riaz_2017")}
    base = cfg["pies"]["module_weights"]

    curated = _auc_for_weights(base, cohorts, gene_panel, pt_frac, cd8, cfg)
    rows = [{"removed_module": "(none/curated)", **{c: round(v, 3) for c, v in curated.items()}}]
    for m in base:
        w = {**base, m: 0.0}
        a = _auc_for_weights(w, cohorts, gene_panel, pt_frac, cd8, cfg)
        rows.append({"removed_module": m,
                     **{c: round(a[c], 3) for c in cohorts},
                     **{f"{c}_delta": round(a[c] - curated[c], 3) for c in cohorts}})
    return pd.DataFrame(rows)


# ── (D) MMR-sign sensitivity ─────────────────────────────────────────────────

def mmr_sign_check(cfg: dict) -> pd.DataFrame:
    """MMR loss → MSI-high → better ICB response, so high-MMR overexpression
    outliers arguably mark MORE evasion (positive weight), opposite to the
    curated -1.5. Flip the MMR sign and report the AUC impact.
    """
    gene_panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    tme = pd.read_csv(results_path(cfg, "aim3_protein_immune_correlation.csv"))
    cd8 = tme[tme["immune_metric"] == "CD8+ T-cells"].set_index("gene")["rho"]
    cohorts = {n: _load_cohort(n, cfg) for n in ("IMvigor210", "Hugo_2016", "Riaz_2017")}
    base = cfg["pies"]["module_weights"]

    rows = []
    for label, mmr_w in [("curated (MMR -1.5)", -1.5), ("MMR flipped (+1.5)", 1.5), ("MMR zeroed (0)", 0.0)]:
        w = {**base, "mismatch_repair": mmr_w}
        a = _auc_for_weights(w, cohorts, gene_panel, pt_frac, cd8, cfg)
        rows.append({"mmr_scheme": label, **{c: round(a[c], 3) for c in cohorts}})
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-draws", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)

    rec_df = recurrence_sweep(cfg)
    save_csv(rec_df, results_path(cfg, "sensitivity_recurrence.csv"), index=False)
    logger.info("Recurrence-threshold sensitivity:\n" + rec_df.to_string(index=False))

    mw_df = module_weight_sweep(cfg)
    save_csv(mw_df, results_path(cfg, "sensitivity_module_weights.csv"), index=False)
    logger.info("\nModule-weight sensitivity (PIES AUC by scheme):\n" + mw_df.to_string(index=False))

    rn_df = random_weight_null(cfg, n_draws=args.n_draws)
    save_csv(rn_df, results_path(cfg, "sensitivity_random_weight_null.csv"), index=False)
    logger.info("\nRandom-weight null (does curation beat chance weighting?):\n" + rn_df.to_string(index=False))

    lomo_df = leave_one_module_out(cfg)
    save_csv(lomo_df, results_path(cfg, "sensitivity_leave_one_module_out.csv"), index=False)
    logger.info("\nLeave-one-module-out ablation:\n" + lomo_df.to_string(index=False))

    mmr_df = mmr_sign_check(cfg)
    save_csv(mmr_df, results_path(cfg, "sensitivity_mmr_sign.csv"), index=False)
    logger.info("\nMMR-sign sensitivity:\n" + mmr_df.to_string(index=False))


if __name__ == "__main__":
    main()

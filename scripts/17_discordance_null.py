#!/usr/bin/env python
"""
Step 17 — Discordance null: is 54% RNA-invisible more than chance?
==================================================================
Reviewer-driven null-model check for the atlas headline (2,964/5,473 = 54.2%
of Q95 protein outlier calls lack a matching RNA elevation).

For every IEP gene:
  rho_g : Spearman correlation of matched protein/RNA across CPTAC tumours
          (rank-based, so invariant to the shipped MAD-Z/kNN transforms).
  obs_g : observed invisible fraction P(|Z_rna| < 1.645 | protein outlier
          call), taken from the shipped aim2 concordance classification.
  exp_g : the same quantity expected under a bivariate-normal pair with
          correlation rho_g — the "no post-transcriptional biology" null,
          i.e. pure measurement noise given each gene's own correlation.

Overall inference:
  observed total vs expected total vs matched non-immune control sets
  (random equal-size gene sets drawn from the same full CPTAC matrices and
  run through the identical MAD-Z-outlier / |Z|-threshold rules).

Outputs
  results/aim7_discordance_null_genes.csv    : per-gene table
  results/aim7_discordance_null_summary.csv  : overall / control summary

Usage:  python scripts/17_discordance_null.py [--mc 200000] [--control-sets 200]
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("17_discordance_null")
Z_OUT = 1.645   # one-sided protein-outlier threshold (Q95 / config)
Z_INV = 1.645   # RNA "elevated" iff Z_rna >= 1.645 (two-sided invisibility)


def madz(mat: pd.DataFrame) -> pd.DataFrame:
    """Per-column (gene) MAD-Z on a samples x genes matrix."""
    med = mat.median(axis=0)
    mad = (mat - med).abs().median(axis=0)
    return (mat - med) / mad.mask(mad == 0, 1e-6)


def load_paired(iep_only: bool) -> pd.DataFrame:
    """MAD-Z'd per-cohort protein values, aligned with same-sample MAD-Z RNA."""
    prot = pd.read_parquet("data/processed/pan_prot_iep.parquet" if iep_only
                           else "data/processed/pan_prot_full.parquet")
    rna = pd.read_parquet("data/processed/pan_rna_iep.parquet" if iep_only
                          else "data/processed/pan_rna_full.parquet")
    prot, rna = prot.set_index("sample_id"), rna.set_index("sample_id")
    if "cancer_type" in prot.columns:
        prot = prot.drop(columns=["cancer_type"])
    common = prot.index.intersection(rna.index)
    prot_z = madz(prot.loc[common])
    rna_z = madz(rna.loc[common])
    return prot_z, rna_z


def expected_invisible(rho: float, n_mc: int, rng: np.random.Generator) -> float:
    """P(Zp >= 1.645, |Zr| < 1.645) under BVN(0, rho), by Monte Carlo."""
    n_blocks, block = 0, 40_000
    acc, zp_acc = 0, 0
    remaining = n_mc
    while remaining > 0:
        n = min(block, remaining)
        zp = rng.standard_normal(n)
        zr = rho * zp + np.sqrt(max(0.0, 1.0 - rho ** 2)) * rng.standard_normal(n)
        acc += int(((zp >= Z_OUT) & (np.abs(zr) < Z_INV)).sum())
        zp_acc += int((zp >= Z_OUT).sum())
        remaining -= n
    # condition on protein-outlier events for a directly comparable rate
    return acc / zp_acc if zp_acc else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--mc", type=int, default=200_000, help="MC pairs per gene")
    ap.add_argument("--control-sets", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = load_gene_panel(cfg)["gene"].tolist()
    rng = np.random.default_rng(args.seed)

    # ── observed calls from the shipped classification ───────────────────────
    calls = pd.read_parquet(results_path(cfg, "aim2_concordance_classification.parquet"))
    obs = (calls.assign(invisible=calls["concordance"] == "post_transcriptional")
                .groupby("gene")["invisible"].agg(["mean", "size"])
                .rename(columns={"mean": "obs_frac", "size": "n_calls"}))
    logger.info(f"observed calls total={int(obs.n_calls.sum())}, "
                f"invisible weighted={np.average(obs.obs_frac, weights=obs.n_calls):.4f}")

    # ── per-gene rho from the paired matrices ────────────────────────────────
    prot_z, rna_z = load_paired(iep_only=True)
    assert not prot_z.index.intersection(rna_z.index).empty
    genes = [g for g in panel if g in prot_z.columns and g in rna_z.columns]
    logger.info(f"paired matrix samples={len(prot_z.index)} | panel genes matched={len(genes)}/137")

    rows = []
    for g in genes:
        pr, rr = prot_z[g].values, rna_z[g].values
        mask = np.isfinite(pr) & np.isfinite(rr)
        rho = float(spearmanr(pr[mask], rr[mask]).statistic) if mask.sum() > 3 else np.nan
        exp = expected_invisible(rho, args.mc, rng)
        o = obs.loc[g] if g in obs.index else None
        rows.append({
            "gene": g, "spearman_rho": rho, "n_pairs": int(mask.sum()),
            "exp_frac": exp,
            "obs_frac_shipped": float(o["obs_frac"]) if o is not None else np.nan,
            "n_calls_shipped": int(o["n_calls"]) if o is not None else 0,
        })
    per_gene = pd.DataFrame(rows)

    # ── self-contained observed re-check on plain MAD-Z matrices ─────────────
    calls_m = (prot_z[genes] >= Z_OUT).reindex(index=rna_z.index)
    invis = (np.abs(rna_z[genes]) < Z_INV)
    tot, tot_inv = int(calls_m.values.sum()), int((calls_m & invis.reindex(columns=genes)).values.sum())
    logger.info(f"plain-MAD-Z cross-check: calls={tot}, invisible={tot_inv} "
                f"({tot_inv / tot:.4f})")

    # ── expected total (weighted by calls) ───────────────────────────────────
    per_gene["excess"] = per_gene["obs_frac_shipped"] - per_gene["exp_frac"]
    exp_total = np.average(per_gene["exp_frac"], weights=per_gene["n_calls_shipped"])
    obs_total = np.average(per_gene["obs_frac_shipped"], weights=per_gene["n_calls_shipped"])
    logger.info(f"EXPECTED total invisible fraction (noise null) = {exp_total:.4f}")
    logger.info(f"OBSERVED total invisible fraction (shipped)    = {obs_total:.4f}")
    logger.info(f"EXCESS = {obs_total - exp_total:+.4f}")

    # ── matched non-immune control sets from the full matrices ───────────────
    prot_f, rna_f = load_paired(iep_only=False)
    shared = [c for c in prot_f.columns
              if c in rna_f.columns and c not in set(panel) and c != "sample_id"]
    logger.info(f"control gene pool (non-IEP, protein+RNA shared): {len(shared)}")
    ctl_obs, ctl_exp = [], []
    for _ in range(args.control_sets):
        gs = list(rng.choice(shared, size=len(panel), replace=False))
        pct = prot_f[gs].values >= Z_OUT
        inv = np.abs(rna_f[gs].values) < Z_INV
        c, i = int(pct.sum()), int((pct & inv).sum())
        if c:
            ctl_obs.append(i / c)
        for g in gs:  # expected per matched gene, same weighting
            pass
        # expected fraction for the control set (MC on its per-gene rhos)
        rhos = []
        pr, rr = prot_f[gs].values, rna_f[gs].values
        for k, g in enumerate(gs):
            m = np.isfinite(pr[:, k]) & np.isfinite(rr[:, k])
            rhos.append(spearmanr(pr[m, k], rr[m, k]).statistic if m.sum() > 3 else 0.0)
        e = [expected_invisible(float(r), 20_000, rng) for r in rhos]
        ctl_exp.append(float(np.mean(e)))
    summary = pd.DataFrame([
        {"quantity": "observed_shipped", "value": obs_total},
        {"quantity": "expected_noise_null", "value": exp_total},
        {"quantity": "excess_over_null", "value": obs_total - exp_total},
        {"quantity": "control_observed_mean", "value": float(np.mean(ctl_obs))},
        {"quantity": "control_observed_sd", "value": float(np.std(ctl_obs))},
        {"quantity": "control_expected_mean", "value": float(np.mean(ctl_exp))},
        {"quantity": "control_z_obs_vs_control_sets",
         "value": (obs_total - np.mean(ctl_obs)) / max(np.std(ctl_obs), 1e-9)},
    ])
    save_csv(per_gene, results_path(cfg, "aim7_discordance_null_genes.csv"), index=False)
    save_csv(summary, results_path(cfg, "aim7_discordance_null_summary.csv"), index=False)
    logger.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()

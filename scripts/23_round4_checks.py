#!/usr/bin/env python
"""
Step 23 — Round-4 checks
========================
Review round 4 items (ordered per the reviewer's re-prioritization):

1. Matched discordance controls (CPTAC; protects the headline 54.2%):
   control gene sets matched to the IEP panel on protein-abundance decile,
   protein-variance decile, and protein-RNA Spearman-rho decile, excluding
   immune genes (IEP panel + TIS signature + housekeepers), run through the
   identical MAD-Z Q95 / |Z|<1.645 rules; where does the IEP fraction fall?
2. Burden direction test: burden alone vs burden + logit(DSS) (unpenalized,
   nested 5-fold OOF, paired) — does the score add anything to burden?
3. Brier reference: no-information (prevalence) Brier and Brier skill
   scores for the raw and nested-Platt-recalibrated OOF probabilities.
4. Immune-annotated random panels (IMvigor210): 137-gene panels drawn from
   an immune-annotated universe (prefix rule, documented below), matched on
   (mean, variance) expression deciles, excluding IEP members; fixed-C ridge
   scored, IEP percentile within.
5. Module-label shuffles within the IEP panel (200): does module structure
   (vs random grouping of the same genes) matter? Fixed-C ridge scored.
6. Missing Table-6 paired deltas: mean-of-10 module-preserving pt_frac
   shuffles vs shipped; gene-level ridge lambda=-3 and shuffled-prior
   lambda=+3 vs lambda=0.
7. Power simulation verification: one-sided Mann-Whitney z-test (alpha=0.05)
   of AUC > 0.5, 30% responders, N in {100,150,200,300}, true AUC in
   {0.60, 0.62, 0.675} — verifies the reviewer's table for the freeze plan.

Outputs:
  results/aim13_round4_checks.csv
  results/aim13_power_simulation.csv
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from oppti_immune.pies_dss import (
    fit_dss, load_cohort, madz_outliers, module_features, nested_cv_dss,
    select_C_via_inner_cv,
)
from oppti_immune.tis import HOUSEKEEPERS, TIS_GENES
from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("23_round4_checks")
logging.getLogger("pies_dss").setLevel(logging.WARNING)

N_BOOT = 2000
SEED = 42
Z_OUT = 1.645
Z_INV = 1.645

# immune-annotated universe prefix rule (pre-declared, documented in Methods)
IMMUNE_PREFIXES = ("HLA", "IL", "CCL", "CCR", "CXCL", "CXCR", "IFIT", "IFI", "IFN",
                   "TAP", "GZM", "TNFRSF", "ICOS", "CTLA", "PDCD", "LAG3", "HAVCR2",
                   "TIGIT", "VSIR", "BTLA", "TBX", "PRF", "CD27", "CD28", "CD40")


def madz(mat: pd.DataFrame) -> pd.DataFrame:
    med = mat.median(axis=0)
    mad = (mat - med).abs().median(axis=0)
    return (mat - med) / mad.mask(mad == 0, 1e-6)


def load_raw_paired() -> tuple[pd.DataFrame, pd.DataFrame]:
    prot = pd.read_parquet("data/processed/pan_prot_full.parquet").set_index("sample_id")
    rna = pd.read_parquet("data/processed/pan_rna_full.parquet").set_index("sample_id")
    if "cancer_type" in prot.columns:
        prot = prot.drop(columns=["cancer_type"])
    common = prot.index.intersection(rna.index)
    return prot.loc[common], rna.loc[common]


def oof_ridge(X: pd.DataFrame, y_nr: np.ndarray, C: float = 3.0,
              scale: np.ndarray | None = None) -> np.ndarray:
    Xv = X.values.astype(float)
    if scale is not None:
        Xv = Xv * scale
    oof = np.full(len(Xv), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xv, y_nr):
        m = LogisticRegression(penalty="l2", C=C, solver="liblinear", max_iter=5000)
        m.fit(Xv[tr], y_nr[tr])
        oof[te] = m.predict_proba(Xv[te])[:, 1]
    return oof


def paired_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        deltas.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.asarray(deltas)
    return {"delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
            "p_boot": float(np.mean(d <= 0) * 2)}


def unpenalized_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = LogisticRegression(C=1e6, max_iter=5000).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


# ─────────────────────────── 1. CPTAC matched controls ────────────────────────
def matched_discordance_controls(cfg: dict, panel_genes: list[str],
                                 n_sets: int = 200, seed: int = 7) -> dict:
    prot_raw, rna_raw = load_raw_paired()
    immune = set(panel_genes) | set(TIS_GENES) | set(HOUSEKEEPERS)
    shared = [c for c in prot_raw.columns
              if c in rna_raw.columns and c not in immune and c != "sample_id"]
    prot_z, rna_z = madz(prot_raw), madz(rna_raw)

    # per-gene stats over pool ∪ IEP panel (so panel genes get decile cells)
    panel_in = [g for g in panel_genes if g in prot_raw.columns and g in rna_raw.columns]
    allg = list(shared) + [g for g in panel_in if g not in shared]
    stats = {}
    for g in allg:
        p, r = prot_raw[g].values, rna_raw[g].values
        m = np.isfinite(p) & np.isfinite(r)
        stats[g] = (float(np.nanmean(p)), float(np.nanvar(p)),
                    float(spearmanr(p[m], r[m]).statistic) if m.sum() > 3 else 0.0)
    st = pd.DataFrame(stats, index=["mean", "var", "rho"]).T
    for c in st.columns:
        st[c + "_dec"] = pd.qcut(st[c].rank(method="first"), 10, labels=False)
    cells = {k: set(v) for k, v in st.groupby(["mean_dec", "var_dec", "rho_dec"]).groups.items()}
    by_rho = {k: set(v) for k, v in st.groupby("rho_dec").groups.items()}

    rng = np.random.default_rng(seed)
    ctl_obs = []
    for it in range(n_sets):
        chosen = []
        for g in panel_in:
            cell = (st.at[g, "mean_dec"], st.at[g, "var_dec"], st.at[g, "rho_dec"])
            cand = [c for c in cells.get(cell, []) if c not in immune and c not in chosen]
            if not cand:
                cand = [c for c in by_rho.get(cell[2], []) if c not in immune and c not in chosen]
            if not cand:
                cand = [c for c in shared if c not in immune and c not in chosen]
            chosen.append(cand[rng.integers(0, len(cand))])
        gs = chosen
        pct = (prot_z[gs].values >= Z_OUT)
        inv = (np.abs(rna_z[gs].values) < Z_INV)
        c, i_ = int(pct.sum()), int((pct & inv).sum())
        if c:
            ctl_obs.append(i_ / c)
    return {"n_sets": len(ctl_obs),
            "matched_control_mean": float(np.mean(ctl_obs)),
            "matched_control_sd": float(np.std(ctl_obs)),
            "matched_control_max": float(np.max(ctl_obs)),
            "matched_control_min": float(np.min(ctl_obs))}


# ─────────────────────────── 7. power simulation ─────────────────────────────
def power_sim(n_grid=(100, 150, 200, 300), auc_grid=(0.60, 0.62, 0.675),
              p_resp=0.30, n_sim=10_000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for N in n_grid:
        n1, n0 = int(round(N * p_resp)), N - int(round(N * p_resp))
        se0 = np.sqrt((N + 1) / (12 * n1 * n0))  # no-margin SE of AUC under H0
        for auc in auc_grid:
            shift = np.sqrt(2) * norm.ppf(auc)
            rej = 0
            for _ in range(n_sim):
                s1 = rng.standard_normal(n1) + shift
                s0 = rng.standard_normal(n0)
                ranks = np.argsort(np.argsort(np.concatenate([s1, s0])))
                auc_hat = ranks[:n1].sum() / (n1 * n0) - (n1 * (n1 - 1) / 2) / (n1 * n0)
                z = (auc_hat - 0.5) / se0
                rej += z > norm.ppf(0.95)
            rows.append({"N": N, "true_auc": auc, "power": rej / n_sim})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config("config/config.yaml")
    panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    rows: list[dict] = []

    # ── 1. CPTAC matched discordance controls ──────────────────────────────
    mc = matched_discordance_controls(cfg, panel["gene"].tolist())
    # observed IEP fraction on the identical plain-MAD-Z rules
    prot_raw, rna_raw = load_raw_paired()
    prot_z, rna_z = madz(prot_raw), madz(rna_raw)
    pg = [g for g in panel["gene"] if g in prot_z.columns and g in rna_z.columns]
    pct = (prot_z[pg].values >= Z_OUT)
    inv = (np.abs(rna_z[pg].values) < Z_INV)
    c, i_ = int(pct.sum()), int((pct & inv).sum())
    iep_frac = i_ / c
    z_m = (iep_frac - mc["matched_control_mean"]) / max(mc["matched_control_sd"], 1e-9)
    rows += [{"section": "matched_discordance", "item": "IEP invisible fraction (plain MAD-Z rules)",
              "value": iep_frac, "n": c},
             {"section": "matched_discordance", "item": "matched control mean (200 sets)",
              "value": mc["matched_control_mean"]},
             {"section": "matched_discordance", "item": "matched control SD",
              "value": mc["matched_control_sd"]},
             {"section": "matched_discordance", "item": "z (IEP vs matched controls)",
              "value": z_m},
             {"section": "matched_discordance", "item": "matched control max",
              "value": mc["matched_control_max"]}]

    # ── IMvigor210 setup for 2-6 ───────────────────────────────────────────
    rna, y = load_cohort("IMvigor210", cfg)
    y_nr = (1 - y.astype(int)).values
    med = rna.median()
    mad = (rna - med).abs().median().replace(0, 1e-9)
    Z = (rna - med) / mad
    im_pg = [g for g in panel["gene"] if g in Z.columns]
    outliers = madz_outliers(rna, panel["gene"].tolist(), z_threshold=Z_OUT)
    feats = module_features(outliers, panel, pt_frac)
    iof = pd.read_csv(results_path(cfg, "aim4_pies_dss_iof_scores.csv")).set_index("sample")
    dss = iof["PIES-DSS"].reindex(feats.index).values
    clin = pd.read_csv("data/immunotherapy/IMvigor210_clinical.csv", index_col=0)
    tmb = pd.to_numeric(clin["FMOne mutation burden per MB"], errors="coerce").reindex(feats.index)
    burden = (Z.abs() >= Z_OUT).astype(float).sum(axis=1).reindex(feats.index)
    shipped = np.asarray(nested_cv_dss(feats, y, seed=SEED)["out_of_fold"]["PIES-DSS"], float)

    # ── 2. burden direction test (burden alone vs burden + score) ──────────
    cc = tmb.notna().values if False else np.ones(len(y_nr), bool)
    bz = ((burden - burden.mean()) / burden.std()).values.reshape(-1, 1)
    lg = np.log(np.clip(dss, 1e-6, 1 - 1e-6) / (1 - np.clip(dss, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    o_b = unpenalized_oof(bz, y_nr)
    o_bs = unpenalized_oof(np.hstack([bz, lg]), y_nr)
    pd_dir = paired_delta(y_nr, o_bs, o_b)
    rows += [{"section": "burden_direction", "item": "burden alone (unpenalized, nested)",
              "value": float(roc_auc_score(y_nr, o_b))},
             {"section": "burden_direction", "item": "burden + logit(DSS)",
              "value": float(roc_auc_score(y_nr, o_bs)),
              "delta_auc": pd_dir["delta_auc"], "ci_lo": pd_dir["ci_lo"],
              "ci_hi": pd_dir["ci_hi"], "p_boot": pd_dir["p_boot"]}]

    # ── 3. Brier reference + skill scores ─────────────────────────────────
    p_prev = y_nr.mean()
    b_ref = p_prev * (1 - p_prev)
    p_raw = np.clip(dss, 1e-6, 1 - 1e-6)
    lg_full = np.log(p_raw / (1 - p_raw)).reshape(-1, 1)
    o_pl = unpenalized_oof(lg_full, y_nr)  # nested Platt (already computed in aim12; recompute)
    b_raw = float(np.mean((dss - y_nr) ** 2))
    b_pl = float(np.mean((o_pl - y_nr) ** 2))
    rows += [{"section": "brier_skill", "item": "no-information (prevalence) Brier",
              "value": b_ref},
             {"section": "brier_skill", "item": "raw OOF Brier", "value": b_raw},
             {"section": "brier_skill", "item": "raw Brier skill score", "value": 1 - b_raw / b_ref},
             {"section": "brier_skill", "item": "nested-Platt OOF Brier", "value": b_pl},
             {"section": "brier_skill", "item": "nested-Platt Brier skill score",
              "value": 1 - b_pl / b_ref}]

    # ── 4. immune-annotated random panels ─────────────────────────────────
    dec_mean = pd.qcut(rna.mean(axis=0), 10, labels=False, duplicates="drop")
    dec_var = pd.qcut(rna.var(axis=0), 10, labels=False, duplicates="drop")
    cell = pd.DataFrame({"dm": dec_mean, "dv": dec_var})
    universe = [g for g in rna.columns
                if g.startswith(IMMUNE_PREFIXES) and g not in set(pg)]
    logger.info("immune-annotated universe (non-IEP): %d genes", len(universe))
    rng = np.random.default_rng(9)
    uni_aucs = []
    for it in range(300):
        genes, picks = [], set()
        for _ in range(len(pg)):
            # match each draw to the IEP gene it replaces (same decile cells)
            g = pg[len(genes)]
            dm, dv = cell.at[g, "dm"], cell.at[g, "dv"]
            cand = [x for x in universe
                    if cell.at[x, "dm"] == dm and cell.at[x, "dv"] == dv and x not in picks]
            if not cand:
                cand = [x for x in universe if x not in picks]
            pick = cand[rng.integers(0, len(cand))]
            picks.add(pick)
            genes.append(pick)
        fake = pd.DataFrame({"gene": genes, "module": panel.set_index("gene")
                             .loc[[g for g in pg], "module"].values})
        fx = module_features((Z[genes].abs() >= Z_OUT).astype(float), fake, pt_frac * 0)
        uni_aucs.append(float(roc_auc_score(y_nr, oof_ridge(fx, y_nr))))
    uni_aucs = np.asarray(uni_aucs)
    iep_uni_auc = float(roc_auc_score(y_nr, oof_ridge(
        module_features(outliers[im_pg], panel, pt_frac * 0), y_nr)))
    pct_uni = float(np.mean(uni_aucs >= iep_uni_auc) * 100)
    rows += [{"section": "immune_panels", "item": "IEP panel, fixed-C ridge, uniform",
              "value": iep_uni_auc},
             {"section": "immune_panels", "item": "immune-annotated matched panels: mean",
              "value": float(uni_aucs.mean())},
             {"section": "immune_panels", "item": "immune-annotated matched panels: SD",
              "value": float(uni_aucs.std())},
             {"section": "immune_panels", "item": "immune-annotated matched panels: max",
              "value": float(uni_aucs.max())},
             {"section": "immune_panels", "item": "percentile of IEP (immune panels)",
              "value": pct_uni}]

    # ── 5. module-label shuffles within the IEP panel ──────────────────────
    mod_of = panel.set_index("gene")["module"]
    sizes = mod_of.loc[pg].value_counts().to_dict()
    shuf_aucs = []
    for it in range(200):
        labs = list(mod_of.loc[pg].values)
        rng.shuffle(labs)
        fake = pd.DataFrame({"gene": pg, "module": labs})
        fx = module_features((Z[pg].abs() >= Z_OUT).astype(float), fake, pt_frac * 0)
        shuf_aucs.append(float(roc_auc_score(y_nr, oof_ridge(fx, y_nr))))
    shuf_aucs = np.asarray(shuf_aucs)
    pct_shuf = float(np.mean(shuf_aucs >= iep_uni_auc) * 100)
    rows += [{"section": "module_shuffles", "item": "module-label shuffles: mean",
              "value": float(shuf_aucs.mean())},
             {"section": "module_shuffles", "item": "module-label shuffles: SD",
              "value": float(shuf_aucs.std())},
             {"section": "module_shuffles", "item": "module-label shuffles: max",
              "value": float(shuf_aucs.max())},
             {"section": "module_shuffles", "item": "percentile of IEP (module shuffles)",
              "value": pct_shuf}]

    # ── 6. missing Table-6 paired deltas ──────────────────────────────────
    shuffle_oofs = []
    for it in range(10):
        sh = pt_frac.copy()
        rng2 = np.random.default_rng(100 + it)
        for m, gset in panel.groupby("module")["gene"]:
            idx = [g for g in gset if g in sh.index]
            sh.loc[idx] = rng2.permutation(sh.loc[idx].values)
        f_sh = module_features(outliers[im_pg], panel, sh)
        shuffle_oofs.append(np.asarray(
            nested_cv_dss(f_sh, y, seed=SEED)["out_of_fold"]["PIES-DSS"], float))
    sh_mean = np.mean(shuffle_oofs, axis=0)
    pd_sh = paired_delta(y_nr, sh_mean, shipped)
    rows.append({"section": "table6_deltas", "item": "mean of 10 module-preserving shuffles vs shipped",
                 "value": float(roc_auc_score(y_nr, sh_mean)),
                 "delta_auc": pd_sh["delta_auc"], "ci_lo": pd_sh["ci_lo"],
                 "ci_hi": pd_sh["ci_hi"], "p_boot": pd_sh["p_boot"]})

    w_g = 1.0 + pt_frac.reindex(im_pg).fillna(0) / 100.0
    Xg = outliers[im_pg]
    o_l0 = oof_ridge(Xg, y_nr, scale=(w_g ** 0).values)
    o_ln3 = oof_ridge(Xg, y_nr, scale=(w_g ** -3).values)
    wj = pd.Series(w_g.values, index=rng.permutation(w_g.index)).reindex(im_pg)
    o_lsh = oof_ridge(Xg, y_nr, scale=(wj ** 3).values)
    for name, o in [("lambda=-3 vs lambda=0", o_ln3), ("shuffled-prior lambda=+3 vs lambda=0", o_lsh)]:
        pdl = paired_delta(y_nr, o, o_l0)
        rows.append({"section": "table6_deltas", "item": f"gene-level ridge {name}",
                     "value": float(roc_auc_score(y_nr, o)),
                     "delta_auc": pdl["delta_auc"], "ci_lo": pdl["ci_lo"],
                     "ci_hi": pdl["ci_hi"], "p_boot": pdl["p_boot"]})

    out = pd.DataFrame(rows)
    save_csv(out, results_path(cfg, "aim13_round4_checks.csv"), index=False)
    logger.info("\n" + out.to_string(index=False))

    pw = power_sim()
    save_csv(pw, results_path(cfg, "aim13_power_simulation.csv"), index=False)
    logger.info("\n" + pw.to_string(index=False))


if __name__ == "__main__":
    main()

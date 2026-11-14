#!/usr/bin/env python
"""
Step 21 — What does the score measure? Negative controls and variant set
=========================================================================
Review round 2, priority items A1-A3, C7 (compact), C8, D13-D14:

1. Negative controls (A1): genome-wide outlier burden, library size
   (sizeFactor), TMB correlation with the DSS OOF score; 500 random gene
   panels matched on module size and (mean, variance) expression deciles,
   scored with a fixed-C ridge — where does PIES-DSS fall?
2. Signed events (A2): 16 features (8 modules x up/down events) under the
   same nested CV; unsigned continuous |Z| module sums as the missing
   representation control.
3. Count normalization (A3): sizeFactor-normalized counts -> log2 -> MAD-Z
   -> events -> modules -> nested CV (sensitivity rerun of the main model).
4. TMB-conditional learning (C8): module weights fitted with TMB as an
   (effectively) unpenalized covariate via column scaling; paired delta-AUC
   vs TMB alone.
5. Prior-leverage test (C7, compact): gene-level event ridge with per-gene
   penalty multipliers s_g = (1 + pt_frac_g/100)^lambda at lambda in
   {0, +3, -3} plus a shuffled prior at lambda = +3 (four pre-declared
   variants, all reported; fixed C).
6. Paired delta-AUCs with bootstrap CIs (D13): uniform vs full model
   (ablation); DSS+TMB+IC vs TMB+IC (Table 5); TMB-conditional vs TMB.
7. Subtype-stratified AUC (D14) and OOF calibration slope/intercept (D13).

Outputs:
  results/aim11_what_score_measures.csv
  results/aim11_variants.csv
  results/aim11_paired_delta.csv
  results/aim11_random_panels.csv
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from oppti_immune.pies_dss import (
    load_cohort, madz_outliers, module_features, nested_cv_dss,
)
from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("21_what_score_measures")
logging.getLogger("pies_dss").setLevel(logging.WARNING)

N_BOOT = 2000
SEED = 42


def oof_ridge(X: pd.DataFrame, y_nr: np.ndarray, C: float = 3.0,
              scale: pd.Series | None = None) -> np.ndarray:
    """Pooled 5-fold OOF predictions of a fixed-C L2 logistic."""
    Xv = X.values.astype(float)
    if scale is not None:
        Xv = Xv * np.asarray(scale, float)
    oof = np.full(len(Xv), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xv, y_nr):
        m = LogisticRegression(penalty="l2", C=C, solver="liblinear", max_iter=5000)
        m.fit(Xv[tr], y_nr[tr])
        oof[te] = m.predict_proba(Xv[te])[:, 1]
    return oof


def nested_oof_generic(feats: pd.DataFrame, y: pd.Series, seed: int = SEED) -> np.ndarray:
    """Nested CV on arbitrary feature counts (no 8-module weight machinery)."""
    from sklearn.model_selection import StratifiedKFold
    from oppti_immune.pies_dss import fit_dss, select_C_via_inner_cv
    y_nr = (1 - y.astype(int)).values
    oof = np.full(len(feats), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(feats, y):
        C = select_C_via_inner_cv(feats.iloc[tr], y_nr[tr], seed=seed)
        m = fit_dss(feats.iloc[tr], y_nr[tr], C)
        oof[te] = m.predict_proba(feats.iloc[te].values)[:, 1]
    return oof


def paired_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Bootstrap paired delta-AUC (a minus b) with 95% CI."""
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        deltas.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.asarray(deltas)
    return {"delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_boot": float(np.mean(d <= 0) * 2)}


def main() -> None:
    cfg = load_config("config/config.yaml")
    panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    z_thr = cfg.get("concordance", {}).get("outlier_z_threshold", 1.645)

    rna, y = load_cohort("IMvigor210", cfg)
    y_nr = (1 - y.astype(int)).values
    outliers = madz_outliers(rna, panel["gene"].tolist(), z_threshold=z_thr)
    feats = module_features(outliers, panel, pt_frac)

    pg = [g for g in panel["gene"] if g in outliers.columns]  # genes present in RNA

    iof = pd.read_csv(results_path(cfg, "aim4_pies_dss_iof_scores.csv")).set_index("sample")
    dss = iof["PIES-DSS"].reindex(feats.index).values
    clin = pd.read_csv("data/immunotherapy/IMvigor210_clinical.csv", index_col=0)
    tmb = pd.to_numeric(clin["FMOne mutation burden per MB"], errors="coerce").reindex(feats.index)
    sf = pd.to_numeric(clin["sizeFactor"], errors="coerce").reindex(feats.index)
    subtype = clin["TCGA Subtype"].reindex(feats.index)

    # ── 1. negative controls ──────────────────────────────────────────────
    med = rna.median()
    mad = (rna - med).abs().median().replace(0, 1e-9)
    Z = (rna - med) / mad
    binall = (Z.abs() >= z_thr).astype(float)
    burden = binall.sum(axis=1).reindex(feats.index)
    cc = tmb.notna()
    ctrl = [
        {"control": "genome-wide outlier burden (|Z|>=1.645 count)",
         "auc_nonresponse": float(roc_auc_score(y_nr, burden.values)),
         "spearman_with_DSS": float(spearmanr(burden, dss).statistic)},
        {"control": "TMB (negated)", "auc_nonresponse": float(roc_auc_score(y_nr[cc], -tmb[cc].values)),
         "spearman_with_DSS": float(spearmanr(tmb[cc], dss[cc]).statistic)},
        {"control": "library size (sizeFactor)",
         "auc_nonresponse": float(roc_auc_score(y_nr[~sf.isna()], -sf[~sf.isna()].values)),
         "spearman_with_DSS": float(spearmanr(sf[~sf.isna()], dss[~sf.isna()].statistic if hasattr(dss, 'statistic') else dss[~sf.isna()]).statistic)},
        {"control": "IEP-panel outlier burden (137 genes)",
         "auc_nonresponse": float(roc_auc_score(y_nr, outliers[pg].sum(axis=1).values)),
         "spearman_with_DSS": float(spearmanr(outliers[pg].sum(axis=1), pd.Series(dss, index=feats.index)).statistic)},
    ]
    ctrl_df = pd.DataFrame(ctrl)
    logger.info("\n" + ctrl_df.to_string(index=False))

    # ── 2. random matched panels ──────────────────────────────────────────
    dec_mean = pd.qcut(rna.mean(axis=0), 10, labels=False, duplicates="drop")
    dec_var = pd.qcut(rna.var(axis=0), 10, labels=False, duplicates="drop")
    cell = pd.DataFrame({"dm": dec_mean, "dv": dec_var})
    panel_genes = set(pg)
    rng = np.random.default_rng(7)
    mod_of = panel.set_index("gene")["module"]
    pan_aucs = []
    for it in range(500):
        genes, mods = [], []
        for g in pg:
            dm, dv = cell.loc[g, "dm"], cell.loc[g, "dv"]
            cand = cell.index[(cell.dm == dm) & (cell.dv == dv) & (~cell.index.isin(panel_genes))]
            if len(cand) == 0:
                cand = cell.index[(cell.dm == dm) & (~cell.index.isin(panel_genes))]
            cand = [c for c in cand if c not in genes]
            if len(cand) == 0:
                cand = [c for c in cell.index if (c not in genes) and (c not in panel_genes)]
            pick = cand[rng.integers(0, len(cand))]
            genes.append(pick); mods.append(mod_of[g])
        fake = pd.DataFrame({"gene": genes, "module": mods})
        fx = module_features(binall, fake, pt_frac * 0)
        pan_aucs.append(roc_auc_score(y_nr, oof_ridge(fx, y_nr)))
    pan_aucs = np.asarray(pan_aucs)
    dss_auc = float(roc_auc_score(y_nr, dss))
    pct = float(np.mean(pan_aucs >= dss_auc) * 100)
    logger.info("random panels: mean %.3f, sd %.3f, max %.3f | DSS %.3f -> percentile %.1f",
                pan_aucs.mean(), pan_aucs.std(), pan_aucs.max(), dss_auc, pct)
    pd.DataFrame({"auc": pan_aucs}).to_csv(results_path(cfg, "aim11_random_panels.csv"), index=False)

    # ── 3. variant set (all reported, none replaces the shipped model) ────
    rows = [{"variant": "shipped (binary |Z| events, pt_frac weights)", "auc": dss_auc}]
    uni = module_features(outliers, panel, pt_frac * 0)
    nc_u = nested_cv_dss(uni, y, seed=SEED)
    rows.append({"variant": "ablation uniform (pt_frac = 0)",
                 "auc": float(roc_auc_score(y_nr, nc_u["out_of_fold"]["PIES-DSS"]))})
    up = (Z[pg] > z_thr).astype(float)
    dn = (Z[pg] < -z_thr).astype(float)
    fs = pd.concat([module_features(up, panel, pt_frac).add_suffix("_up"),
                    module_features(dn, panel, pt_frac).add_suffix("_down")], axis=1)
    s_updn = nested_oof_generic(fs, y)
    rows.append({"variant": "signed events (16 features: 8 modules x up/down)", "auc": float(roc_auc_score(y_nr, s_updn))})
    fz = module_features(Z.abs()[pg], panel, pt_frac)
    nc_c = nested_cv_dss(fz, y, seed=SEED)
    rows.append({"variant": "unsigned continuous |Z| module sums", "auc": float(roc_auc_score(y_nr, np.asarray(nc_c["out_of_fold"]["PIES-DSS"], float)))})
    # size-factor normalization sensitivity
    raw = pd.read_csv("data/immunotherapy/IMvigor210_expr.csv", index_col=0).T
    raw = raw.loc[feats.index]
    sfv = sf.reindex(feats.index).fillna(1.0)
    rna_sf = np.log2(raw.div(sfv, axis=0) + 1)
    m2 = rna_sf.median(); d2 = (rna_sf - m2).abs().median().replace(0, 1e-9)
    Zsf = (rna_sf - m2) / d2
    out_sf = (Zsf[pg].abs() >= z_thr).astype(float)
    f_sf = module_features(out_sf, panel, pt_frac)
    nc_sf = nested_cv_dss(f_sf, y, seed=SEED)
    rows.append({"variant": "sizeFactor-normalized counts (main model rerun)",
                 "auc": float(roc_auc_score(y_nr, np.asarray(nc_sf["out_of_fold"]["PIES-DSS"], float)))})
    var_df = pd.DataFrame(rows)
    logger.info("\n" + var_df.to_string(index=False))

    # ── 4. prior-leverage test (gene-level ridge, fixed C) ────────────────
    w_g = 1.0 + pt_frac.reindex(pg).fillna(0) / 100.0
    Xg = outliers[pg]
    for lam, name in [(0, "lambda=0 (uniform)"), (3, "lambda=+3 (prior-promoted)"),
                      (-3, "lambda=-3 (inverse)"), (3.5, "lambda=+3, shuffled prior")]:
        if "shuffled" in name:
            wj = pd.Series(w_g.values, index=rng.permutation(w_g.index)).reindex(pg)
        else:
            wj = w_g
        scale = (wj ** lam).values if lam != 0 else None
        auc = roc_auc_score(y_nr, oof_ridge(Xg, y_nr, C=3.0, scale=scale))
        var_df = pd.concat([var_df, pd.DataFrame(
            [{"variant": f"gene-level events ridge, {name}", "auc": float(auc)}])], ignore_index=True)
    logger.info("\n" + var_df.tail(4).to_string(index=False))

    # ── 5. TMB-conditional learning + paired deltas ───────────────────────
    ccmask = cc.values
    f234, y234 = feats[ccmask], y_nr[ccmask]
    t234 = tmb[ccmask].values
    Xcond = pd.concat([f234, pd.Series(t234 * 100.0, index=f234.index, name="tmb_scaled")], axis=1)
    cond_oof = oof_ridge(Xcond, y234, C=3.0)
    var_df = pd.concat([var_df, pd.DataFrame(
        [{"variant": "TMB-conditional (TMB unpenalized + modules, n=234)",
          "auc": float(roc_auc_score(y234, cond_oof))}])], ignore_index=True)
    save_csv(var_df, results_path(cfg, "aim11_variants.csv"), index=False)
    logger.info("\n" + var_df.to_string(index=False))

    uni_full = np.asarray(nc_u["out_of_fold"]["PIES-DSS"], float)
    shipped = np.asarray(nested_cv_dss(feats, y, seed=SEED)["out_of_fold"]["PIES-DSS"], float)
    # Table-5 models (complete cases), same samples/folds
    ic = pd.to_numeric(clin["IC Level"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    ic234 = (ic.reindex(feats.index) >= 2).astype(float)[ccmask].values
    d234 = pd.Series(dss, index=feats.index)[ccmask].values
    Xb = pd.DataFrame({"tmb": t234, "ic": ic234})
    base_oof = oof_ridge(Xb, y234, C=1.0)
    Xd = pd.DataFrame({"dss": d234, "tmb": t234, "ic": ic234})
    dss_oof = oof_ridge(Xd, y234, C=1.0)
    deltas = pd.DataFrame([
        {"comparison": "uniform vs shipped (ablation, d4)", **paired_delta(y_nr, uni_full, shipped)},
        {"comparison": "DSS+TMB+IC vs TMB+IC (Table 5, n=234)", **paired_delta(y234, dss_oof, base_oof)},
        {"comparison": "TMB-conditional vs TMB alone (n=234)", **paired_delta(y234, cond_oof, -t234)},
    ])
    save_csv(deltas, results_path(cfg, "aim11_paired_delta.csv"), index=False)
    logger.info("\n" + deltas.to_string(index=False))

    # ── 6. subtype AUC + calibration ──────────────────────────────────────
    sub_rows = []
    for s, idx in pd.Series(dss, index=feats.index).groupby(subtype).groups.items():
        ii = feats.index.get_indexer(idx)
        yy = y_nr[ii]
        if len(set(yy)) == 2 and len(yy) >= 20:
            sub_rows.append({"subtype": s, "n": int(len(yy)),
                             "auc": float(roc_auc_score(yy, dss[ii]))})
    # calibration: y ~ a + b*logit(p)
    p = np.clip(dss, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    cal = LogisticRegression(C=1e6).fit(lg.reshape(-1, 1), y_nr)
    ctrl_df = pd.concat([ctrl_df, pd.DataFrame(sub_rows)], ignore_index=True)
    ctrl_df = pd.concat([ctrl_df, pd.DataFrame([
        {"control": "calibration intercept (0 = calibrated)", "auc_nonresponse": float(cal.intercept_[0])},
        {"control": "calibration slope (1 = calibrated)", "auc_nonresponse": float(cal.coef_[0][0])},
        {"control": "random-panel percentile of DSS (%)", "auc_nonresponse": pct},
    ])], ignore_index=True)
    save_csv(ctrl_df, results_path(cfg, "aim11_what_score_measures.csv"), index=False)
    logger.info("\n" + ctrl_df.to_string(index=False))


if __name__ == "__main__":
    main()

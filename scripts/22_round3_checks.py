#!/usr/bin/env python
"""
Step 22 — Round-3 checks (evidence gaps)
========================================
Review round 3, items 1-6 (quick statistics; v2 freeze is a separate step):

1. Burden covariate: module features with genome-wide burden as an
   (effectively unpenalized, scaled) covariate; paired dAUC vs modules alone.
   Plus burden-normalized variant: module events / total panel events.
2. Calibrated random-panel comparator: IEP panel scored with the same
   fixed-C ridge as the 500 matched random panels -> percentile in that
   distribution (replaces the 97th-percentile claim).
3. Winsorized continuous comparators: signed and unsigned module means of
   MAD-Z clipped at +/-3 (the shipped 0.500/0.504 used unclipped sums).
4. Paired dAUCs for Table 6 rows vs the shipped model (signed events,
   sizeFactor normalization, winsorized variants, burden-ratio, burden
   covariate) and lambda=+3 vs lambda=0 (prior-leverage ridge).
5. Stratified AUCs with n and bootstrap CIs: immune phenotype + TCGA
   molecular subtype (the "TCGA Subtype" column holds clusters I-IV,
   not pathological stage).
6. Calibration fix: nested (5-fold) Platt recalibration of the pooled OOF
   probabilities; report post-fix slope/intercept.

Output: results/aim12_round3_checks.csv
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from oppti_immune.pies_dss import (
    fit_dss, load_cohort, madz_outliers, module_features, nested_cv_dss,
    select_C_via_inner_cv,
)
from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("22_round3_checks")
logging.getLogger("pies_dss").setLevel(logging.WARNING)

N_BOOT = 2000
SEED = 42


def oof_ridge(X: pd.DataFrame, y_nr: np.ndarray, C: float = 3.0,
              scale: pd.Series | None = None) -> np.ndarray:
    Xv = X.values.astype(float)
    if scale is not None:
        Xv = Xv * np.asarray(scale, float)
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
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_boot": float(np.mean(d <= 0) * 2)}


def nested_oof_generic(feats: pd.DataFrame, y: pd.Series, seed: int = SEED) -> np.ndarray:
    y_nr = (1 - y.astype(int)).values
    oof = np.full(len(feats), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(feats, y):
        C = select_C_via_inner_cv(feats.iloc[tr], y_nr[tr], seed=seed)
        m = fit_dss(feats.iloc[tr], y_nr[tr], C)
        oof[te] = m.predict_proba(feats.iloc[te].values)[:, 1]
    return oof


def oof_auc(df: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, float]:
    nc = nested_cv_dss(df, y, seed=SEED)
    oof = np.asarray(nc["out_of_fold"]["PIES-DSS"], float)
    return oof, float(roc_auc_score((1 - y.astype(int)).values, oof))


def main() -> None:
    cfg = load_config("config/config.yaml")
    panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    z_thr = cfg.get("concordance", {}).get("outlier_z_threshold", 1.645)

    rna, y = load_cohort("IMvigor210", cfg)
    y_nr = (1 - y.astype(int)).values
    med = rna.median()
    mad = (rna - med).abs().median().replace(0, 1e-9)
    Z = (rna - med) / mad
    pg = [g for g in panel["gene"] if g in Z.columns]
    outliers = madz_outliers(rna, panel["gene"].tolist(), z_threshold=z_thr)
    feats = module_features(outliers, panel, pt_frac)
    iof = pd.read_csv(results_path(cfg, "aim4_pies_dss_iof_scores.csv")).set_index("sample")
    dss = iof["PIES-DSS"].reindex(feats.index).values
    clin = pd.read_csv("data/immunotherapy/IMvigor210_clinical.csv", index_col=0)
    tmb = pd.to_numeric(clin["FMOne mutation burden per MB"], errors="coerce").reindex(feats.index)
    sf = pd.to_numeric(clin["sizeFactor"], errors="coerce").reindex(feats.index)

    rows: list[dict] = []

    # ── shipped OOF (same folds throughout) ───────────────────────────────
    shipped, auc_ship = oof_auc(feats, y)
    rows.append({"section": "variants", "item": "shipped model (reference)",
                 "auc": auc_ship})

    # ── 1. burden covariate + burden-normalized variant ──────────────────
    burden_all = (Z.abs() >= z_thr).astype(float).sum(axis=1).reindex(feats.index)
    mod_only = oof_ridge(feats, y_nr)
    auc_mod = float(roc_auc_score(y_nr, mod_only))
    # unpenalized adjustment on the OOF logit: [logit(DSS), burden] vs [logit(DSS)]
    lg_dss = np.log(np.clip(dss, 1e-6, 1 - 1e-6) /
                    (1 - np.clip(dss, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    b_sd = ((burden_all - burden_all.mean()) / burden_all.std()).values.reshape(-1, 1)
    oof_adj = np.full(len(y_nr), np.nan)
    oof_base = np.full(len(y_nr), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(lg_dss, y_nr):
        a = LogisticRegression(C=1e6, max_iter=5000).fit(np.hstack([lg_dss, b_sd])[tr], y_nr[tr])
        oof_adj[te] = a.predict_proba(np.hstack([lg_dss, b_sd])[te])[:, 1]
        r = LogisticRegression(C=1e6, max_iter=5000).fit(lg_dss[tr], y_nr[tr])
        oof_base[te] = r.predict_proba(lg_dss[te])[:, 1]
    pd_b = paired_delta(y_nr, oof_adj, oof_base)
    from scipy.stats import chi2
    Xm = np.hstack([lg_dss, b_sd])
    m_full = LogisticRegression(C=1e6, max_iter=5000).fit(Xm, y_nr)
    m_red = LogisticRegression(C=1e6, max_iter=5000).fit(lg_dss, y_nr)
    ll_full = m_full.predict_log_proba(Xm)[np.arange(len(y_nr)), y_nr].sum()
    ll_red = m_red.predict_log_proba(lg_dss)[np.arange(len(y_nr)), y_nr].sum()
    lrt_p = float(chi2.sf(2 * (ll_full - ll_red), 1))
    cc_t = tmb.notna().values
    tmb_sd = ((tmb[cc_t] - tmb[cc_t].mean()) / tmb[cc_t].std()).values.reshape(-1, 1)
    oof_tb = np.full(cc_t.sum(), np.nan)
    lg_cc, y_cc = lg_dss[cc_t], y_nr[cc_t]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(lg_cc, y_cc):
        a = LogisticRegression(C=1e6, max_iter=5000).fit(
            np.hstack([lg_cc, tmb_sd])[tr], y_cc[tr])
        oof_tb[te] = a.predict_proba(np.hstack([lg_cc, tmb_sd])[te])[:, 1]
    pd_t = paired_delta(y_cc, oof_tb, oof_base[cc_t])
    rows += [
        {"section": "burden_covariate", "item": "modules alone (fixed C=3 ridge)",
         "auc": auc_mod},
        {"section": "burden_covariate", "item": "logit(DSS) + burden (unpenalized, nested)",
         "auc": float(roc_auc_score(y_nr, oof_adj)), "delta_auc": pd_b["delta_auc"],
         "ci_lo": pd_b["ci_lo"], "ci_hi": pd_b["ci_hi"], "p_boot": pd_b["p_boot"]},
        {"section": "burden_covariate", "item": "burden LRT p (logit(DSS)+burden vs logit(DSS))",
         "auc": lrt_p},
        {"section": "burden_covariate", "item": "logit(DSS) + TMB (unpenalized, nested, n=234)",
         "auc": float(roc_auc_score(y_cc, oof_tb)), "delta_auc": pd_t["delta_auc"],
         "ci_lo": pd_t["ci_lo"], "ci_hi": pd_t["ci_hi"], "p_boot": pd_t["p_boot"]},
        {"section": "burden_covariate",
         "item": "spearman(burden, DSS)", "auc": float(spearmanr(burden_all, dss).statistic)},
    ]
    total_ev = outliers[pg].sum(axis=1).reindex(feats.index).replace(0, np.nan)
    ratio = feats.div(total_ev, axis=0).fillna(0.0)
    oof_r, auc_ratio = oof_auc(ratio, y)
    pd_r = paired_delta(y_nr, oof_r, shipped)
    rows.append({"section": "burden_covariate",
                 "item": "module events / total panel events (nested CV)",
                 "auc": auc_ratio, "delta_auc": pd_r["delta_auc"],
                 "ci_lo": pd_r["ci_lo"], "ci_hi": pd_r["ci_hi"], "p_boot": pd_r["p_boot"]})

    # ── 2. IEP panel inside the matched random-panel distribution ────────
    uni = module_features(outliers, panel, pt_frac * 0)
    iep_uni_auc = float(roc_auc_score(y_nr, oof_ridge(uni, y_nr)))
    iep_pri_auc = float(roc_auc_score(y_nr, oof_ridge(feats, y_nr)))
    panels = pd.read_csv(results_path(cfg, "aim11_random_panels.csv"))["auc"].values
    pct_uni = float(np.mean(panels >= iep_uni_auc) * 100)
    pct_pri = float(np.mean(panels >= iep_pri_auc) * 100)
    rows += [
        {"section": "random_panel_calibrated", "item": "IEP panel, fixed-C ridge, uniform weights",
         "auc": iep_uni_auc, "p_boot": pct_uni},
        {"section": "random_panel_calibrated", "item": "IEP panel, fixed-C ridge, prior weights",
         "auc": iep_pri_auc, "p_boot": pct_pri},
        {"section": "random_panel_calibrated", "item": "matched random panels: mean",
         "auc": float(panels.mean()), "ci_lo": float(panels.std()),
         "ci_hi": float(panels.max())},
    ]

    # ── 3. winsorized continuous comparators ─────────────────────────────
    Zc = Z.clip(-3, 3)
    oof_wc, auc_wc = oof_auc(module_features(Zc[pg], panel, pt_frac), y)
    pd_wc = paired_delta(y_nr, oof_wc, shipped)
    oof_wu, auc_wu = oof_auc(module_features(Zc[pg].abs(), panel, pt_frac), y)
    pd_wu = paired_delta(y_nr, oof_wu, shipped)
    rows += [
        {"section": "winsorized_continuous",
         "item": "signed module means, MAD-Z winsorized at +/-3", "auc": auc_wc,
         "delta_auc": pd_wc["delta_auc"], "ci_lo": pd_wc["ci_lo"],
         "ci_hi": pd_wc["ci_hi"], "p_boot": pd_wc["p_boot"]},
        {"section": "winsorized_continuous",
         "item": "unsigned module means, |Z| winsorized at 3", "auc": auc_wu,
         "delta_auc": pd_wu["delta_auc"], "ci_lo": pd_wu["ci_lo"],
         "ci_hi": pd_wu["ci_hi"], "p_boot": pd_wu["p_boot"]},
    ]

    # ── 4. Table-6 paired deltas vs shipped ──────────────────────────────
    up = (Z[pg] > z_thr).astype(float)
    dn = (Z[pg] < -z_thr).astype(float)
    fs = pd.concat([module_features(up, panel, pt_frac).add_suffix("_up"),
                    module_features(dn, panel, pt_frac).add_suffix("_down")], axis=1)
    s_updn = nested_oof_generic(fs, y)
    auc_s = float(roc_auc_score(y_nr, s_updn))
    pd_s = paired_delta(y_nr, s_updn, shipped)
    rows.append({"section": "variants",
                 "item": "signed events (16 features: 8 modules x up/down)",
                 "auc": auc_s, "delta_auc": pd_s["delta_auc"], "ci_lo": pd_s["ci_lo"],
                 "ci_hi": pd_s["ci_hi"], "p_boot": pd_s["p_boot"]})
    sfv = sf.reindex(feats.index).fillna(1.0)
    raw = pd.read_csv("data/immunotherapy/IMvigor210_expr.csv", index_col=0).T.loc[feats.index]
    rna_sf = np.log2(raw.div(sfv, axis=0) + 1)
    m2 = rna_sf.median()
    d2 = (rna_sf - m2).abs().median().replace(0, 1e-9)
    out_sf = (Zsf := (rna_sf - m2) / d2)[pg].abs().ge(z_thr).astype(float)
    oof_sf, auc_sf = oof_auc(module_features(out_sf, panel, pt_frac), y)
    pd_sf = paired_delta(y_nr, oof_sf, shipped)
    rows.append({"section": "variants",
                 "item": "sizeFactor-normalized counts (main model rerun)",
                 "auc": auc_sf, "delta_auc": pd_sf["delta_auc"], "ci_lo": pd_sf["ci_lo"],
                 "ci_hi": pd_sf["ci_hi"], "p_boot": pd_sf["p_boot"]})

    # lambda=+3 vs lambda=0 (gene-level prior-leverage ridge)
    w_g = 1.0 + pt_frac.reindex(pg).fillna(0) / 100.0
    Xg = outliers[pg]
    oof_l0 = oof_ridge(Xg, y_nr, scale=(w_g ** 0).values)
    oof_l3 = oof_ridge(Xg, y_nr, scale=(w_g ** 3).values)
    auc_l0 = float(roc_auc_score(y_nr, oof_l0))
    auc_l3 = float(roc_auc_score(y_nr, oof_l3))
    pd_l = paired_delta(y_nr, oof_l3, oof_l0)
    rows += [
        {"section": "prior_leverage", "item": "gene-level ridge lambda=0 (uniform)",
         "auc": auc_l0},
        {"section": "prior_leverage", "item": "gene-level ridge lambda=+3",
         "auc": auc_l3, "delta_auc": pd_l["delta_auc"], "ci_lo": pd_l["ci_lo"],
         "ci_hi": pd_l["ci_hi"], "p_boot": pd_l["p_boot"]},
    ]

    # ── 5. strata with n and bootstrap CIs ────────────────────────────────
    dss_s = pd.Series(dss, index=feats.index)
    for col, label in [("Immune phenotype", "immune phenotype"),
                       ("TCGA Subtype", "TCGA molecular subtype (cluster)")]:
        strat = clin[col].reindex(feats.index)
        for s, idx in dss_s.groupby(strat).groups.items():
            yy = y_nr[feats.index.get_indexer(idx)]
            if len(set(yy)) < 2 or len(yy) < 20:
                continue
            vals = dss_s.loc[idx].values
            auc = float(roc_auc_score(yy, vals))
            rng = np.random.default_rng(11)
            bs = [roc_auc_score(yy[i], vals[i]) for _ in range(1000)
                  if len(set(yy[i := rng.integers(0, len(yy), len(yy))])) == 2]
            rows.append({"section": "strata", "item": f"{label}: {s}",
                         "n": int(len(yy)), "auc": auc,
                         "ci_lo": float(np.percentile(bs, 2.5)),
                         "ci_hi": float(np.percentile(bs, 97.5))})

    # ── 6. nested Platt recalibration ────────────────────────────────────
    p = np.clip(dss, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    cal_full = LogisticRegression(C=1e6).fit(lg, y_nr)
    oof_p = np.full(len(y_nr), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(lg, y_nr):
        c = LogisticRegression(C=1e6).fit(lg[tr], y_nr[tr])
        oof_p[te] = c.predict_proba(lg[te])[:, 1]
    lg2 = np.log(np.clip(oof_p, 1e-6, 1 - 1e-6) /
                 (1 - np.clip(oof_p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    cal_post = LogisticRegression(C=1e6).fit(lg2, y_nr)
    rows += [
        {"section": "calibration", "item": "pre-fix slope", "auc": float(cal_full.coef_[0][0])},
        {"section": "calibration", "item": "pre-fix intercept", "auc": float(cal_full.intercept_[0])},
        {"section": "calibration", "item": "post-nested-recalibration slope",
         "auc": float(cal_post.coef_[0][0])},
        {"section": "calibration", "item": "post-nested-recalibration intercept",
         "auc": float(cal_post.intercept_[0])},
        {"section": "calibration", "item": "Brier score, pre-fix",
         "auc": float(np.mean((dss - y_nr) ** 2))},
        {"section": "calibration", "item": "Brier score, post-nested-recalibration",
         "auc": float(np.mean((oof_p - y_nr) ** 2))},
    ]

    out = pd.DataFrame(rows)
    save_csv(out, results_path(cfg, "aim12_round3_checks.csv"), index=False)
    logger.info("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()

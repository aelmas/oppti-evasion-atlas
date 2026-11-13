#!/usr/bin/env python
"""
Step 20 — Incremental-value baselines, feature diagnostics, and OS analysis
==========================================================================
Review follow-up (grill Q5, issues 5-6):

1. TMB and PD-L1 IC level in IMvigor210: univariate AUCs (non-response
   orientation) and pooled 5-fold OOF logistic models (TMB+IC; DSS+TMB+IC)
   on the same samples as the DSS out-of-fold scores.
2. Continuous module-mean features (why binarize?): nested CV on the same
   module structure with continuous MAD-Z features instead of binary Q95
   outlier events.
3. Feature diagnostics for the pt_frac ablation: per-module correlation
   between pt_frac-weighted and uniform features (does the prior change
   the features at all?), and pt_frac vs per-gene protein-RNA Spearman rho.
4. Overall survival: Cox PH per SD of the DSS OOF score (+ TMB), and KM
   log-rank across score tertiles.
5. Readout of per-repeat full-refit permutation p values from step 18.

Outputs:
  results/aim10_baselines_incremental.csv
  results/aim10_feature_diagnostics.csv
  results/aim10_os.csv
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

from oppti_immune.pies_dss import (
    bootstrap_ci, load_cohort, madz_outliers, module_features, nested_cv_dss,
)
from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("20_incremental")


def main() -> None:
    cfg = load_config("config/config.yaml")
    panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    z_thr = cfg.get("concordance", {}).get("outlier_z_threshold", 1.645)

    rna, y = load_cohort("IMvigor210", cfg)
    outliers = madz_outliers(rna, panel["gene"].tolist(), z_threshold=z_thr)
    feats = module_features(outliers, panel, pt_frac)
    y_nr = (1 - y.astype(int))

    # ── 1. continuous module means (nested CV, same structure) ──
    med = rna.median()
    mad = (rna - med).abs().median().replace(0, 1e-9)
    zmat = (rna - med) / mad
    cont = module_features(zmat, panel, pt_frac)
    nc = nested_cv_dss(cont, y, seed=42)
    cont_auc = bootstrap_ci(y_nr.values,
                            np.asarray(nc["out_of_fold"]["PIES-DSS"], float))
    logger.info("continuous module means: pooled OOF AUC %.3f [%.3f,%.3f] "
                "(fold mean %.3f)", cont_auc["auc"], cont_auc["ci_lo"],
                cont_auc["ci_hi"], nc["auc_mean"])

    # ── 2. feature diagnostics: pt_frac-weighted vs uniform ──
    uni = module_features(outliers, panel, pt_frac * 0)
    rows = []
    for m in feats.columns:
        r = float(np.corrcoef(feats[m].values, uni[m].values)[0, 1])
        rows.append({"module": m, "pearson_r_weighted_vs_uniform": r})
    diag = pd.DataFrame(rows)
    # pt_frac vs per-gene rho (if the per-gene null file carries rho)
    try:
        genes = pd.read_csv(results_path(cfg, "aim7_discordance_null_genes.csv"))
        rho_col = [c for c in genes.columns if "rho" in c.lower()]
        if rho_col:
            mg = pt_frac.rename("pt_frac").to_frame().join(
                genes.set_index("gene")[[rho_col[0]]], how="inner")
            r_pf = float(spearmanr(mg.pt_frac, mg[rho_col[0]]).statistic)
            logger.info("pt_frac vs per-gene rho: Spearman r = %.3f (n=%d)",
                        r_pf, len(mg))
            diag_attr = {"pt_frac_vs_rho_spearman": r_pf,
                         "n_genes_ptfrac_rho": len(mg)}
        else:
            diag_attr = {}
    except FileNotFoundError:
        diag_attr = {}
    save_csv(diag, results_path(cfg, "aim10_feature_diagnostics.csv"), index=False)
    logger.info("weighted-vs-uniform feature correlation: min %.4f, max %.4f",
                diag.pearson_r_weighted_vs_uniform.min(),
                diag.pearson_r_weighted_vs_uniform.max())

    # ── 3. TMB / IC incremental value ──
    clin = pd.read_csv("data/immunotherapy/IMvigor210_clinical.csv", index_col=0)
    iof = pd.read_csv(results_path(cfg, "aim4_pies_dss_iof_scores.csv"))
    iof = iof.set_index("sample")
    tmb = pd.to_numeric(clin["FMOne mutation burden per MB"], errors="coerce")
    ic_raw = clin["IC Level"].astype(str)
    ic_num = pd.to_numeric(ic_raw.str.extract(r"(\d+)")[0], errors="coerce")
    ic_high = (ic_num >= 2).astype(float)
    df = pd.DataFrame({
        "y_nr": iof["nonresponder"].astype(int),
        "dss": iof["PIES-DSS"],
        "hand": iof["PIES (hand)"],
        "tis": iof["TIS"],
    }).join(tmb.rename("tmb"), how="left").join(ic_high.rename("ic_high"), how="left")
    df = df.dropna(subset=["tmb"])
    logger.info("covariate analysis n=%d (tmb non-missing)", len(df))
    yv = df.y_nr.values

    def pooled_cv(cols):
        X = df[cols].values
        oof = np.full(len(df), np.nan)
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for tr, te in skf.split(X, yv):
            lr = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                    max_iter=5000)
            lr.fit(X[tr], yv[tr])
            oof[te] = lr.predict_proba(X[te])[:, 1]
        return float(roc_auc_score(yv, oof))

    base = {
        "DSS (OOF, fixed)": float(roc_auc_score(yv, df.dss.values)),
        "TMB (higher→response, negated)": float(roc_auc_score(yv, -df.tmb.values)),
        "PD-L1 IC≥2 (higher→response, negated)": float(roc_auc_score(yv, -df.ic_high.values)),
        "TMB+IC (CV logistic)": pooled_cv(["tmb", "ic_high"]),
        "DSS+TMB+IC (CV logistic)": pooled_cv(["dss", "tmb", "ic_high"]),
        "DSS+TMB+IC+subtype (CV logistic)": None,
    }
    sub = clin["TCGA Subtype"]
    df2 = df.join(pd.get_dummies(sub).add_prefix("sub_"), how="left").fillna(0)
    subcols = [c for c in df2.columns if c.startswith("sub_")]
    X2 = df2[["dss", "tmb", "ic_high"] + subcols].values
    y2 = df2.y_nr.values
    oof2 = np.full(len(df2), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X2, y2):
        lr = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=5000)
        lr.fit(X2[tr], y2[tr])
        oof2[te] = lr.predict_proba(X2[te])[:, 1]
    base["DSS+TMB+IC+subtype (CV logistic)"] = float(roc_auc_score(y2, oof2))
    inc = pd.DataFrame(list(base.items()), columns=["model", "pooled_oof_auc"])
    save_csv(inc, results_path(cfg, "aim10_baselines_incremental.csv"), index=False)
    logger.info("\n" + inc.to_string(index=False))

    # ── 4. overall survival ──
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    osdf = pd.DataFrame({
        "dss": df.dss, "tmb": df.tmb, "y_nr": df.y_nr,
        "os": clin["os"].reindex(df.index), "event": clin["censOS"].reindex(df.index),
    }).dropna()
    osdf["dss_z"] = (osdf.dss - osdf.dss.mean()) / osdf.dss.std(ddof=0)
    cph = CoxPHFitter().fit(osdf[["dss_z", "os", "event"]], "os", "event")
    hr1 = float(np.exp(cph.params_["dss_z"])); p1 = float(cph.summary.loc["dss_z", "p"])
    cph2 = CoxPHFitter().fit(osdf[["dss_z", "tmb", "os", "event"]], "os", "event")
    hr2 = float(np.exp(cph2.params_["dss_z"])); p2 = float(cph2.summary.loc["dss_z", "p"])
    q1, q2 = osdf.dss.quantile([1/3, 2/3])
    lo = osdf[osdf.dss <= q1]; hi = osdf[osdf.dss >= q2]
    lr = logrank_test(lo.os, hi.os, event_observed_A=lo.event, event_observed_B=hi.event)
    osrows = [
        {"analysis": "Cox PH, DSS_z", "hr_per_sd": hr1, "p": p1, "n": len(osdf)},
        {"analysis": "Cox PH, DSS_z adjusted for TMB", "hr_per_sd": hr2, "p": p2, "n": len(osdf)},
        {"analysis": "log-rank, bottom vs top tertile", "hr_per_sd": np.nan,
         "p": float(lr.p_value), "n": len(lo) + len(hi)},
    ]
    osr = pd.DataFrame(osrows)
    save_csv(osr, results_path(cfg, "aim10_os.csv"), index=False)
    logger.info("\n" + osr.to_string(index=False))

    # ── 5. per-repeat full-refit p (step 18 readout) ──
    rep = pd.read_csv(results_path(cfg, "aim8_repeated_cv.csv"))
    logger.info("per-repeat full-refit p: %s",
                np.array2string(rep.perm_p_fullrefit.values, precision=3))


if __name__ == "__main__":
    main()

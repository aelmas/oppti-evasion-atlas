#!/usr/bin/env python
"""
Step 24 — v2 evaluation (per V2_FREEZE.md, frozen 16 Nov 2026)
==============================================================
All model choices below follow the frozen spec in V2_FREEZE.md; nothing here
may alter that file. Three blocks:

1. Selection-honest permutation null (200 full-refit permutations):
   every variant in the declared Table-6 family is scored inside each
   permutation (features rebuilt, model refit on permuted labels, fixed
   C = 3, folds seeded 10_000 + p to mirror script 18), and the maximum
   variant-AUC per permutation is the null draw. Each observed variant AUC
   is tested against that max-statistic null — including the shipped model
   for comparability.
2. 20-repeat nested CV (full inner C selection, as script 18) for the two
   declared v2 variants: signed events and sizeFactor normalization.
3. TMB+IC incremental model: TMB+IC vs +v2 (signed, +sizeFactor) on
   complete cases, pooled 5-fold CV logistic, paired bootstrap ΔAUC.

Outputs:
  results/aim14_v2_observed.csv           observed fixed-C AUCs, all variants
  results/aim14_v2_maxnull.csv            per-permutation per-variant AUCs
  results/aim14_v2_selection_honest.csv   max-null p-values per variant
  results/aim14_v2_repeat_cv.csv          20-repeat nested CV, v2 variants
  results/aim14_v2_incremental.csv        TMB+IC incremental models
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from oppti_immune.pies_dss import (
    load_cohort, madz_outliers, module_features, nested_cv_dss,
)
from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("24_v2_evaluation")
logging.getLogger("pies_dss").setLevel(logging.WARNING)

SEED = 42
C_FIXED = 3.0
N_PERM = 200
N_BOOT = 2000
Z_THR = 1.645


def fixed_C_oof(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = LogisticRegression(penalty="l2", C=C_FIXED, solver="liblinear", max_iter=5000)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def paired_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    deltas = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])
              for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))
              if len(set(y[i])) >= 2]
    d = np.asarray(deltas)
    return {"delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
            "p_boot": float(np.mean(d <= 0) * 2)}


def main() -> None:
    cfg = load_config("config/config.yaml")
    panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    rna, y = load_cohort("IMvigor210", cfg)
    y_nr = (1 - y.astype(int)).values
    med, mad = rna.median(), (rna - rna.median()).abs().median().replace(0, 1e-9)
    Z = (rna - med) / mad
    pg = [g for g in panel["gene"] if g in Z.columns]
    outliers = madz_outliers(rna, panel["gene"].tolist(), z_threshold=Z_THR)

    clin = pd.read_csv("data/immunotherapy/IMvigor210_clinical.csv", index_col=0)
    sf = pd.to_numeric(clin["sizeFactor"], errors="coerce").reindex(rna.index).fillna(1.0)
    raw = pd.read_csv("data/immunotherapy/IMvigor210_expr.csv", index_col=0).T.loc[rna.index]
    rna_sf = np.log2(raw.div(sf, axis=0) + 1)
    m2 = rna_sf.median()
    Zsf = (rna_sf - m2) / (rna_sf - m2).abs().median().replace(0, 1e-9)
    out_sf = (Zsf[pg].abs() >= Z_THR).astype(float)

    tmb = pd.to_numeric(clin["FMOne mutation burden per MB"], errors="coerce").reindex(rna.index)
    ic_num = pd.to_numeric(clin["IC Level"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    ic_high = (ic_num >= 2).astype(float).reindex(rna.index)

    # ── variant features (label-independent construction) ──────────────────
    rng0 = np.random.default_rng(11)
    w_g = 1.0 + pt_frac.reindex(pg).fillna(0) / 100.0
    w_shuf = pd.Series(w_g.values, index=rng0.permutation(w_g.index)).reindex(pg)
    feats_ship = module_features(outliers[pg], panel, pt_frac)
    cc = tmb.notna().values
    variants: dict[str, tuple[pd.DataFrame, np.ndarray]] = {
        "shipped (binary |Z| events, pt_frac weights)": (feats_ship, np.ones(len(y_nr), bool)),
        "uniform gene weights (pt_frac = 0)": (module_features(outliers[pg], panel, pt_frac * 0),
                                               np.ones(len(y_nr), bool)),
        "module-preserving pt_frac shuffle": (None, np.ones(len(y_nr), bool)),  # built per draw
        "signed events (16 features)": (pd.concat(
            [module_features((Z[pg] > Z_THR).astype(float), panel, pt_frac).add_suffix("_up"),
             module_features((Z[pg] < -Z_THR).astype(float), panel, pt_frac).add_suffix("_down")],
            axis=1), np.ones(len(y_nr), bool)),
        "signed continuous (unclipped)": (module_features(Z[pg], panel, pt_frac),
                                          np.ones(len(y_nr), bool)),
        "signed continuous (winsorized ±3)": (module_features(Z[pg].clip(-3, 3), panel, pt_frac),
                                              np.ones(len(y_nr), bool)),
        "unsigned continuous (unclipped)": (module_features(Z[pg].abs(), panel, pt_frac),
                                            np.ones(len(y_nr), bool)),
        "unsigned continuous (winsorized ±3)": (module_features(Z[pg].abs().clip(0, 3), panel, pt_frac),
                                                np.ones(len(y_nr), bool)),
        "sizeFactor-normalized counts": (module_features(out_sf, panel, pt_frac),
                                         np.ones(len(y_nr), bool)),
        "gene-level ridge λ = −3": (outliers[pg].mul((w_g ** -3).values, axis=1),
                                    np.ones(len(y_nr), bool)),
        "gene-level ridge λ = +3": (outliers[pg].mul((w_g ** 3).values, axis=1),
                                    np.ones(len(y_nr), bool)),
        "gene-level ridge λ = +3, shuffled prior": (outliers[pg].mul((w_shuf ** 3).values, axis=1),
                                                    np.ones(len(y_nr), bool)),
        "TMB-conditional (modules + TMB, n=234)": (
            pd.concat([feats_ship, (tmb * 100.0).rename("tmb_scaled")], axis=1), cc),
    }

    def modshuffle_feats(seed: int) -> pd.DataFrame:
        sh = pt_frac.copy()
        r = np.random.default_rng(seed)
        for _m, gset in panel.groupby("module")["gene"]:
            idx = [g for g in gset if g in sh.index]
            sh.loc[idx] = r.permutation(sh.loc[idx].values)
        return module_features(outliers[pg], panel, sh)

    # ── observed fixed-C AUCs (seed 42) ────────────────────────────────────
    obs = {}
    for name, (X, mask) in variants.items():
        if X is None:
            X = modshuffle_feats(SEED)
        obs[name] = float(roc_auc_score(y_nr[mask], fixed_C_oof(X[mask].values, y_nr[mask], SEED)))
    obs_df = pd.DataFrame({"variant": list(obs), "observed_fixed_C_auc": list(obs.values()),
                           "n": [int(variants[k][1].sum()) for k in obs]})
    save_csv(obs_df, results_path(cfg, "aim14_v2_observed.csv"), index=False)
    logger.info("\n" + obs_df.to_string(index=False))

    # ── 1. selection-honest max-statistic null (200 full-refit perms) ──────
    names = list(variants)
    null_path = results_path(cfg, "aim14_v2_maxnull.csv")
    if (null_path).exists():
        pp = pd.read_csv(null_path)
        logger.info("loaded existing max-null (%d perms)", len(pp))
    else:
        per_perm = []
        for p in range(N_PERM):
            seed = 10_000 + p
            y_perm = pd.Series(np.random.default_rng(seed).permutation(y.values), index=y.index)
            aucs = {}
            for name in names:
                X, mask = variants[name]
                if X is None:
                    X = modshuffle_feats(seed)
                yv = y_perm.values[mask].astype(int)
                aucs[name] = float(roc_auc_score(yv, fixed_C_oof(X[mask].values, yv, seed)))
            per_perm.append({"perm": p + 1, "null_max": max(aucs.values()), **aucs})
            if (p + 1) % 50 == 0:
                logger.info("  perm %d/%d", p + 1, N_PERM)
        pp = pd.DataFrame(per_perm)
        save_csv(pp, results_path(cfg, "aim14_v2_maxnull.csv"), index=False)

    null_max = pp["null_max"].values
    sel = pd.DataFrame({"variant": obs_df["variant"],
                        "observed_fixed_C_auc": obs_df["observed_fixed_C_auc"],
                        "null_max_mean": float(np.mean(null_max)),
                        "null_max_p95": float(np.percentile(null_max, 95)),
                        "p_vs_maxnull": [float(np.mean(null_max >= a)) for a in obs_df["observed_fixed_C_auc"]]})
    save_csv(sel, results_path(cfg, "aim14_v2_selection_honest.csv"), index=False)
    logger.info("\n" + sel.to_string(index=False))

    # ── 2. 20-repeat nested CV for the two v2 variants ─────────────────────
    from oppti_immune.pies_dss import fit_dss, select_C_via_inner_cv

    def nested_oof_generic(feats: pd.DataFrame, y: pd.Series, seed: int) -> np.ndarray:
        """Nested CV on arbitrary feature counts (mirrors script 21 exactly)."""
        y_nr_ = (1 - y.astype(int)).values
        oof = np.full(len(feats), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(feats, y):
            C = select_C_via_inner_cv(feats.iloc[tr], y_nr_[tr], seed=seed)
            m = fit_dss(feats.iloc[tr], y_nr_[tr], C)
            oof[te] = m.predict_proba(feats.iloc[te].values)[:, 1]
        return oof

    rep_rows = []
    for name in ["signed events (16 features)", "sizeFactor-normalized counts"]:
        X = variants[name][0]
        for i in range(20):
            oof = nested_oof_generic(X, y, seed=1 + i)  # seeds 1..20, mirroring script 18
            rep_rows.append({"variant": name, "repeat": i + 1,
                             "pooled_auc": float(roc_auc_score(y_nr, oof))})
        aucs = [r["pooled_auc"] for r in rep_rows if r["variant"] == name]
        logger.info("%s: 20-repeat nested CV %.3f ± %.3f", name, np.mean(aucs), np.std(aucs))
    save_csv(pd.DataFrame(rep_rows), results_path(cfg, "aim14_v2_repeat_cv.csv"), index=False)

    # ── 3. TMB+IC incremental (complete cases on tmb) ──────────────────────
    dfin = pd.DataFrame({"y": y_nr, "tmb": tmb.values, "ic": ic_high.values},
                        index=rna.index).loc[cc]
    v2_sig = fixed_C_oof(variants["signed events (16 features)"][0].values, y_nr, SEED)
    v2_sf = fixed_C_oof(variants["sizeFactor-normalized counts"][0].values, y_nr, SEED)
    Xs = np.column_stack([dfin.tmb.values, dfin.ic.values])
    lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
    inc = {}
    oof_cache = {}
    for label, cols in [("TMB+IC", None),
                        ("signed+TMB+IC", lg(v2_sig[cc])),
                        ("sizeFactor+TMB+IC", lg(v2_sf[cc]))]:
        X = Xs if cols is None else np.column_stack([Xs, cols])
        o = fixed_C_oof(X, dfin.y.values, SEED)
        oof_cache[label] = o
        inc[label] = float(roc_auc_score(dfin.y.values, o))
    rows = [{"model": k, "auc": v} for k, v in inc.items()]
    for lab, base in [(lab, "TMB+IC") for lab in list(inc)[1:]]:
        pdl = paired_delta(dfin.y.values, oof_cache[lab], oof_cache[base])
        rows.append({"model": f"Δ {lab} vs {base}", **pdl})
    inc_df = pd.DataFrame(rows)
    save_csv(inc_df, results_path(cfg, "aim14_v2_incremental.csv"), index=False)
    logger.info("\n" + inc_df.to_string(index=False))


if __name__ == "__main__":
    main()

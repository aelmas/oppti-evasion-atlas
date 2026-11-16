#!/usr/bin/env python
"""
Step 25 — paired repeat analysis: signed events (v2) vs shipped model
=====================================================================
Reviewer item (verification pass, 16 Nov 2026): the SD across repeats is not
a CI for the difference. Adds:
  1. paired per-repeat comparison (same seeds 1..20 as script 18, same fold
     splits): k/20 repeats where signed beats shipped, mean paired ΔAUC
  2. paired bootstrap ΔAUC on repeat-averaged OOF predictions
Shipped model: module_features(pt_frac) via nested_cv_dss (as script 18,
identical seeds). v2: signed events via nested_oof_generic (as script 24).

Output: results/aim15_v2_paired_repeats.csv
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from oppti_immune.pies_dss import fit_dss, load_cohort, madz_outliers, module_features, nested_cv_dss, select_C_via_inner_cv
from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("25_v2_paired_repeats")
logging.getLogger("pies_dss").setLevel(logging.WARNING)

SEED = 42
Z_THR = 1.645


def nested_oof_generic(feats: pd.DataFrame, y: pd.Series, seed: int) -> np.ndarray:
    """Nested CV on arbitrary feature counts (mirrors script 24 exactly)."""
    y_nr = (1 - y.astype(int)).values
    oof = np.full(len(feats), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(feats, y):
        C = select_C_via_inner_cv(feats.iloc[tr], y_nr[tr], seed=seed)
        m = fit_dss(feats.iloc[tr], y_nr[tr], C)
        oof[te] = m.predict_proba(feats.iloc[te].values)[:, 1]
    return oof


def main() -> None:
    from sklearn.model_selection import StratifiedKFold

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
    feats_ship = module_features(outliers[pg], panel, pt_frac)
    fs = pd.concat(
        [module_features((Z[pg] > Z_THR).astype(float), panel, pt_frac).add_suffix("_up"),
         module_features((Z[pg] < -Z_THR).astype(float), panel, pt_frac).add_suffix("_down")],
        axis=1)

    rows, ship_oofs, sig_oofs = [], [], []
    for i in range(20):
        seed = 1 + i  # mirror script 18's default seeds (1..20)
        nc = nested_cv_dss(feats_ship, y, seed=seed)
        a_ship = float(roc_auc_score(nc["out_of_fold"]["y_nr"],
                                     np.asarray(nc["out_of_fold"]["PIES-DSS"], float)))
        o_ship = np.asarray(nc["out_of_fold"]["PIES-DSS"], float)
        o_sig = nested_oof_generic(fs, y, seed=seed)
        a_sig = float(roc_auc_score(y_nr, o_sig))
        rows.append({"repeat": i + 1, "shipped_auc": a_ship, "signed_auc": a_sig,
                     "delta": a_sig - a_ship})
        ship_oofs.append(o_ship)
        sig_oofs.append(o_sig)
        if (i + 1) % 5 == 0:
            logger.info("  repeat %d/20", i + 1)

    df = pd.DataFrame(rows)
    k = int((df.delta > 0).sum())
    ship_mean_oof = np.mean(ship_oofs, axis=0)
    sig_mean_oof = np.mean(sig_oofs, axis=0)
    rng = np.random.default_rng(0)
    deltas = [roc_auc_score(y_nr[j], sig_mean_oof[j]) - roc_auc_score(y_nr[j], ship_mean_oof[j])
              for j in (rng.integers(0, len(y_nr), len(y_nr)) for _ in range(2000))
              if len(set(y_nr[j])) >= 2]
    d = np.asarray(deltas)
    out = pd.DataFrame([
        {"item": "k/20 repeats where signed beats shipped", "value": k},
        {"item": "mean per-repeat paired ΔAUC", "value": float(df.delta.mean())},
        {"item": "per-repeat ΔAUC range",
         "value": f"[{df.delta.min():+.3f}, {df.delta.max():+.3f}]"},
        {"item": "shipped 20-repeat mean AUC", "value": float(df.shipped_auc.mean())},
        {"item": "signed 20-repeat mean AUC", "value": float(df.signed_auc.mean())},
        {"item": "ΔAUC on repeat-averaged OOF predictions",
         "value": float(roc_auc_score(y_nr, sig_mean_oof) - roc_auc_score(y_nr, ship_mean_oof))},
        {"item": "paired bootstrap CI lo (repeat-averaged)", "value": float(np.percentile(d, 2.5))},
        {"item": "paired bootstrap CI hi (repeat-averaged)", "value": float(np.percentile(d, 97.5))},
    ])
    save_csv(out, results_path(cfg, "aim15_v2_paired_repeats.csv"), index=False)
    logger.info("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()

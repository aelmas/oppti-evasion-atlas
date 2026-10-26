#!/usr/bin/env python
"""
Step 16 — Ablation: does the pt_frac prior actually matter?
==========================================================
Reviewer-driven ablation of the only *prior* ingredient in PIES-DSS (the
per-gene RNA-invisibility fraction, pt_frac, which reweights genes *within*
each module before the learned module weights are applied).

Variants (all share the identical outer/inner fold split, seed 42, so all
numbers are paired):
  full          : shipped PIES-DSS, gene weight = 1 + pt_frac/100
  uniform       : pt_frac forced to 0 for every gene (weight = 1.0)
  shuffled_s*   : pt_frac permuted across genes WITHIN each module
                  (preserves module marginals, breaks gene-level pairing);
                  repeated with seeds 1..N

Outputs
  results/aim5_ablation.csv        : per-variant pooled-IoF AUC with bootstrap
                                     CI, permutation p, mean-of-folds +-std,
                                     learned module weights.
  results/aim5_ablation_delong.csv : paired DeLong p of each variant vs full.

Usage:  python scripts/16_ablate_ptfrac.py [--n-shuffle 10]
"""

import argparse

import numpy as np
import pandas as pd

from oppti_immune.pies_dss import (
    MODULE_ORDER,
    bootstrap_ci,
    delong_paired,
    load_cohort,
    madz_outliers,
    module_features,
    nested_cv_dss,
    permutation_p,
)
from oppti_immune.utils import get_logger, load_config, load_gene_panel, results_path, save_csv

logger = get_logger("16_ablate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-shuffle", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gene_panel = load_gene_panel(cfg)
    pt = pd.read_csv(results_path(cfg, "aim2_post_transcriptional_fraction.csv"))
    pt_frac = pt.groupby("gene")["pct_post_transcriptional"].mean()
    z_thr = cfg.get("concordance", {}).get("outlier_z_threshold", 1.645)

    rna, y = load_cohort("IMvigor210", cfg)
    outliers = madz_outliers(rna, gene_panel["gene"].tolist(), z_threshold=z_thr)
    module_map = gene_panel.set_index("gene")["module"]

    def feats_for(pt_series: pd.Series) -> pd.DataFrame:
        return module_features(outliers, gene_panel, pt_series)

    results: list[dict] = []
    scores: dict[str, tuple[pd.Series, np.ndarray]] = {}  # name -> (y_nr, score)

    def run_variant(name: str, pt_variant: pd.Series) -> None:
        feats = feats_for(pt_variant)
        nc = nested_cv_dss(feats, y, seed=args.seed,
                           hand_weights=cfg["pies"]["module_weights"], tis_scores=None)
        iof = nc["out_of_fold"]
        y_nr = iof["y_nr"]
        score = np.asarray(iof["PIES-DSS"], dtype=float)
        b = bootstrap_ci(y_nr, score)
        w = nc["weights"]
        w_ser = w.set_index("module")["coef_median"] if isinstance(w, pd.DataFrame) else pd.Series(w)
        results.append({
            "variant": name,
            "pooled_auc": b["auc"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
            "perm_p": permutation_p(y_nr, score, n_perm=args.n_perm,
                                    seed=args.seed)["perm_p_two_sided"],
            "fold_mean": nc["auc_mean"], "fold_std": nc["auc_std"],
            "weights": ";".join(f"{m}={float(w_ser[m]):+.3f}" for m in MODULE_ORDER),
        })
        scores[name] = (y_nr, score)
        logger.info(f"  {name:<14s} pooled AUC={b['auc']:.3f} "
                    f"[{b['ci_lo']:.3f},{b['ci_hi']:.3f}] fold {nc['auc_mean']:.3f}")

    run_variant("full", pt_frac)
    run_variant("uniform", pd.Series(0.0, index=pt_frac.index))

    genes_in_panel = [g for g in gene_panel["gene"] if g in pt_frac.index]
    for s in range(1, args.n_shuffle + 1):
        pt_s = pt_frac.copy()
        for mod in MODULE_ORDER:
            gs = [g for g in genes_in_panel if module_map.get(g) == mod and g in pt_s.index]
            if len(gs) > 1:
                pt_s.loc[gs] = pt_s.loc[gs].sample(frac=1.0, random_state=s).values
        run_variant(f"shuffled_s{s}", pt_s)

    df = pd.DataFrame(results)
    y_nr_full, full_score = scores["full"]
    delong = []
    for name, (y_nr_v, score_v) in scores.items():
        if name == "full":
            continue
        delong.append({"variant": name,
                       "delong_p_vs_full": float(delong_paired(y_nr_full, full_score, score_v))})
    save_csv(df, results_path(cfg, "aim5_ablation.csv"), index=False)
    save_csv(pd.DataFrame(delong), results_path(cfg, "aim5_ablation_delong.csv"), index=False)

    logger.info("\npooled out-of-fold results —\n"
                + df.drop(columns=["weights"]).to_string(index=False))
    logger.info("\nlearned module weights —\n"
                + df[["variant", "weights"]].to_string(index=False))


if __name__ == "__main__":
    main()

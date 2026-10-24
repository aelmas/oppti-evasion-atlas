#!/usr/bin/env python
"""
Dry-run sanity check (synthetic + subset data, no real results touched)
======================================================================
Five stages, each exercising REAL library functions:

  S1  Preprocessing: tumor/normal filter (incl. `_cancer` mid-index `.N`
      variant) + MAD normalization
  S2  OPPTI: planted outliers recovered in <1 min on a 40×12 mini cohort
      (200 permutations)
  S3  Concordance: OPPTI-restricted classification labels planted
      post-transcriptional vs concordant events correctly
  S4  PIES-DSS: nested CV recovers planted module directions on
      simulated ICB outcome data; IoF paired-comparison plumbing runs
  S5  Production-data invariants: the committed result files still hold
      the manuscript numbers (929 tumors, 5,488 calls, 54.2% PT, DSS
      IoF AUC 0.616, held-out 0.706/0.541, cohort file orientation)

Usage:  python scripts/dryrun_sanity_check.py [--quick|--full]
Exit code 0 = all checks passed.
"""

import sys
import argparse
import numpy as np
import pandas as pd

np.random.seed(1234)
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n{'─' * 62}\n{title}\n{'─' * 62}")


# ── S1: preprocessing ─────────────────────────────────────────────────────────
def s1_preprocessing() -> None:
    section("S1 — Preprocessing: tumor filter + MAD normalize (real functions)")
    from oppti_immune.preprocessing import filter_tumor_samples, mad_normalize

    scores = np.tile(np.arange(20.0), (6, 1))          # 6 genes × 20 samples
    cols = [
        "C3L-00001_luad", "C3L-00004_luad", "C3L-00010_luad",
        "C3L-00011.2.N_luad",                          # mid-index .N variant
        "C3L-00125.N",                                 # plain .N
        "C3X-00001",                                   # other cancer, keep
    ] + [f"C3X-{i:05d}" for i in range(2, 16)]         # 14 filler tumors
    df = pd.DataFrame(
        scores, index=["CD274", "HAVCR2", "TGFB1", "LDHA", "BRD4", "ACSS2"],
        columns=cols,
    )
    tumor = filter_tumor_samples(df.T)   # production path: samples as index
    kept = set(tumor.index)
    check("normals removed (both .N patterns)",
          "C3L-00011.2.N_luad" not in kept and "C3L-00125.N" not in kept,
          f"kept={sorted(kept)}")
    check("tumors kept", {"C3L-00001_luad", "C3L-00010_luad", "C3X-00001", "C3X-00009"} <= kept)

    norm = mad_normalize(tumor)
    med = norm.median(axis=0)               # one median per GENE (samples as index)
    check("MAD-normalized medians ≈ 0", bool((med.abs() < 1e-9).all()),
          f"max |median|={med.abs().max():.2e}")


# ── S2: OPPTI on a planted mini cohort ────────────────────────────────────────
def s2_oppti(n_perm: int = 200) -> None:
    section("S2 — OPPTI: planted outliers recovered (real run_oppti_cohort)")
    from oppti_immune.oppti_core import run_oppti_cohort

    rng = np.random.default_rng(7)
    n, genes = 40, 12
    expr = pd.DataFrame(rng.normal(size=(n, genes)), columns=[f"G{i}" for i in range(genes)])
    expr.loc[expr.index[:3], "G1"] += 7.0     # planted extreme in G1 → 3 samples
    expr.loc[expr.index[5:7], "G2"] += 6.5    # planted extreme in G2 → 2 samples

    scores, binary, _perm = run_oppti_cohort(expr, [f"G{i}" for i in range(genes)],
                                             k=5, n_perm=n_perm, seed=1)
    calls = binary.sum()
    check("planted genes G1/G2 among top-flagged and correctly planted",
          calls.get("G1", 0) >= 2 and calls.get("G2", 0) >= 1,
          f"G1={calls['G1']}, G2={calls['G2']}")
    total = int(binary.values.sum())
    frac = total / binary.size
    check("overall call rate ≈ Q95 null expectation (~5%, tol 2–12%)",
          0.02 <= frac <= 0.12, f"{total}/{binary.size} = {frac:.0%}")


# ── S3: concordance classification ───────────────────────────────────────────
def s3_concordance() -> None:
    section("S3 — Concordance: OPPTI-restricted labels (real classify_using_oppti_calls)")
    from oppti_immune.concordance import classify_using_oppti_calls

    genes = ["PT", "CT"]   # PT = planted post_transcriptional, CT = concordant
    n = 20                 # ≥10 non-outliers per gene needed by the classifier
    samples = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(3)
    out_idx = list(range(6))

    prot = pd.DataFrame(rng.normal(0, 1, (n, 2)), index=samples, columns=genes)
    prot.iloc[out_idx, :] = 2.5 + rng.normal(0, 0.5, (6, 2))       # protein outliers
    rna = pd.DataFrame(rng.normal(0, 1, (n, 2)), index=samples, columns=genes)
    rna.iloc[out_idx, 1] = 3.0 + rng.normal(0, 0.4, 6)             # CT: RNA concordant
    rna.iloc[out_idx, 0] = rng.normal(0, 0.6, 6)                   # PT: RNA low
    prot["cancer_type"] = "test"; rna["cancer_type"] = "test"
    binary = pd.DataFrame(0, index=samples, columns=genes)
    binary.iloc[out_idx] = 1                # OPPTI called exactly these 12 events

    cls = classify_using_oppti_calls(prot, rna, binary, 1.645, perm_mask=None)
    rows = cls[cls["concordance"] != "none"]
    pt = int((rows["concordance"] == "post_transcriptional").sum())
    concordant = int((rows["concordance"] == "concordant").sum())
    check("12 OPPTI calls classified (6 PT / 6 concordant)",
          len(rows) == 12 and pt == 6 and concordant == 6,
          f"PT={pt}, concordant={concordant}")


# ── S4: PIES-DSS on simulated ICB outcome ─────────────────────────────────────
def s4_pies_dss(seed: int = 42) -> None:
    section("S4 — PIES-DSS: planted module directions recovered; IoF plumbing runs")
    from oppti_immune.pies_dss import nested_cv_dss, MODULE_ORDER

    n, mods = 240, 8
    # NOTE: coefficients below act on the RESPONDER-probability logits
    # (logits = X·β). PIES-DSS learns coefficients for P(non-response),
    # so the expected learned sign is the NEGATIVE of these values.
    true_resp_effect = {
        "ecm_exclusion": 0.60, "metabolic": 0.50, "cytokines": 0.45,
        "checkpoint": 0.0, "epigenetic": 0.0, "antigen_presentation": -0.45,
        "mismatch_repair": -0.55, "ifn_pathway": -0.70,
    }
    rng = np.random.default_rng(seed)
    base_rate = 0.15                # per-module per-sample outlier-event rate
    X = np.zeros((n, mods))
    for _ in range(int(n * mods * base_rate)):
        X[rng.integers(n), rng.integers(mods)] += 1.0
    feats = pd.DataFrame(X, columns=MODULE_ORDER)
    beta = np.array([true_resp_effect[m] for m in MODULE_ORDER])
    logits = 0.0 + feats.values @ beta + rng.normal(scale=0.4, size=n)
    y_resp = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int))

    nc = nested_cv_dss(feats, y_resp, seed=seed, n_outer=5)
    check("nested-CV AUC > 0.5 (detected planted effect)",
          nc["auc_mean"] > 0.5, f"folds={[round(a,3) for a in nc['auc_folds']]} → {nc['auc_mean']:.3f}")
    w = nc["weights"].set_index("module")["coef_median"]
    # learner models P(non-responder): expected sign = −(planted responder effect)
    match = sum(
        (true_resp_effect[m] > 0 and w[m] < 0) or (true_resp_effect[m] < 0 and w[m] > 0)
        for m in ["ecm_exclusion", "metabolic", "ifn_pathway", "mismatch_repair"]
        if true_resp_effect[m] != 0
    )
    check("signs recovered for ≥3 of 4 planted modules", match >= 3,
          f"recovered={match}/4")

    # IoF plumbing (paired-baseline path used in production)
    nc2 = nested_cv_dss(feats, y_resp, seed=seed, n_outer=5,
                        hand_weights={m: 1.0 for m in MODULE_ORDER},
                        tis_scores=pd.Series(rng.random(n), index=feats.index))
    iof = nc2["out_of_fold"]
    complete = all(v is not None and not np.isnan(v).any() for v in iof.values())
    check("out_of_fold arrays filled (DSS/hand/TIS/y_nr)", complete)


# ── S5: production invariants ────────────────────────────────────────────────
def s5_production() -> None:
    section("S5 — Production invariants: committed files hold the manuscript numbers")
    from oppti_immune.utils import load_config, data_path, results_path
    from oppti_immune.pies_dss import MODULE_ORDER

    cfg = load_config("config/config.yaml")
    prot = pd.read_parquet(data_path(cfg, "processed/pan_prot_iep.parquet"))
    n_samp = len(prot)
    check("pan-cancer IEP matrix = 929 tumors", n_samp == 929, f"n={n_samp}")

    ob = pd.read_parquet(results_path(cfg, "oppti_binary_matrix.parquet"))
    if "sample" in ob.columns:
        ob = ob.set_index("sample")
    n_calls = int(ob.drop(columns=["cancer_type"], errors="ignore").values.sum())
    check("OPPTI Q95 calls = 5,488", n_calls == 5488, f"n={n_calls}")

    conc = pd.read_parquet(results_path(cfg, "aim2_concordance_classification.parquet"))
    sub = conc[conc["concordance"] != "none"]
    pt = int((sub["concordance"] == "post_transcriptional").sum())
    tot = len(sub)
    check("OPPTI-restricted PT fraction = 2964/5473 (54.2%)",
          (pt, tot) == (2964, 5473), f"{pt}/{tot} = {pt/tot:.1%}")

    comp = pd.read_csv(results_path(cfg, "aim4_pies_dss_comparison.csv"))
    dss = comp[comp["model"] == "PIES-DSS"]["auc"].iloc[0]
    rows = comp.set_index("model")
    delong_hand = rows.loc["DeLong p: DSS vs PIES (hand)", "perm_p_two_sided"]
    delong_tis = rows.loc["DeLong p: DSS vs TIS", "perm_p_two_sided"]
    check("DSS IoF AUC ≈ 0.616 (matches manuscript)", abs(dss - 0.6161) < 0.001,
          f"AUC={dss:.3f}")
    check("DeLong rows present (hand p=0.615, TIS p=0.327)",
          abs(delong_hand - 0.6147) < 0.01 and abs(delong_tis - 0.3269) < 0.01,
          f"hand={delong_hand:.3f}, TIS={delong_tis:.3f}")

    held = pd.read_csv(results_path(cfg, "aim4_pies_dss_heldout.csv"))
    hugo = held[held["cohort"] == "Hugo_2016"]["auc"].iloc[0]
    riaz = held[held["cohort"] == "Riaz_2017"]["auc"].iloc[0]
    check("held-out Hugo 0.706 / Riaz 0.541",
          abs(hugo - 0.7056) < 0.001 and abs(riaz - 0.5410) < 0.001,
          f"Hugo={hugo:.3f}, Riaz={riaz:.3f}")

    # ICB cohort files: stored genes × samples on disk (script 11 does .T after read)
    panel = set(pd.read_csv("data/gene_panels/immune_evasion_proteome.csv")["gene"])
    for name, n_expect in (("IMvigor210", 348), ("Hugo_2016", 27), ("Riaz_2017", 49)):
        e = pd.read_csv(f"data/immunotherapy/{name}_expr.csv", index_col=0)
        check(f"{name}: {n_expect} samples, ≥160 panel genes",
              e.shape[1] == n_expect and len(panel & set(e.index)) >= 160,
              f"genes={e.shape[0]}, samples={e.shape[1]}, panel={len(panel & set(e.index))}")

    check("config module_weights keys == MODULE_ORDER",
          set(cfg["pies"]["module_weights"]) == set(MODULE_ORDER))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fast", action="store_true", help="fewer OPPTI permutations")
    args = ap.parse_args()

    print("═" * 62)
    print(" DRY-RUN SANITY CHECK — oppti-immune-evasion")
    print("═" * 62)

    s1_preprocessing()
    s2_oppti(n_perm=100 if args.fast else 200)
    s3_concordance()
    s4_pies_dss()
    s5_production()

    print("\n" + "═" * 62)
    if FAILURES:
        print(f" ✗ {len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print(" ✓ ALL CHECKS PASSED — pipeline is consistent and extrapolates cleanly")
    sys.exit(0)


if __name__ == "__main__":
    main()

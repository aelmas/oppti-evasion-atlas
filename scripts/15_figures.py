#!/usr/bin/env python
"""
Step 15: Submission figures F1-F4 (Biomarkers / T&F package)
============================================================
Reads only result files; no model fitting. Writes PDF (vector, print 300dpi
equivalent) + PNG (300 dpi) per figure into figures/.

  F1  Pan-cancer IEP outlier atlas (recurrent proteins x 9 cancer types)
  F2  RNA-protein concordance restricted to OPPTI Q95 calls
  F3  Development cohort (IMvigor210): fold AUCs, pooled IoF ROC, weights
  F4  Held-out validation (Hugo_2016, Riaz_2017)

Usage: python scripts/15_figures.py
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oppti_immune.utils import project_root, load_config  # noqa: E402

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,          # editable text in vector PDFs
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

C_DSS, C_HAND, C_TIS = "#0c6bad", "#f0a13c", "#7b6bb5"
C_HIGHEST = "midnightblue"


def results(name: str) -> pd.DataFrame:
    return pd.read_csv(project_root() / "results" / name)


def save_both(fig: plt.Figure, out: Path) -> None:
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


CANON = {  # cohort code -> display label
    "brca": "BRCA", "ccrcc": "ccRCC", "gbm": "GBM", "hnscc": "HNSCC",
    "lscc": "LSCC", "luad": "LUAD", "ov": "OV", "pdac": "PDAC", "ucec": "UCEC",
}
MODULE_LABEL = {
    "ecm_exclusion": "ECM exclusion", "metabolic": "Metabolic escape",
    "cytokines": "Suppressive cytokines", "checkpoint": "Checkpoint ligands",
    "epigenetic": "Epigenetic", "antigen_presentation": "Antigen presentation",
    "mismatch_repair": "Mismatch repair", "ifn_pathway": "IFN signalling",
}
HAND_W = {  # hand-set weights from config/config.yaml (sign: evasion-positive)
    "checkpoint": 3.0, "antigen_presentation": -2.5, "cytokines": 2.0,
    "ifn_pathway": -2.0, "metabolic": 1.5, "ecm_exclusion": 2.0,
    "epigenetic": 1.0, "mismatch_repair": -1.5,
}


# ── Figure 1 ─────────────────────────────────────────────────────────────────
def fig1(cfg) -> None:
    summ = results("aim1_pan_cancer_oppti_summary.csv")
    recurrent = results("aim1_recurrent_IEP_outliers.csv")
    proteins = recurrent["protein"].tolist()
    modules = dict(zip(recurrent["protein"], recurrent["modules"]))

    mat = (summ[summ.protein.isin(proteins)]
           .assign(ct=lambda d: d.cancer_type.map(CANON))
           .pivot(index="protein", columns="ct", values="pct_outlier"))
    mat = mat.reindex(index=proteins, columns=list(CANON.values()))

    # supporting: overall outlier rates (same data, all 137 IEPs) for context
    overall = summ.groupby("cancer_type")["pct_outlier"].mean()
    overall.index = [CANON.get(c, c.upper()) for c in overall.index]

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.9, 1.5], wspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    im = ax1.imshow(mat.values, cmap="Blues", vmin=0, vmax=20, aspect="auto")
    ax1.set_xticks(range(len(mat.columns)), mat.columns, rotation=45,
                   ha="right")
    ax1.set_yticks(range(len(mat.index)), [
        f"{g}  ({modules[g].split(',')[0]})" for g in mat.index])
    for i, g in enumerate(mat.index):
        for j, v in enumerate(mat.columns):
            v = mat.loc[g, v]
            if not np.isnan(v):
                ax1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                         color="white" if v > 12 else "black")
    ax1.set_title("Outlier rate (%)  — Q95-recurrent IEPs\n(colour: blue = high; blank = not called)")

    rates = (summ.groupby("cancer_type")["pct_outlier"].mean()
             .reindex([k for k in CANON]))
    ax2.barh(np.arange(len(rates)), rates.values, color="#0c6bad",
             height=0.75, edgecolor="none")
    ax2.set_yticks(np.arange(len(rates)),
                   [CANON[c] for c in rates.index], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(rates.values) * 1.35)
    for i, v in enumerate(rates.values):
        ax2.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=6.5)
    ax2.set_xlabel("Mean IEP outlier rate (%)\n(all 137 proteins)", fontsize=7)
    ax2.tick_params(left=False)

    fig.suptitle("Proteomic immune-evasion outliers across nine CPTAC cancer types (n = 929)",
                 y=1.06, fontsize=9)
    save_both(fig, project_root() / "figures" / "Figure1_outlier_atlas")


# ── Figure 2 ─────────────────────────────────────────────────────────────────
def fig2(cfg) -> None:
    frac = results("aim2_post_transcriptional_fraction.csv")
    # federal summary row (cancer_type=="all") = pooled estimate
    # headline numbers: 2,964/5,473 (54.2%) Q95; 1,085/2,305 (47.1%) perm-sig.
    per_gene = (
        frac[frac.cancer_type == "all"]
        .assign(pt_frac=lambda d: d.post_transcriptional / d.n_outlier_total)
        .sort_values("pt_frac", ascending=False))
    top = per_gene.head(15).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.8))

    ax1.bar([0, 1], [54.2, 47.1], width=0.55, color=["#0c6bad", "#5b9bd5"])
    ax1.set_xticks([0, 1], ["All Q95 calls\n(2,964 / 5,473)",
                            "Permutation-significant\n(1,085 / 2,305)"])
    ax1.set_ylim(0, 70)
    ax1.set_ylabel("% post-transcriptional")
    ax1.axhline(0, color="k", lw=0.7)
    for x, v in zip([0, 1], [54.2, 47.1]):
        ax1.text(x, v + 2, f"{v:.1f}%", ha="center", fontsize=8)
    ax1.set_title("RNA-invisible protein outliers\n(OPPTI-restricted denominator)")

    ax2.barh(range(len(top)), top.post_transcriptional / top.n_outlier_total * 100,
             color="#0c6bad")
    ax2.set_yticks(range(len(top)), top.gene)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Post-transcriptional fraction (%)")
    for i, (g, row) in enumerate(top.iterrows()):
        ax2.text(row.post_transcriptional / row.n_outlier_total * 100 - 1, i,
                 f"{row.n_outlier_total}", va="center", ha="right", fontsize=6,
                 color="white")
    ax2.set_title("Top proteins by RNA invisibility\n(n outlier cases, text)")

    fig.suptitle("RNA-protein concordance of CPTAC paired tumours", y=1.06, fontsize=9)
    save_both(fig, project_root() / "figures" / "Figure2_concordance")


# ── Figure 3 ─────────────────────────────────────────────────────────────────
def _roc(ax, y_resp: np.ndarray, score: np.ndarray, label: str, color: str,
         axlabel: bool = False) -> float:
    """ROC vs responders: scores are oriented toward non-response, so use -s."""
    fpr, tpr, _ = roc_curve(y_resp, -np.asarray(score, dtype=float))
    auc = float(np.trapezoid(tpr, fpr))
    ax.plot(fpr, tpr, color=color, lw=1.4,
            label=f"{label} — AUC {auc:.3f}")
    if axlabel:
        ax.plot([0, 1], [0, 1], ls=":", lw=0.8, color="grey")
        ax.set_xlabel("1 − specificity")
        ax.set_ylabel("Sensitivity")
    return auc


def fig3(cfg) -> None:
    folds = results("aim4_pies_dss_nested_cv.csv")
    comp = results("aim4_pies_dss_comparison.csv")
    iof = results("aim4_pies_dss_iof_scores.csv")
    iof["nonresponder"] = iof.nonresponder.astype(int)
    y = (1 - iof.nonresponder).values            # responder labels
    w = results("aim4_pies_dss_weights.csv")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    fig.subplots_adjust(wspace=0.55)

    # A: nested-CV outer fold AUCs
    ax = axes[0]
    ax.bar(np.arange(1, 6), folds.auc, color="#0c6bad", width=0.65)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    mu = float(np.mean(folds.auc))          # population SD, matches manuscript
    sd = float(np.std(folds.auc, ddof=0))
    ax.axhline(mu, color=C_HIGHEST, lw=1.0)
    ax.text(3.1, mu - 0.075, f"mean {mu:.3f} ± {sd:.3f}", fontsize=7,
            color=C_HIGHEST, ha="center")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(1, 6), folds.outer_fold.astype(int))
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("AUC (outer test folds)")
    ax.set_title("A  Nested 5×5 CV, fold AUCs")

    # B: pooled IoF ROC, three models
    ax = axes[1]
    for col, lab, c in [("PIES-DSS", "PIES-DSS", C_DSS),
                        ("PIES (hand)", "PIES (hand)", C_HAND),
                        ("TIS", "TIS", C_TIS)]:
        _roc(ax, y, iof[col].values, lab, c, axlabel=True)
    ax.plot([0, 1], [0, 1], ls=":", lw=0.8, color="grey")
    ax.legend(frameon=False, loc="lower right", fontsize=7,
              handlelength=1.6)
    ax.set_title("B  Pooled out-of-fold ROC (IMvigor210, n = 298)")

    # C: learned vs hand weights
    ax = axes[2]
    order = w.sort_values("coef_median")["module"].tolist()
    wm = w.set_index("module").loc[order]
    pos = np.arange(len(wm))
    ax.hlines(pos, wm.coef_min, wm.coef_max, color="#9dbdd6", lw=3)
    ax.plot(wm.coef_median, pos, "o", color=C_HIGHEST, ms=5,
            label="learned (median,\nmin–max over folds)")
    hw = np.array([HAND_W[m] for m in order])
    ax.plot(hw, pos, "x", color="#c1440e", ms=6, label="hand-set")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(pos, [MODULE_LABEL[m] for m in order], fontsize=7)
    ax.set_xlabel("Module weight (log-odds / event)\n"
                  "● learned (median, whiskers = fold range)   ✕ hand-set",
                  fontsize=7)
    ax.set_title("C  Learned vs hand-set weights")

    fig.suptitle("PIES-DSS development-cohort performance (IMvigor210)",
                 y=1.08, fontsize=9)
    save_both(fig, project_root() / "figures" / "Figure3_development")


# ── Figure 4 ─────────────────────────────────────────────────────────────────
def fig4(cfg) -> None:
    held = results("aim4_pies_dss_heldout.csv")           # PIES-DSS rows only
    honest = results("aim4_honest_validation.csv")        # hand-PIES rows
    tis = results("aim4_AUC_by_cohort.csv")               # tis_auc (+CI) per cohort

    displayed = {"Hugo_2016": "Hugo (pembrolizumab, n = 27)",
                 "Riaz_2017": "Riaz (nivolumab, n = 49)"}
    models = ["PIES-DSS", "PIES (hand)", "TIS"]
    colors = [C_DSS, C_HAND, C_TIS]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, cohort in zip(axes, ["Hugo_2016", "Riaz_2017"]):
        vals, los, his, labels = [], [], [], []
        # PIES-DSS
        r = held[held.cohort == cohort].iloc[0]
        vals.append(r.auc); los.append(r.ci_lo); his.append(r.ci_hi)
        labels.append(r.perm_p_two_sided)
        # hand-PIES (bootstrap CI from honest validation)
        h = honest[honest.cohort == cohort].iloc[0]
        vals.append(h.observed_auc); los.append(h.ci_lo); his.append(h.ci_hi)
        labels.append(np.nan)
        # TIS (point + CI when available; hk-normalized faithful baseline)
        t = tis[tis.cohort_name == cohort].iloc[0]
        vals.append(t.tis_auc); los.append(t.tis_ci_lo if "tis_ci_lo" in tis.columns else np.nan)
        his.append(t.tis_ci_hi if "tis_ci_hi" in tis.columns else np.nan); labels.append(np.nan)

        x = np.arange(3)
        ax.bar(x, vals, width=0.55, color=colors, edgecolor="none")
        for xi in x:
            if not np.isnan(los[xi]) and not np.isnan(his[xi]):
                ax.errorbar(xi, vals[xi], yerr=[[vals[xi] - los[xi]],
                                                [his[xi] - vals[xi]]],
                            fmt="none", ecolor="black", lw=0.9, capsize=3)
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        for xi, (v, pv, hi) in enumerate(zip(vals, labels, his)):
            top = max(v, hi if not np.isnan(hi) else 0)
            txt = f"{v:.3f}" + (f"\np = {pv:.3f}" if not np.isnan(pv) else "")
            ax.text(xi, top + 0.05, txt, ha="center", fontsize=7)
        ax.set_xticks(x, models, fontsize=7, rotation=0)
        ax.set_title(displayed[cohort], fontsize=8)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("AUC for non-response (95% CI)")
    fig.suptitle("Held-out melanoma validation — frozen model, frozen panel",
                 y=1.08, fontsize=9)
    save_both(fig, project_root() / "figures" / "Figure4_heldout")


# ── Supplementary tables ─────────────────────────────────────────────────────
def supp_tables(cfg) -> None:
    out = project_root() / "manuscript" / "supplementary"
    out.mkdir(parents=True, exist_ok=True)

    # S1 — per-cohort OPPTI call counts and permutation-pass rates
    s1 = (results("aim1_pan_cancer_oppti_summary.csv")
          .groupby("cancer_type")
          .agg(proteins=("protein", "nunique"),
               outliers_q95=("n_outlier_cases", "sum"),
               outliers_perm=("n_outlier_fdr", "sum"),
               samples=("n_total", "first"))
          .assign(perm_pass_rate=lambda d: (d.outliers_perm / d.outliers_q95))
          .rename(columns={"outliers_q95": "q95_calls",
                           "outliers_perm": "perm_sig_calls",
                           "perm_pass_rate": "perm_pass_fraction"}))
    s1.loc["TOTAL"] = {
        "proteins": 137,
        "q95_calls": s1["q95_calls"].sum(),
        "perm_sig_calls": s1["perm_sig_calls"].sum(),
        "samples": s1["samples"].sum(),
        "perm_pass_fraction": s1["perm_sig_calls"].sum() / s1["q95_calls"].sum(),
    }
    s1.round(4).to_csv(out / "S1_oppti_calls_by_cohort.csv")

    # S2 — per-gene × cohort post-transcriptional fractions (Q95-restricted)
    frac = results("aim2_post_transcriptional_fraction.csv")
    cls_ = pd.read_parquet(project_root() / "results" /
                           "aim2_concordance_classification.parquet")
    cls_["cohort"] = cls_["sample"].astype(str).str.rsplit("_", n=1).str[-1]
    piv = (cls_.assign(hit=cls_["concordance"] == "post_transcriptional")
           .groupby(["gene", "cohort"])["hit"].agg(["sum", "count"]))
    wide = (piv["sum"].unstack() / piv["count"].unstack() * 100).round(1)
    wide.columns = [f"pt_frac_pct_{CANON.get(c, c)}" for c in wide.columns]
    n_all = piv["sum"].unstack().sum(axis=1).astype(int)
    pt_all = (frac.set_index("gene")
              .rename(columns={"post_transcriptional": "pt_all",
                               "n_outlier_total": "n_calls_Q95",
                               "pct_post_transcriptional":
                                   "pt_frac_pct_pooled"}))
    wide = wide.join(pt_all.reindex(wide.index))
    wide["n_calls_Q95"] = n_all
    wide = wide.drop(columns=["pt_all"])
    wide.to_csv(out / "S2_post_transcriptional_fractions.csv")

    # S3 — CNV-protein correlations (655 tested; CNV-driven flagged by BH-FDR)
    cnv = results("aim2_cnv_mechanism.csv")
    cnv_q = cnv.sort_values("pvalue").reset_index(drop=True)
    m = len(cnv_q)
    q = np.minimum.accumulate(
        (cnv_q["pvalue"].values * m / np.arange(1, m + 1))[::-1])[::-1]
    cnv_q["q_bh"] = np.clip(q, 0, 1)
    cnv_q["cnv_driven_bh"] = cnv_q["q_bh"] < 0.05
    cnv_q = cnv_q.sort_values(["cnv_driven_bh", "pvalue"],
                              ascending=[False, True])
    cnv_q.to_csv(out / "S3_cnv_protein_correlations.csv", index=False)

    # S4 — nested-CV outer-fold detail
    nc = results("aim4_pies_dss_nested_cv.csv")
    weights = results("aim4_pies_dss_weights.csv")
    w_long = weights.melt(id_vars="module", value_vars=[
        "coef_median", "coef_mean", "coef_min", "coef_max"],
        var_name="statistic", value_name="value")
    s4 = (nc.assign(source="outer-fold summary")
          .rename(columns={"outer_fold": "outer_fold",
                           "C_selected": "C_inner_selected",
                           "auc": "outer_test_auc",
                           "n_test": "n_test",
                           "n_responders_test": "n_responders_test"}))
    s4.to_csv(out / "S4_nested_cv_folds.csv", index=False)
    w_long.to_csv(out / "S4b_module_weight_range.csv", index=False)

    # S5 — hand-PIES honest validation (bootstrap/CV/permutation/DeLong/meta)
    hv = results("aim4_honest_validation.csv")
    meta = results("aim4_meta_auc.csv")
    comp = results("aim4_pies_dss_comparison.csv")
    delong = comp[comp.model.str.contains("DeLong")][
        ["model", "cohort", "perm_p_two_sided"]].rename(
        columns={"perm_p_two_sided": "delong_p"})
    hv.to_csv(out / "S5_honest_validation_handPIES.csv", index=False)
    with open(out / "S5_meta_and_delong.csv", "w") as f:
        pass
    meta.to_csv(out / "S5_meta_and_delong.csv", index=False)
    delong.to_csv(out / "S5_delong_paired.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fig1(cfg)
    fig2(cfg)
    fig3(cfg)
    fig4(cfg)
    supp_tables(cfg)
    print("figures/ + manuscript/supplementary/ written.")


if __name__ == "__main__":
    main()

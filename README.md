# OPPTI Immune Evasion Atlas

**Post-transcriptional immune evasion in CPTAC pan-cancer proteogenomics, and a
data-derived, cross-validated RNA-based classifier of immune-checkpoint-blockade
response (PIES-DSS).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![R 4.3+](https://img.shields.io/badge/R-4.3%2B-276DC3.svg)](https://www.r-project.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the reproducible analysis pipeline accompanying a study
of immune-evasion protein outliers and immunotherapy response prediction, built
on the [OPPTI](https://www.nature.com/articles/s42003-021-02636-7) (OverexPressed
Protein and Transcript target Identifier) pan-cancer proteomic outlier framework.

### The question

Immune checkpoint blockade (ICB) fails in ~60% of patients even when
transcriptomic biomarkers (TMB, MSI, tumour-inflammation signatures) look
favorable. Because immune evasion is frequently executed at the protein level,
we ask: **how much of the proteomic evasion signal is RNA-invisible, and can
weighting RNA events by that invisibility produce a better RNA-only response
classifier?** — with the question answered by a per-gene-correlation noise
null and a direct ablation, both included in this repository.

### What the pipeline does

| Step | Analysis | Output |
|------|----------|--------|
| 1–3 | OPPTI Q95 outlier calling on a 137-protein immune-evasion panel across 9 CPTAC cancer types (929 paired proteome–transcriptome tumours), with permutation annotation | Pan-cancer outlier atlas (5,488 Q95 calls) |
| 4 | RNA–protein concordance restricted to OPPTI calls; CNV–protein correlation analysis | 54.2% of protein outliers lack matched RNA elevation — indistinguishable from matched control proteins (55.7 ± 0.9%) and below the 71.1% correlation-implied expectation (see step 17); 47.1% of permutation-annotated calls |
| 5 | Tumour microenvironment deconvolution and signature scoring | Protein outlier ↔ immune-infiltration correlations |
| 6–7 | Protein-level Immune Evasion Score (hand-PIES) via RNA→protein imputation models; validation in 3 ICB cohorts | Scores + cohort AUCs |
| 11 | Full hand-PIES validation: bootstrap CIs, cross-validated AUC, permutation p, paired DeLong, random-effects meta-analysis | Full validation tables |
| 14 | **PIES-DSS**: module-level RNA score whose per-gene prior (pt_frac) is post-transcriptional invisibility and whose 8 module weights are learned by L2 logistic regression inside 5×5 nested CV (IMvigor210, n = 298); frozen-model transfer to Hugo 2016 and Riaz 2017 | Learned weights + paired OOF comparison + held-out AUCs |
| 15 | Publication figures and supplementary tables | `figures/`, `supplementary/` |
| 16–22 | Ablation (uniform / module-shuffled pt_frac) with feature diagnostics; per-gene-correlation discordance expectation with matched control gene sets (calibration check); 20-repeat nested CV + 200-permutation full-refit null; bootstrap CIs on learned module coefficients + module-feature correlation matrix; per-sample score export; incremental value vs TMB / PD-L1 IC / TCGA molecular subtype + continuous-feature baseline + OS (Cox, tertile log-rank); negative controls (burden, library size, TMB), 500 matched random panels, signed/unsigned event variants (unclipped + winsorized), sizeFactor normalization, prior-leverage ridge, TMB-conditional learning, phenotype/subtype stratification with CIs, nested Platt recalibration; round-4 checks — gene-by-gene matched CPTAC discordance controls (protein-abundance/variance/correlation deciles), both-direction burden tests, Brier skill scores, 300 immune-annotated matched panels, 200 module-label shuffles, remaining paired Table-6 deltas, and a confirmatory-cohort power simulation; v2 evaluation — selection-honest max-statistic null (200 full-refit permutations, all declared variants inside each), 20-repeat nested CV for the v2 candidates (seeds 1–20, mirroring script 18), and paired repeat analysis (19/20 repeats above shipped) | Head-to-head evidence (`supplementary/S6–S8`) |

Key results (reproducible from this pipeline): **the proteomic RNA-invisibility
prior does not help an RNA-only ICB score** — uniform gene weights match or
exceed the prioritized model (pooled OOF AUC 0.637 vs 0.616), its weighted
features are ≥ 0.99 correlated with uniform ones, and pt_frac itself proxies
per-gene protein–RNA correlation (r = −0.51). The observed 54.2% discordance
matches matched non-immune control gene sets
(55.7 ± 0.9%, z = −1.77; under gene-by-gene matching on protein-abundance,
protein-variance, and protein–RNA-correlation deciles with immune genes
excluded, 50.5% vs matched controls 50.4 ± 0.6%, z = +0.18) and below the
71.1% *correlation-implied expectation*
— which itself over-predicts invisibility for control proteins (75.9% expected
vs 55.7% observed) and is therefore an upper bound, not a calibrated noise
benchmark. PIES-DSS itself: pooled out-of-fold AUC **0.616 [0.539–0.686],
permutation p = 0.008**, robust across 20 repeated nested-CV re-runs (14/20
repeats above all 200 full-refit null draws, p ≤ 0.005); versus 0.597
hand-set weights and 0.560 hk-normalized TIS (paired DeLong p = 0.61 / 0.34,
n.s.). **Incremental value over standard covariates is nil in the development
cohort**: TMB alone 0.728; TMB + PD-L1 IC 0.728; adding PIES-DSS 0.736
(+0.008 [−0.007, +0.023]); TMB-conditional learning is no better (−0.027
[−0.079, +0.028]; unpenalized spec: TMB 0.720 → +logit(DSS) 0.726,
Δ +0.006 [−0.010, +0.021]). **Negative controls**: the score shares signal
with genome-wide aberration burden (ρ = −0.29) but is not reducible to it
(burden-adjusted Δ +0.002 [−0.038, +0.040], LRT p = 0.20; reverse
direction, burden alone → burden+score Δ +0.024 [−0.034, +0.085] — neither
adds much to the other; library size ρ = −0.09; burden-normalized events
0.537); the IEP panel scored with the identical fixed-C ridge sits above all
500 matched random panels (0.642 vs 0.534 ± 0.043, max 0.640), all 300
immune-annotated matched panels (0.520 ± 0.052, max 0.639) and all 200
module-label shuffles of the same 137 genes (0.532 ± 0.046, max 0.627) —
with the caveat that the panel's curation draws on the same literature base
as the discordance-benchmark literature. **Variants**: signed events
0.675 and sizeFactor normalization 0.652 hold the highest point estimates
(paired CIs include zero; with ~14 variants some overshoot is expected) and
are declared a pre-specified v2, to be evaluated with a selection-honest
permutation null (all variants inside each permutation, maximum taken) and
confirmatory cohorts — not declared confirmed from this cohort; power
simulation (one-sided AUC > 0.5 test, α = 0.05, 30% responders): N =
100/150/200/300 gives power 0.61/0.76/0.86/0.95 for a true AUC of 0.62; winsorized continuous
features 0.626/0.625 (the unclipped 0.500/0.504 are heavy-tail artifacts);
OOF calibration slope 0.44 → 0.60 after nested Platt recalibration;
Brier skill over the prevalence baseline is −0.02 raw and only +0.002 after
recalibration — the score discriminates (AUC) but its probabilities carry no
skill and should not be thresholded without recalibration on a larger
cohort.
Held-out melanoma AUCs 0.706 (Hugo) / 0.541 (Riaz) — small cohorts,
inconclusive. Recurrent proteomic outlier proteins: ACSS2, HDAC6, CD274
(PD-L1), CANX.

## Repository structure

```
oppti-evasion-atlas/
├── config/
│   └── config.yaml              # All pipeline parameters (edit here)
├── data/
│   ├── gene_panels/             # IEP panel + provenance (tracked)
│   ├── cptac/                   # Downloaded by script 01 (not tracked)
│   ├── immunotherapy/           # Downloaded by scripts 01/02b (not tracked)
│   └── processed/               # Built by script 02 (not tracked)
├── workflow/
│   └── Snakefile                # End-to-end Snakemake orchestration
├── R/                           # OPPTI R implementation + package installer
├── src/oppti_immune/            # Installable Python package (core logic)
├── scripts/                     # Step runner scripts (see table above)
├── tests/                       # Unit tests (pies, preprocessing)
├── figures/                     # Generated figures (PDF + PNG, 300 dpi)
├── supplementary/               # Supplementary tables S1–S5 (CSV)
├── scripts/dryrun_sanity_check.py  # 19-assertion end-to-end sanity check
├── environment.yml              # Conda environment (Python + R)
├── requirements.txt             # Pip requirements
└── pyproject.toml               # Package metadata
```

## Quick start

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate oppti

# 2. (Optional) install the package locally
pip install -e .

# 3. Run the full pipeline
snakemake --cores 8 --use-conda

# Or run steps individually, in order:
python scripts/01_acquire_data.py        # CPTAC + ICB cohort download
python scripts/02_preprocess.py          # normalisation, QC, panel subsetting
python scripts/02b_prepare_icb_geo_cohorts.py   # Hugo/Riaz cohort matrices from GEO supplements
Rscript scripts/03_run_oppti.R           # OPPTI outlier calls (R implementation)
python scripts/04_concordance_analysis.py
Rscript scripts/05_tme_deconvolution.R && python scripts/05_tme_scoring.py
python scripts/06_build_pies.py
python scripts/07_validate_icb_cohorts.py
python scripts/11_honest_pies_validation.py
python scripts/14_pies_dss.py
python scripts/15_figures.py             # figures/ + supplementary/

# Sanity check (19 assertions; requires acquired + processed data)
python scripts/dryrun_sanity_check.py
```

Python users can also use the ported OPPTI implementation
(`scripts/03_run_oppti_py.py`; its agreement with the R implementation is
benchmark-tested in `scripts/13_oppti_benchmark.py`), and additional robustness
sweeps (recurrence-threshold and module-weight schemes) are available in
`scripts/10_sensitivity_analysis.py`.

## Data sources (all public)

| Dataset | Source | Access |
|---------|--------|--------|
| CPTAC proteomics + transcriptomics (9 cancer types) | NCI Clinical Proteomic Tumor Analysis Consortium | `cptac` Python package (`pip install cptac`) |
| Hugo 2016 (pembrolizumab, melanoma) | GEO GSE78220 | Supplements (PatientFPKM.xlsx) |
| Riaz 2017 (nivolumab, melanoma) | GEO GSE91061 | Supplements (BMS038109Sample.hg19KnownGene.fpkm.csv.gz) |
| IMvigor210 (atezolizumab, urothelial) | Genentech | Public research release (IMvigor210CoreBiologies package) |

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this pipeline, please cite the OPPTI framework paper
([Elmas et al., *Communications Biology*, 2021](https://www.nature.com/articles/s42003-021-02636-7))
and this repository (see [CITATION.cff](CITATION.cff)).

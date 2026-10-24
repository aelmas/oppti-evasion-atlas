# Immune Evasion Proteome (IEP) panel — provenance & versioning

**File:** `immune_evasion_proteome.csv`
**Version:** v1.0
**Genes:** 164 across 8 functional modules.

## Curation statement

This panel is **expert-curated**, not derived by an automated, reproducible
selection from a single database. Each gene was assigned to one of eight
functional modules on the basis of the published immune-evasion literature and
public pathway resources. The `source_category` column records the primary
sourcing basis **per module** (below); the `rationale` column gives a free-text
justification per gene. We document this curation explicitly because a
hand-built panel is a limitation: the universe of tested proteins and the PIES
module structure both depend on it.

## Module composition and source basis

| Module | n | Source category |
|--------|---|-----------------|
| checkpoint | 28 | Immune-checkpoint literature + Open Targets immuno-oncology targets |
| antigen_presentation | 27 | Reactome MHC-I antigen processing/presentation; HLA + APM genes |
| ecm_exclusion | 27 | TGFβ/ECM T-cell-exclusion programs (pan-F-TBRS; Mariathasan 2018) |
| cytokines | 26 | Immunosuppressive cytokine/receptor literature (TGFB, IL10, CSF1 axes) |
| epigenetic | 19 | Chromatin-modifier / tumor-intrinsic immune-evasion literature |
| metabolic | 17 | Immunometabolism literature (T-cell-suppressive enzymes) |
| ifn_pathway | 13 | MSigDB Hallmark IFN-γ response; Reactome IFN signaling |
| mismatch_repair | 7 | Core MMR genes (MSI / neoantigen biology) |

## Directionality convention (relevant to PIES module weights)

OPPTI detects **overexpression** outliers only. PIES module weights therefore
encode the consequence of a gene being **high**: positive weight → high protein
= more predicted evasion; negative weight → high protein = less evasion
(protective/inflamed). Loss-of-function evasion (HLA loss, IFN-pathway loss, MMR
loss) is **not** directly detected and appears only as the absence of a
protective overexpression outlier. See `config/config.yaml` `module_weights`.

## Known limitations

1. Module assignment is single-annotator; no inter-annotator agreement was measured.
2. Several genes are pleiotropic (e.g., SMAD3, STAT1) and could defensibly sit in
   more than one module; the assignment reflects the dominant immune-evasion role.
3. The `mismatch_repair` sign in PIES is debatable (MMR loss → MSI-high → better
   ICB response), tested in `results/sensitivity_mmr_sign.csv`.
4. A fully reproducible alternative — intersecting MSigDB/Reactome gene sets
   programmatically — is recommended for a v2.0 panel and would remove the
   single-annotator dependency.

## Reproducing downstream results with a different panel

All pipeline steps read the panel via `utils.load_gene_panel()`, which keys only
on the `gene` and `module` columns. A drop-in replacement panel with those two
columns will propagate through OPPTI, concordance, PIES, and the figures without
code changes.

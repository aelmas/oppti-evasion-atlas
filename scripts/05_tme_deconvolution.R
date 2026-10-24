#!/usr/bin/env Rscript
# ════════════════════════════════════════════════════════════════════════════
# Step 5: Tumor Microenvironment Deconvolution
# ════════════════════════════════════════════════════════════════════════════
# Uses EPIC + xCell to estimate immune cell fractions from bulk RNA,
# then correlates each OPPTI outlier protein with cell-type infiltration.
#
# Outputs:
#   results/aim3_tme_epic_fractions.csv
#   results/aim3_tme_xcell_scores.csv
#   results/aim3_protein_immune_correlation.csv
# ════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(tidyverse)
  library(yaml)
  library(arrow)
  library(EPIC)
  library(xCell)
})

args   <- commandArgs(trailingOnly = TRUE)
cfg_path <- if (length(args) >= 1) args[[1]] else "config/config.yaml"
cfg    <- read_yaml(cfg_path)

# ── Load pan-cancer full RNA matrix ─────────────────────────────────────────
rna_path <- file.path(cfg$paths$data_root, "processed", "pan_rna_full.parquet")
if (!file.exists(rna_path)) stop("Run scripts/02_preprocess.py first.")

rna_df <- read_parquet(rna_path)
rna_mat <- rna_df %>%
  { rownames(.) <- rownames(rna_df); . } %>%
  as.data.frame() %>%
  t()    # genes × samples for deconvolution

cat("Pan-cancer RNA matrix:", nrow(rna_mat), "genes ×", ncol(rna_mat), "samples\n")

# ── EPIC ────────────────────────────────────────────────────────────────────
cat("Running EPIC deconvolution...\n")
epic_res <- tryCatch(
  EPIC(bulk = rna_mat, reference = TRef, withOtherCells = TRUE),
  error = function(e) { cat("EPIC failed:", conditionMessage(e), "\n"); NULL }
)

if (!is.null(epic_res)) {
  epic_df <- as.data.frame(epic_res$cellFractions) %>%
    rownames_to_column("sample")
  write.csv(epic_df,
            file.path(cfg$paths$results_root, "aim3_tme_epic_fractions.csv"),
            row.names = FALSE)
}

# ── xCell (64 cell types) ───────────────────────────────────────────────────
cat("Running xCell enrichment...\n")
xcell_res <- tryCatch(
  xCellAnalysis(as.matrix(rna_mat)),
  error = function(e) { cat("xCell failed:", conditionMessage(e), "\n"); NULL }
)

if (!is.null(xcell_res)) {
  xcell_df <- as.data.frame(t(xcell_res)) %>%
    rownames_to_column("sample")
  write.csv(xcell_df,
            file.path(cfg$paths$results_root, "aim3_tme_xcell_scores.csv"),
            row.names = FALSE)
}

# ── Correlate OPPTI outliers with immune cell types ─────────────────────────
outlier_path <- file.path(cfg$paths$results_root, "oppti_binary_matrix.parquet")
if (!file.exists(outlier_path) || is.null(xcell_res)) {
  cat("Outlier or xCell results missing; skipping correlation step.\n")
  quit(save = "no")
}

outlier_df <- read_parquet(outlier_path)
if ("sample" %in% colnames(outlier_df)) {
  outlier_df <- outlier_df %>% column_to_rownames("sample")
}

metrics <- cfg$tme$cell_types_of_interest
iep    <- read.csv(cfg$paths$gene_panel, stringsAsFactors = FALSE)
genes  <- iep$gene

shared  <- intersect(rownames(outlier_df), xcell_df$sample)
cat("Correlating across", length(shared), "samples,",
    length(intersect(genes, colnames(outlier_df))), "genes,",
    length(metrics), "cell types...\n")

rows <- list()
for (g in intersect(genes, colnames(outlier_df))) {
  for (m in intersect(metrics, colnames(xcell_df))) {
    ov   <- outlier_df[shared, g]
    xv   <- xcell_df[match(shared, xcell_df$sample), m]
    ok   <- !is.na(ov) & !is.na(xv)
    if (sum(ok) < 20) next
    ct   <- suppressWarnings(cor.test(ov[ok], xv[ok], method = "spearman"))
    rows[[paste(g, m, sep = "__")]] <- data.frame(
      gene = g, immune_metric = m, rho = ct$estimate, pvalue = ct$p.value
    )
  }
}

corr_df <- bind_rows(rows) %>%
  mutate(padj = p.adjust(pvalue, method = "BH")) %>%
  arrange(pvalue)

write.csv(corr_df,
          file.path(cfg$paths$results_root, "aim3_protein_immune_correlation.csv"),
          row.names = FALSE)

cat("Step 5 complete. Top 10 correlations:\n")
print(head(corr_df, 10))

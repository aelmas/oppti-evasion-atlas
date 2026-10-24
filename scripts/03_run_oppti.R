#!/usr/bin/env Rscript
# ════════════════════════════════════════════════════════════════════════════
# Step 3: OPPTI Analysis on the Immune Evasion Proteome
# ════════════════════════════════════════════════════════════════════════════
# For each cancer type, runs OPPTI on the IEP subset to identify
# protein-level outliers, then aggregates results pan-cancer.
#
# Outputs:
#   results/oppti/<cancer>_scores.csv       -- continuous outlier scores
#   results/oppti/<cancer>_outlier_binary.csv
#   results/oppti_binary_matrix.parquet     -- pan-cancer binary matrix
#   results/aim1_pan_cancer_oppti_summary.csv
#   results/aim1_recurrent_IEP_outliers.csv
# ════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(oppti)
  library(tidyverse)
  library(yaml)
  library(arrow)
})

# ── Load config ─────────────────────────────────────────────────────────────
args   <- commandArgs(trailingOnly = TRUE)
cfg_path <- if (length(args) >= 1) args[[1]] else "config/config.yaml"
# Force UTF-8 reading so unicode characters in comments don't break parser
Sys.setlocale("LC_ALL", "en_US.UTF-8")
cfg <- read_yaml(cfg_path, fileEncoding = "UTF-8")

dir.create(file.path(cfg$paths$results_root, "oppti"), showWarnings = FALSE, recursive = TRUE)

# ── Load pan-cancer IEP proteomics ──────────────────────────────────────────
pan_prot_path <- file.path(cfg$paths$data_root, "processed", "pan_prot_iep.parquet")
if (!file.exists(pan_prot_path)) stop("Run scripts/02_preprocess.py first.")

pan_prot <- read_parquet(pan_prot_path)
cancer_labels <- as.character(pan_prot$cancer_type)
# Parquet doesn't preserve rownames → synthesize stable sample IDs
sample_ids <- paste0("S", seq_len(nrow(pan_prot)), "_", cancer_labels)

prot_mat <- data.matrix(pan_prot[, setdiff(colnames(pan_prot), "cancer_type")])
rownames(prot_mat) <- sample_ids

cat("Loaded pan-cancer IEP matrix:", nrow(prot_mat), "samples ×",
    ncol(prot_mat), "proteins\n")
cat("Numeric? ", is.numeric(prot_mat), "  NA frac:",
    round(mean(is.na(prot_mat)), 3), "\n\n")

# ── Per-cancer OPPTI runs ───────────────────────────────────────────────────
oppti_results <- list()
n_perm <- cfg$oppti$n_permutations
k_val  <- cfg$oppti$k_neighbors
pct    <- cfg$oppti$outlier_percentile / 100

for (ct in unique(cancer_labels)) {
  cat("▶ OPPTI on", ct, "...\n")

  idx     <- which(cancer_labels == ct)
  sub_mat <- t(prot_mat[idx, , drop = FALSE])    # OPPTI expects proteins × samples
  storage.mode(sub_mat) <- "double"              # ensure numeric

  # Drop proteins with > miss_thresh missing
  keep <- rowMeans(is.na(sub_mat)) < 0.4
  sub_mat <- sub_mat[keep, , drop = FALSE]
  if (nrow(sub_mat) < 10 || ncol(sub_mat) < 10) {
    cat("  skip (insufficient data)\n"); next
  }

  # OPPTI API — try current API signature, fall back to legacy
  res <- tryCatch(
    oppti(list(sub_mat),
          cohort.names       = ct,
          panel              = "global",
          panel.markers      = list(rownames(sub_mat)),
          mad.norm           = FALSE,   # already MAD-normalized upstream
          ku                 = k_val,
          tol.nas            = 40,
          permutation.tests  = (n_perm > 0),
          n.per.test         = n_perm,
          verbose            = FALSE),
    error = function(e) {
      cat("  oppti() error:", conditionMessage(e), "\n")
      # Legacy API fallback (older oppti versions)
      tryCatch(
        oppti(list(sub_mat), cohort.names = ct, mad.norm = FALSE,
              k = k_val, n.permutations = n_perm, verbose = FALSE),
        error = function(e2) { cat("  legacy fallback error:", conditionMessage(e2), "\n"); NULL }
      )
    }
  )
  if (is.null(res)) next

  # res is a list; first element contains the outlier score matrix/list
  scores_list <- res[[1]]
  scores      <- as.data.frame(if (is.list(scores_list)) scores_list[[1]] else scores_list)

  # Validate — if all NA, skip (R oppti API mismatch with input format)
  if (all(is.na(unlist(scores)))) {
    cat("  skip: all NA scores (R oppti API incompatibility — use 03_run_oppti_py.py)\n"); next
  }
  thresh      <- quantile(unlist(scores), pct, na.rm = TRUE)
  binary_mat  <- as.data.frame(scores > thresh & !is.na(scores))

  write.csv(scores,     file.path(cfg$paths$results_root, "oppti",
                                  paste0(ct, "_scores.csv")),
            row.names = TRUE)
  write.csv(binary_mat, file.path(cfg$paths$results_root, "oppti",
                                  paste0(ct, "_outlier_binary.csv")),
            row.names = TRUE)

  oppti_results[[ct]] <- list(
    scores  = scores, binary = binary_mat,
    summary = data.frame(
      protein         = rownames(scores),
      n_outlier_cases = rowSums(binary_mat, na.rm = TRUE),
      pct_outlier     = rowMeans(binary_mat, na.rm = TRUE) * 100,
      cancer_type     = ct
    )
  )
  cat("  ", nrow(scores), "proteins,",
      sum(binary_mat, na.rm = TRUE), "outlier calls\n")
}

# ── Consolidate pan-cancer binary matrix ────────────────────────────────────
bin_list <- lapply(names(oppti_results), function(ct) {
  bin <- oppti_results[[ct]]$binary
  t_bin <- as.data.frame(t(bin))   # samples × proteins
  t_bin$sample <- rownames(t_bin)
  t_bin
})
bin_wide <- bind_rows(bin_list) %>% column_to_rownames("sample")
bin_wide[is.na(bin_wide)] <- 0

arrow::write_parquet(
  tibble::rownames_to_column(bin_wide, "sample"),
  file.path(cfg$paths$results_root, "oppti_binary_matrix.parquet")
)

# ── Pan-cancer summary ──────────────────────────────────────────────────────
pan_summary <- bind_rows(lapply(oppti_results, `[[`, "summary"))
write.csv(pan_summary,
          file.path(cfg$paths$results_root, "aim1_pan_cancer_oppti_summary.csv"),
          row.names = FALSE)

recurrent <- pan_summary %>%
  filter(pct_outlier >= cfg$pies$min_outlier_fraction * 100) %>%
  group_by(protein) %>%
  summarise(
    n_cancer_types = n_distinct(cancer_type),
    mean_pct       = mean(pct_outlier),
    cancers        = paste(sort(cancer_type), collapse = ",")
  ) %>%
  filter(n_cancer_types >= cfg$pies$min_cancer_types) %>%
  arrange(desc(n_cancer_types), desc(mean_pct))

write.csv(recurrent,
          file.path(cfg$paths$results_root, "aim1_recurrent_IEP_outliers.csv"),
          row.names = FALSE)

cat("\n══════════════════════════════════════════════════════════════\n")
cat(" OPPTI complete.", nrow(recurrent), "recurrent IEP outlier proteins.\n")
cat("══════════════════════════════════════════════════════════════\n")
print(head(recurrent, 20))

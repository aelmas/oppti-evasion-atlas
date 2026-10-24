# ════════════════════════════════════════════════════════════════════════════
# oppti_runner.R — Reusable OPPTI wrapper functions.
# Sourced by scripts/03_run_oppti.R and by workflow/Snakefile.
# ════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(oppti)
  library(tidyverse)
})

#' Run OPPTI on a proteomics sub-matrix for a single cancer type.
#'
#' @param prot_mat  samples × proteins matrix (rows = samples).
#' @param n_perm    number of permutations for significance.
#' @param k         k for k-nearest co-expressed neighbors.
#' @param percentile  quantile cutoff for outlier calling (0–1).
#' @param min_expressed_frac  require ≥ this fraction of samples to express each protein.
#'
#' @return list with: scores (protein × sample), binary (same shape, 0/1)
run_oppti_one <- function(prot_mat, n_perm = 1000, k = 5,
                          percentile = 0.95, min_expressed_frac = 0.6) {
  if (nrow(prot_mat) < 10 || ncol(prot_mat) < 10) {
    return(NULL)
  }
  mat_pxs <- t(prot_mat)                       # proteins × samples
  keep    <- rowMeans(!is.na(mat_pxs)) >= min_expressed_frac
  mat_pxs <- mat_pxs[keep, , drop = FALSE]
  if (nrow(mat_pxs) < 10) return(NULL)

  res <- tryCatch(
    oppti(mat_pxs, n.permutations = n_perm, k = k),
    error = function(e) { message("OPPTI error: ", conditionMessage(e)); NULL }
  )
  if (is.null(res)) return(NULL)

  scores <- as.data.frame(res[[1]])
  thr    <- quantile(unlist(scores), percentile, na.rm = TRUE)
  binary <- as.data.frame(scores > thr & !is.na(scores))

  list(scores = scores, binary = binary, threshold = thr)
}

#' Summarize OPPTI outlier calls per protein for one cancer type.
#' @param binary  protein × sample binary DataFrame
#' @param cancer  cancer-type label (added as a column)
summarize_outliers <- function(binary, cancer) {
  data.frame(
    protein         = rownames(binary),
    n_outlier_cases = rowSums(binary, na.rm = TRUE),
    pct_outlier     = rowMeans(binary, na.rm = TRUE) * 100,
    cancer_type     = cancer,
    stringsAsFactors = FALSE
  )
}

#' Assemble a pan-cancer sample × protein binary outlier matrix.
#' @param per_cancer  named list of per-cancer binary matrices (protein × sample)
pan_cancer_binary <- function(per_cancer) {
  frames <- lapply(per_cancer, function(b) {
    d <- as.data.frame(t(b))
    d$sample <- rownames(d)
    d
  })
  wide <- dplyr::bind_rows(frames) %>%
    tibble::column_to_rownames("sample")
  wide[is.na(wide)] <- 0
  wide
}

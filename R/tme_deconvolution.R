# ════════════════════════════════════════════════════════════════════════════
# tme_deconvolution.R — Reusable EPIC + xCell wrappers.
# ════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(EPIC)
  library(xCell)
})

#' Run EPIC on a bulk RNA matrix.
#'
#' @param rna_mat  genes × samples TPM matrix
#' @param reference  EPIC reference set; default TRef (tumor-adapted)
#' @return data frame (samples × cell-type fractions) with `sample` column
run_epic <- function(rna_mat, reference = TRef) {
  res <- EPIC(bulk = rna_mat, reference = reference, withOtherCells = TRUE)
  df  <- as.data.frame(res$cellFractions)
  df$sample <- rownames(df)
  df
}

#' Run xCell on a bulk RNA matrix.
#' @param rna_mat  genes × samples expression (TPM or log2 TPM)
#' @return data frame (samples × cell-type scores) with `sample` column
run_xcell <- function(rna_mat) {
  res <- xCellAnalysis(as.matrix(rna_mat))
  df  <- as.data.frame(t(res))
  df$sample <- rownames(df)
  df
}

#' Classify tumors into immune phenotypes: inflamed / excluded / desert.
#' @param df  xCell data frame with at minimum CD8+ T-cells, CAF, TGF-β1
classify_immune_phenotype <- function(df) {
  need <- c("CD8+ T-cells", "Cancer associated fibroblasts", "TGF-b1")
  if (!all(need %in% colnames(df))) {
    message("Required columns missing; returning NA phenotypes.")
    df$immune_phenotype <- NA_character_
    return(df)
  }
  cd8_med  <- median(df[["CD8+ T-cells"]], na.rm = TRUE)
  caf_med  <- median(df[["Cancer associated fibroblasts"]], na.rm = TRUE)
  tgfb_med <- median(df[["TGF-b1"]], na.rm = TRUE)

  df$immune_phenotype <- dplyr::case_when(
    df[["CD8+ T-cells"]] >  cd8_med &
      df[["Cancer associated fibroblasts"]] < caf_med ~ "inflamed",
    df[["CD8+ T-cells"]] <= cd8_med &
      df[["TGF-b1"]]     >  tgfb_med                  ~ "excluded",
    TRUE                                              ~ "desert"
  )
  df
}

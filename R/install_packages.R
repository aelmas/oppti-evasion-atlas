#!/usr/bin/env Rscript
# ════════════════════════════════════════════════════════════════════════════
# Install all R dependencies for the OPPTI Immune Evasion pipeline.
# Run once after creating the conda environment:
#   Rscript R/install_packages.R
# ════════════════════════════════════════════════════════════════════════════

options(repos = c(CRAN = "https://cloud.r-project.org"))

cran_pkgs <- c(
  "BiocManager", "devtools", "tidyverse", "yaml", "arrow",
  "survival", "survminer", "ggplot2", "pheatmap",
  "circlize", "Matrix"
)

to_install <- setdiff(cran_pkgs, rownames(installed.packages()))
if (length(to_install) > 0) {
  cat("Installing CRAN packages:", paste(to_install, collapse = ", "), "\n")
  install.packages(to_install, Ncpus = max(1, parallel::detectCores() - 1))
}

# Bioconductor packages
bioc_pkgs <- c("oppti", "limma", "DESeq2", "GSVA", "piano",
               "sva", "edgeR", "fgsea", "ComplexHeatmap",
               "GEOquery", "Biobase", "IMvigor210CoreBiologies")

library(BiocManager)
BiocManager::install(bioc_pkgs, update = FALSE, ask = FALSE, Ncpus = 4)

# GitHub packages
if (!requireNamespace("xCell", quietly = TRUE)) {
  devtools::install_github("dviraran/xCell", upgrade = "never")
}
if (!requireNamespace("EPIC", quietly = TRUE)) {
  devtools::install_github("GfellerLab/EPIC", upgrade = "never", build_vignettes = FALSE)
}

cat("\n══════════════════════════════════════════════════════════════\n")
cat(" All R dependencies installed.\n")
cat("══════════════════════════════════════════════════════════════\n")

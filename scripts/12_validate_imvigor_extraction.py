#!/usr/bin/env python
"""
Step 12: Validate IMvigor210 DESeq-stub extraction
==================================================

IMvigor210 counts were recovered from the Genentech `IMvigor210CoreBiologies`
DESeq `CountDataSet` (a legacy S4 object) via a minimal `DESeq` stub package
built only to deserialize it. This script confirms the extraction preserved
gene-sample correspondence, using checks that would fail if rows/columns were
scrambled:

  [1] Published cohort facts (Mariathasan et al. 2018 Nature):
      binaryResponse ≈ 68 CR/PR vs 230 SD/PD.
  [2] Housekeeping genes (ACTB, GAPDH, B2M, …) rank near the top of mean expression.
  [3] CD274 (PD-L1) increases monotonically across desert < excluded < inflamed
      immune phenotypes — a known IMvigor210 result.
  [4] CD8A tracks the same ordering.

A scrambled gene axis would break [2]; a scrambled sample axis would break [3]/[4].

Usage:  python scripts/12_validate_imvigor_extraction.py
"""

import numpy as np
import pandas as pd

from oppti_immune.utils import get_logger

logger = get_logger("12_validate_imvigor")

HOUSEKEEPING = ["ACTB", "GAPDH", "B2M", "EEF1A1", "TPT1"]


def main() -> None:
    expr = pd.read_csv("data/immunotherapy/IMvigor210_expr.csv", index_col=0)
    clin = pd.read_csv("data/immunotherapy/IMvigor210_clinical.csv", index_col=0)
    expr_log = np.log2(expr + 1)

    ok = True

    logger.info(f"Expr {expr.shape[0]} genes x {expr.shape[1]} samples; clinical {clin.shape[0]}")

    # [1] response split
    vc = clin["binaryResponse"].value_counts(dropna=False).to_dict()
    logger.info(f"[1] binaryResponse: {vc}")
    crpr = vc.get("CR/PR", 0)
    if not (60 <= crpr <= 75):
        logger.warning(f"    CR/PR count {crpr} outside published ~68 — check extraction")
        ok = False

    # [2] housekeeping percentile
    mean_expr = expr_log.mean(axis=1)
    for g in HOUSEKEEPING:
        if g in mean_expr.index:
            pct = float((mean_expr < mean_expr[g]).mean() * 100)
            logger.info(f"[2] {g}: {pct:.1f}th percentile")
            if pct < 85:
                logger.warning(f"    {g} unexpectedly low — possible gene-axis scramble")
                ok = False

    # [3]/[4] phenotype monotonicity
    ip_col = next((c for c in clin.columns if "phenotype" in c.lower()), None)
    if ip_col:
        order = ["desert", "excluded", "inflamed"]
        for gene in ("CD274", "CD8A"):
            if gene not in expr_log.index:
                continue
            vals = expr_log.loc[gene]
            means = {}
            for ph in order:
                samps = clin[ip_col][clin[ip_col] == ph].index.intersection(vals.index)
                means[ph] = float(vals.reindex(samps).mean())
            logger.info(f"[3/4] {gene} by phenotype: " +
                        ", ".join(f"{ph}={means[ph]:.2f}" for ph in order))
            if not (means["desert"] < means["excluded"] < means["inflamed"]):
                logger.warning(f"    {gene} not monotonic across phenotypes — possible sample-axis scramble")
                ok = False

    if ok:
        logger.info("✓ All IMvigor210 extraction validation checks passed.")
    else:
        logger.error("✗ One or more validation checks failed — review extraction.")
    return ok


if __name__ == "__main__":
    main()

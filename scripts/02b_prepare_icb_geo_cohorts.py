#!/usr/bin/env python
"""
Step 02b: prepare Hugo_2016 / Riaz_2017 cohort matrices from GEO supplements
==============================================================================
GEO series matrices ship empty expression matrices for these series; the real
data are in the supplementary files:

  * GSE78220  → PatientFPKM.xlsx   (25,268 genes × 28 samples, baseline + 1 OnTx)
  * GSE91061  → BMS038109Sample.hg19KnownGene.fpkm.csv.gz (Entrez-ID rows × 108)

Outputs (matching the schema the pipeline expects):
  data/immunotherapy/Hugo_2016_expr.csv            genes × samples
  data/immunotherapy/Hugo_2016_clinical.csv        pData subset (response + OS)
  data/immunotherapy/Riaz_2017_expr.csv            genes × samples
  data/immunotherapy/Riaz_2017_clinical.csv

Hugo sample labels (PtN.baseline / PtN.OnTx) map to pData titles (PtN); Riaz
labels (PtN_Pre_...) map to pData titles as well. Entrez IDs are translated to
gene symbols with mygene.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/immunotherapy")
XLSX = "/tmp/GSE78220_PatientFPKM.xlsx"
FPKM = "/tmp/GSE91061_fpkm.csv.gz"

# ── Hugo 2016 ────────────────────────────────────────────────────────────────

def prepare_hugo() -> None:
    expr_df = pd.read_excel(XLSX, sheet_name="FPKM")
    expr = expr_df.set_index("Gene")
    expr.index.name = None

    clin = pd.read_csv(DATA / "Hugo_2016_clinical.csv", index_col=0)
    clin["patient_id"] = clin["title"].astype(str).str.strip()

    # Column 'Pt16.OnTx' ↔ pData title 'Pt16'; keep the trailing biopsy tag.
    sample_to_pid, keep_cols = {}, []
    for col in expr.columns:
        pid = col.split(".")[0].strip()
        sample_to_pid[col] = pid
        keep_cols.append(col)
    missing = [c for c, p in sample_to_pid.items()
               if p not in set(clin["patient_id"])]
    if missing:
        raise ValueError(f"Hugo pData missing patient ids: {missing}")

    expr = expr[keep_cols]
    pids = [sample_to_pid[c] for c in expr.columns]
    clin = clin.set_index("patient_id").loc[pids]

    clin_out = pd.DataFrame({
        "response": clin["anti-pd-1 response:ch1"].values,
        "biopsy_time": clin["biopsy time:ch1"].values,
        "os_days": pd.to_numeric(clin["overall survival (days):ch1"].values,
                                 errors="coerce"),
        "os_event": (clin["vital status:ch1"].values == "Dead").astype(int),
    }, index=pids)
    clin_out.index = expr.columns   # now aligned with the sample columns
    clin_out["os"] = clin_out["os_days"] / 30.44  # → OS_months for config
    pre = clin_out["biopsy_time"].astype(str).str.strip() == "pre-treatment"
    clin_out = clin_out[pre]
    expr = expr.loc[:, clin_out.index]

    # rows = genes, columns = samples (matches GEO-R template schema).
    # Store log2(FPKM + 1): the downstream MAD-Z outlier calls assume
    # log-scale expression for these cohorts (script 11 log2-transforms
    # only IMvigor210, which sits on disk as raw counts).
    np.log2(expr + 1).T.to_csv(DATA / "Hugo_2016_expr.csv")
    clin_out.to_csv(DATA / "Hugo_2016_clinical.csv")
    responders = int(clin_out["response"].isin(
        ["Complete Response", "Partial Response"]).sum())
    print(f"Hugo: {expr.shape[0]} genes × {expr.shape[1]} pre-treatment samples "
          f"(n_responders={responders})")


# ── Riaz 2017 ────────────────────────────────────────────────────────────────

def prepare_riaz() -> None:
    import mygene

    mg = mygene.MyGeneInfo()
    raw = pd.read_csv(FPKM, index_col=0)
    entrez_ids = raw.index.astype(str).tolist()
    hits = mg.querymany(entrez_ids, scopes="entrezgene", fields="symbol",
                        species="human", verbose=False, returnall=False)
    symbol_map = {}
    for h in hits:
        if isinstance(h, dict) and "symbol" in h:
            symbol_map[str(h["query"])] = h["symbol"]

    raw.index = [symbol_map.get(str(e), f"ENTREZ_{e}") for e in raw.index]
    raw = raw[~raw.index.str.startswith("ENTREZ_")]
    raw = raw[~raw.index.duplicated(keep="first")]

    clin = pd.read_csv(DATA / "Riaz_2017_clinical.csv", index_col=0)
    clin["sample_id"] = clin["title"].astype(str).str.strip()

    # Columns like 'Pt1_Pre_AD101148-6' ↔ pData titles 'Pt1_Pre_AD101148-6'
    matched = [c for c in raw.columns if c in set(clin["sample_id"])]
    raw = raw[matched]
    clin = clin.set_index("sample_id").loc[matched]

    pre_mask = clin["visit (pre or on treatment):ch1"].astype(str).str.lower() == "pre"
    resp = clin["response:ch1"].astype(str)
    usable = pre_mask & resp.isin(["PD", "PR", "CR", "PRCR", "SD"])

    clin_out = clin[usable].copy()
    clin_out["response"] = clin_out["response:ch1"]
    clin_out["os"] = np.nan          # OS not provided in the series matrix
    clin_out["OS_months"] = np.nan
    clin_out["OS_event"] = np.nan

    raw = raw.loc[:, clin_out.index]
    # log2(FPKM + 1) — matching the original GEO-R download convention.
    np.log2(raw + 1).T.to_csv(DATA / "Riaz_2017_expr.csv")
    clin_out.to_csv(DATA / "Riaz_2017_clinical.csv")
    n_resp = int(clin_out["response"].isin(["PR", "CR", "PRCR"]).sum())
    print(f"Riaz: {raw.shape[0]} genes × {raw.shape[1]} pre-treatment samples "
          f"(n_responders={n_resp})")


if __name__ == "__main__":
    prepare_hugo()
    prepare_riaz()

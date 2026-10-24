"""Step 02c: re-download bcm transcriptomics for all 9 CPTAC cancer types (sample-matched)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import cptac
from oppti_immune.utils import load_config, tag_samples

cfg = load_config("config/config.yaml")
out = Path("data/cptac")

for ct in cfg["cptac"]["cancer_types"]:
    ds = getattr(cptac, ct.capitalize())()
    rna = ds.get_transcriptomics(source="bcm")
    import pandas as pd
    if isinstance(rna.columns, pd.MultiIndex):
        rna.columns = rna.columns.get_level_values(0)
    if rna.columns.has_duplicates:
        rna = rna.T.groupby(level=0).mean().T
    rna = tag_samples(rna, ct)
    rna.to_parquet(out / f"{ct}_transcriptomics.parquet")
    print(f"{ct}: bcm RNA ✓ ({rna.shape})")

print("RNA-REDOWNLOAD-DONE")

#!/usr/bin/env python
"""
Step 1: Data Acquisition
========================
Downloads all datasets required by the pipeline:
  - CPTAC proteomics (13 cancer types)
  - Immunotherapy validation cohorts (IMvigor210, Riaz, Hugo, Jung)

Usage:  python scripts/01_acquire_data.py [--config path/to/config.yaml]
"""

import argparse

from oppti_immune.acquisition import run_all
from oppti_immune.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_all(cfg)


if __name__ == "__main__":
    main()

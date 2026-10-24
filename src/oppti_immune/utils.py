"""
utils.py — Shared utilities: config loading, logging, I/O helpers.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load the master YAML config file.

    Parameters
    ----------
    config_path : path to config/config.yaml (relative to project root or absolute)

    Returns
    -------
    dict with all configuration parameters
    """
    config_path = Path(config_path)
    if not config_path.exists():
        # Try resolving relative to the project root (two levels up from this file)
        project_root = Path(__file__).resolve().parents[2]
        config_path  = project_root / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Run from the project root or provide an absolute path."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return cfg


def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def results_path(cfg: dict, *parts: str) -> Path:
    """Return a Path inside the results directory, creating it if needed."""
    p = project_root() / cfg["paths"]["results_root"] / Path(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def figures_path(cfg: dict, *parts: str) -> Path:
    """Return a Path inside the figures directory, creating it if needed."""
    p = project_root() / cfg["paths"]["figures_root"] / Path(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def models_path(cfg: dict, *parts: str) -> Path:
    """Return a Path inside the models directory, creating it if needed."""
    p = project_root() / cfg["paths"]["models_root"] / Path(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def data_path(cfg: dict, *parts: str) -> Path:
    """Return a Path inside the data directory."""
    return project_root() / cfg["paths"]["data_root"] / Path(*parts)


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger.

    Parameters
    ----------
    name  : logger name (usually __name__ of the calling module)
    level : logging level (default INFO)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_gene_panel(cfg: dict) -> pd.DataFrame:
    """Load the curated Immune Evasion Proteome gene panel CSV."""
    panel_path = project_root() / cfg["paths"]["gene_panel"]
    df = pd.read_csv(panel_path)
    required_cols = {"gene", "module"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Gene panel must contain columns: {required_cols}")
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to parquet, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save a DataFrame to CSV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)


# ── Miscellaneous ─────────────────────────────────────────────────────────────

def tag_samples(df: pd.DataFrame, cancer_type: str) -> pd.DataFrame:
    """Append a cancer-type suffix to all sample index labels."""
    df = df.copy()
    df.index = df.index.astype(str) + f"_{cancer_type}"
    return df


def harmonize_response(series: pd.Series, cfg: dict) -> pd.Series:
    """Map diverse response labels to binary 1 (responder) / 0 (non-responder)."""
    responders    = set(cfg["validation"]["response_labels"]["responder"])
    nonresponders = set(cfg["validation"]["response_labels"]["nonresponder"])

    def _map(val):
        if pd.isna(val):
            return pd.NA
        s = str(val).strip()
        if s in responders:
            return 1
        elif s in nonresponders:
            return 0
        return pd.NA

    return series.map(_map).astype("Int64")

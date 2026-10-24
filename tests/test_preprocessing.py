"""Unit tests for oppti_immune.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oppti_immune.preprocessing import (
    apply_normalization,
    filter_missing_proteins,
    filter_tumor_samples,
    mad_normalize,
    subset_to_panel,
    zscore_normalize,
)


@pytest.fixture
def toy_proteomics() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = [f"S{i:03d}" for i in range(20)] + ["N01.N", "N02.N", "N03.N"]
    cols = ["CD274", "IDO1", "ARG1", "NOISY_PROT", "TGFB1"]
    data = rng.normal(0, 1, (23, 5))
    # Add some missingness
    data[:, 3] = np.nan                       # NOISY_PROT fully missing
    data[0:5, 0] = np.nan                     # CD274 missing in 5 samples
    return pd.DataFrame(data, index=idx, columns=cols)


def test_filter_tumor_samples(toy_proteomics):
    df = filter_tumor_samples(toy_proteomics)
    # 23 samples; patterns drop the .N-marked adjacent normals (N01, N02)
    # and N03 (simulating the mid-index .N convention, e.g. "C3N-xxxx.N_luad")
    assert len(df) == 20
    assert not df.index.str.contains(r"\.N", regex=True).any()
    assert "normal" not in " ".join(df.index.str.lower())


def test_filter_missing_proteins(toy_proteomics):
    df = filter_missing_proteins(toy_proteomics, threshold=0.5)
    assert "NOISY_PROT" not in df.columns   # 100% missing removed
    assert "CD274" in df.columns             # 5/23 missing → kept


def test_mad_normalize_center_and_scale(toy_proteomics):
    df = toy_proteomics.drop(columns=["NOISY_PROT"])
    normed = mad_normalize(df)
    # Medians should be ~0
    for col in normed.columns:
        assert abs(normed[col].median()) < 1e-10
    # Should not produce inf
    assert np.isfinite(normed.fillna(0).values).all()


def test_zscore_normalize(toy_proteomics):
    df = toy_proteomics.drop(columns=["NOISY_PROT"])
    normed = zscore_normalize(df)
    for col in normed.columns:
        assert abs(normed[col].mean()) < 1e-10
        assert abs(normed[col].std() - 1.0) < 0.1  # close to 1


def test_apply_normalization_dispatch(toy_proteomics):
    df = toy_proteomics.drop(columns=["NOISY_PROT"])
    for method in ("mad", "zscore"):
        out = apply_normalization(df, method)
        assert out.shape == df.shape


def test_apply_normalization_invalid_method(toy_proteomics):
    df = toy_proteomics.drop(columns=["NOISY_PROT"])
    with pytest.raises(ValueError):
        apply_normalization(df, "unknown_method")


def test_subset_to_panel(toy_proteomics):
    panel = ["CD274", "IDO1", "NONEXISTENT_GENE"]
    df = subset_to_panel(toy_proteomics, panel)
    assert set(df.columns) == {"CD274", "IDO1"}

"""Unit tests for oppti_immune.pies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oppti_immune.pies import compute_pies_score, compute_pies_all_samples


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    return pd.DataFrame({
        "gene":   ["CD274", "IDO1", "HLA-A", "STAT1", "TGFB1"],
        "module": ["checkpoint", "checkpoint",
                   "antigen_presentation", "ifn_pathway", "cytokines"],
    })


@pytest.fixture
def module_weights() -> dict[str, float]:
    return {
        "checkpoint":            3.0,
        "antigen_presentation": -2.5,
        "ifn_pathway":          -2.0,
        "cytokines":             2.0,
    }


@pytest.fixture
def pt_fraction() -> pd.Series:
    return pd.Series({
        "CD274": 80.0,  # Heavily post-transcriptional
        "IDO1":  50.0,
        "HLA-A": 20.0,
        "STAT1": 10.0,
        "TGFB1": 70.0,
    })


@pytest.fixture
def cd8_correlation() -> pd.Series:
    return pd.Series({
        "CD274": -0.4,   # Negative: drives evasion
        "IDO1":  -0.3,
        "HLA-A":  0.5,   # Positive: loss reduces CD8 infiltration
        "STAT1":  0.4,
        "TGFB1": -0.2,
    })


def test_pies_zero_when_no_outliers(toy_panel, module_weights, pt_fraction, cd8_correlation):
    outliers = pd.Series({"CD274": 0, "IDO1": 0, "HLA-A": 0, "STAT1": 0, "TGFB1": 0})
    score = compute_pies_score(
        outliers, toy_panel.set_index("gene")["module"],
        module_weights, pt_fraction, cd8_correlation,
    )
    assert score == 0.0


def test_pies_positive_for_checkpoint_outlier(toy_panel, module_weights, pt_fraction, cd8_correlation):
    outliers = pd.Series({"CD274": 1, "IDO1": 0, "HLA-A": 0, "STAT1": 0, "TGFB1": 0})
    score = compute_pies_score(
        outliers, toy_panel.set_index("gene")["module"],
        module_weights, pt_fraction, cd8_correlation,
    )
    # CD274: module weight +3, pt fraction 80% → weight multiplier 1 + 0.8 = 1.8
    # Module weights already encode evasion direction (see pies.py docstring);
    # the score is higher when evasion-related protein outliers are present.
    assert np.isclose(score, 3.0 * 1.8)


def test_pies_post_transcriptional_upweighted(toy_panel, module_weights, pt_fraction, cd8_correlation):
    """Gene with higher PT fraction should contribute more heavily."""
    high_pt = pd.Series({"CD274": 1, "IDO1": 0, "HLA-A": 0, "STAT1": 0, "TGFB1": 0})
    low_pt  = high_pt.copy()

    pt_high = pd.Series({"CD274": 90.0})
    pt_low  = pd.Series({"CD274": 0.0})

    mod_map = toy_panel.set_index("gene")["module"]

    s_high = compute_pies_score(high_pt, mod_map, module_weights, pt_high, cd8_correlation)
    s_low  = compute_pies_score(low_pt,  mod_map, module_weights, pt_low,  cd8_correlation)
    # With negative direction, stronger signal → more negative score
    assert abs(s_high) > abs(s_low)


def test_compute_pies_all_samples_shape(toy_panel, module_weights, pt_fraction, cd8_correlation):
    outlier_mat = pd.DataFrame(
        [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0]],
        index=["S1", "S2", "S3"],
        columns=["CD274", "IDO1", "HLA-A", "STAT1", "TGFB1"],
    )
    cfg = {
        "pies": {
            "pt_weight_multiplier": 1.0,
            "module_weights": module_weights,
        }
    }
    pies = compute_pies_all_samples(
        outlier_mat, toy_panel, pt_fraction, cd8_correlation, cfg
    )
    assert pies.shape == (3,)
    assert list(pies.index) == ["S1", "S2", "S3"]
    assert pies["S2"] == 0.0   # No outliers → zero
    assert pies["S1"] != 0.0
    assert pies["S3"] != 0.0

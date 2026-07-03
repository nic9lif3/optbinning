"""Tests for optbinning_namnh: PSI constraint in optimal binning."""

import numpy as np
import pytest

from optbinning import ContinuousOptimalBinning
from optbinning import OptimalBinning

from optbinning_namnh import PSIContinuousOptimalBinning
from optbinning_namnh import PSIOptimalBinning


EPS = 1e-6


def _psi(p, q, eps=EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    return float(np.sum((p - q) * np.log(p / q)))


def _achieved_psi(splits, x_fit, x_val):
    """PSI over the final bins (using the optimal splits)."""
    splits = np.asarray(splits, dtype=float)
    n_bins = len(splits) + 1
    fi = np.digitize(np.asarray(x_fit, dtype=float), splits, right=False)
    vi = np.digitize(np.asarray(x_val, dtype=float), splits, right=False)
    pf = np.bincount(fi, minlength=n_bins) / len(x_fit)
    pv = np.bincount(vi, minlength=n_bins) / len(x_val)
    return _psi(pf, pv)


def _make_binary(seed=42, n=6000, shift=0.6):
    rng = np.random.RandomState(seed)
    x_fit = rng.normal(0.0, 1.0, n)
    prob = 1.0 / (1.0 + np.exp(-x_fit))
    y = (rng.rand(n) < prob).astype(int)
    # validation set is shifted -> high PSI if many small bins are kept
    x_val = rng.normal(shift, 1.3, n)
    return x_fit, y, x_val


def _make_continuous(seed=7, n=6000, shift=0.6):
    rng = np.random.RandomState(seed)
    x_fit = rng.normal(0.0, 1.0, n)
    y = 2.0 * x_fit + rng.normal(0.0, 0.5, n)
    x_val = rng.normal(shift, 1.3, n)
    return x_fit, y, x_val


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------

def test_binary_none_matches_base():
    x, y, _ = _make_binary()
    base = OptimalBinning(solver="cp").fit(x, y)
    psi = PSIOptimalBinning(solver="cp").fit(x, y)  # psi_threshold=None
    np.testing.assert_array_almost_equal(base.splits, psi.splits)
    assert psi.status == base.status


def test_binary_pairing_validation():
    x, y, xv = _make_binary()
    with pytest.raises(ValueError):
        PSIOptimalBinning(psi_threshold=0.1).fit(x, y)          # missing x_valid
    with pytest.raises(ValueError):
        PSIOptimalBinning().fit(x, y, x_valid=xv)               # missing threshold


def test_binary_constraint_respected():
    x, y, xv = _make_binary()
    thr = 0.05
    ob = PSIOptimalBinning(solver="cp", psi_threshold=thr).fit(
        x, y, x_valid=xv)
    achieved = _achieved_psi(ob.splits, x, xv)
    # small tolerance due to rounding coefficients to int (M=1e6)
    assert achieved <= thr + 1e-3, (achieved, thr)


def test_binary_tighter_threshold_not_more_psi():
    x, y, xv = _make_binary()
    loose = PSIOptimalBinning(solver="cp", psi_threshold=0.5).fit(
        x, y, x_valid=xv)
    tight = PSIOptimalBinning(solver="cp", psi_threshold=0.02).fit(
        x, y, x_valid=xv)
    assert _achieved_psi(tight.splits, x, xv) <= 0.02 + 1e-3
    assert _achieved_psi(loose.splits, x, xv) >= \
        _achieved_psi(tight.splits, x, xv) - 1e-9


# ---------------------------------------------------------------------------
# Continuous
# ---------------------------------------------------------------------------

def test_continuous_none_matches_base():
    x, y, _ = _make_continuous()
    base = ContinuousOptimalBinning().fit(x, y)
    psi = PSIContinuousOptimalBinning().fit(x, y)
    np.testing.assert_array_almost_equal(base.splits, psi.splits)


def test_continuous_constraint_respected():
    x, y, xv = _make_continuous()
    thr = 0.05
    ob = PSIContinuousOptimalBinning(psi_threshold=thr).fit(x, y, x_valid=xv)
    achieved = _achieved_psi(ob.splits, x, xv)
    assert achieved <= thr + 1e-3, (achieved, thr)

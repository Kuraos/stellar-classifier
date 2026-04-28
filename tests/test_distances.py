from __future__ import annotations

import numpy as np

from src.distances import (
    best_distance_bayesian,
    distance_uncertainty_asymmetric,
    absolute_magnitude_bayesian,
)


def test_best_distance_priority_photogeo() -> None:
    r_phot = np.array([100.0, np.nan])
    r_geo = np.array([120.0, 110.0])
    parallax = np.array([10.0, 9.09])
    out = best_distance_bayesian(r_phot, r_geo, parallax)
    assert out[0] == 100.0
    assert np.isfinite(out[1]) and out[1] == 110.0


def test_best_distance_fallback_geo() -> None:
    r_phot = np.array([np.nan])
    r_geo = np.array([150.0])
    parallax = np.array([5.0])
    out = best_distance_bayesian(r_phot, r_geo, parallax)
    assert out[0] == 150.0


def test_best_distance_fallback_parallax() -> None:
    r_phot = np.array([np.nan])
    r_geo = np.array([np.nan])
    parallax = np.array([20.0])
    out = best_distance_bayesian(r_phot, r_geo, parallax)
    assert np.isfinite(out[0]) and np.allclose(out[0], 50.0)


def test_best_distance_negative_parallax_recovered() -> None:
    r_phot = np.array([130.0])
    r_geo = np.array([np.nan])
    parallax = np.array([-5.0])
    out = best_distance_bayesian(r_phot, r_geo, parallax)
    assert out[0] == 130.0


def test_consistency_small_error() -> None:
    # For very small fractional parallax error, bayesian and 1000/parallax should agree
    parallax = np.array([10.0, 20.0])
    # create photogeo near geometric
    r_phot = 1000.0 / parallax * 1.001
    r_geo = np.full_like(r_phot, np.nan)
    out = best_distance_bayesian(r_phot, r_geo, parallax)
    simple = 1000.0 / parallax
    rel_diff = np.abs(out - simple) / simple
    assert np.all(rel_diff < 0.01)


def test_divergence_large_error() -> None:
    parallax = np.array([2.0, 3.0])
    r_phot = np.array([500.0, 300.0])
    r_geo = np.full_like(r_phot, np.nan)
    out = best_distance_bayesian(r_phot, r_geo, parallax)
    assert np.all(np.isfinite(out)) and np.all(out > 0)


def test_uncertainty_asymmetric() -> None:
    r_lo = np.array([90.0])
    r_med = np.array([100.0])
    r_hi = np.array([120.0])
    sigma_lo, sigma_hi = distance_uncertainty_asymmetric(r_lo, r_med, r_hi)
    assert sigma_lo[0] == 10.0
    assert sigma_hi[0] == 20.0


def test_absolute_magnitude_bayesian() -> None:
    g = np.array([10.0])
    d = np.array([100.0])
    m = absolute_magnitude_bayesian(g, d)
    assert np.isfinite(m[0])

from __future__ import annotations

import numpy as np

from src.temperature import (
    absolute_magnitude,
    bv_from_bprp,
    luminosity_solar,
    spectral_type,
    teff_from_bv,
)


def test_bv_from_bprp_solar_like_value() -> None:
    bp_rp = np.array([0.82])
    bv = bv_from_bprp(bp_rp)
    assert 0.60 < bv[0] < 0.90


def test_teff_from_bv_solar_like_value() -> None:
    bv = np.array([0.65])
    teff = teff_from_bv(bv)
    assert 5600.0 < teff[0] < 6000.0


def test_absolute_magnitude_known_case() -> None:
    # m=10, paralaje=10 mas (100 pc) -> M~5
    m_abs = absolute_magnitude(np.array([10.0]), np.array([10.0]))
    assert np.isclose(m_abs[0], 5.0, atol=1e-6)


def test_luminosity_solar_reference() -> None:
    lum = luminosity_solar(np.array([4.74]))
    assert np.isclose(lum[0], 1.0, atol=1e-10)


def test_spectral_type_scalar_and_array() -> None:
    assert spectral_type(15000.0) == "B"
    arr = spectral_type(np.array([32000.0, 5800.0, 3300.0]))
    assert np.array_equal(arr, np.array(["O", "G", "M"]))

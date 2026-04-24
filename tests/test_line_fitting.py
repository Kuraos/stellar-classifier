from __future__ import annotations

import numpy as np

from src.line_fitting import fit_absorption_line, gaussian_absorption


def test_fit_absorption_line_recovers_center() -> None:
    rng = np.random.default_rng(7)
    wavelength = np.linspace(650.0, 660.0, 500)

    true_center = 656.28
    true_sigma = 0.18
    flux_clean = gaussian_absorption(
        wavelength,
        continuum=1.0,
        depth=0.35,
        center=true_center,
        sigma=true_sigma,
    )
    flux = flux_clean + rng.normal(0.0, 0.005, size=wavelength.size)

    result = fit_absorption_line(wavelength, flux, center_guess=656.2, window=1.5)

    assert abs(result["center"] - true_center) < 0.05
    assert abs(result["sigma"] - true_sigma) < 0.08
    assert result["fwhm"] > 0.0
    assert result["equivalent_width"] > 0.0

"""Ajuste simple de lineas espectrales para perfiles de absorcion."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit


def _validate_arrays(wavelength: ArrayLike, flux: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Limpia, valida y ordena los vectores espectrales antes del ajuste."""
    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(flux, dtype=float)
    if x.shape != y.shape:
        raise ValueError("wavelength y flux deben tener la misma forma")
    if x.ndim != 1:
        raise ValueError("wavelength y flux deben ser arreglos 1D")
    if x.size < 5:
        raise ValueError("Se requieren al menos 5 puntos para ajustar")
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    if x_valid.size < 5:
        raise ValueError("No hay suficientes datos finitos para ajustar")
    order = np.argsort(x_valid)
    return x_valid[order], y_valid[order]


def gaussian_absorption(
    wavelength: ArrayLike,
    continuum: float,
    depth: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    """Perfil gaussiano de absorcion sobre un continuo.

    El flujo modelado es: continuum * (1 - depth * exp(-(x-center)^2/(2*sigma^2))).
    """
    x = np.asarray(wavelength, dtype=float)
    return continuum * (1.0 - depth * np.exp(-0.5 * ((x - center) / sigma) ** 2))


def estimate_continuum(wavelength: ArrayLike, flux: ArrayLike, degree: int = 1) -> np.ndarray:
    """Estima el continuo con un ajuste polinomial sobre los puntos mas altos.

    La heuristica toma el percentil 70 de flujo para evitar que la linea de
    absorcion sesgue el ajuste del continuo.
    """
    x, y = _validate_arrays(wavelength, flux)

    threshold = np.nanpercentile(y, 70)
    high_mask = y >= threshold
    if high_mask.sum() <= degree + 1:
        high_mask = np.ones_like(y, dtype=bool)

    coeffs = np.polyfit(x[high_mask], y[high_mask], deg=degree)
    return np.polyval(coeffs, x)


def normalize_flux(wavelength: ArrayLike, flux: ArrayLike, degree: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normaliza el flujo respecto al continuo estimado.

    Retorna el eje espectral limpio, el flujo normalizado y el continuo usado
    para la normalizacion.
    """
    x, y = _validate_arrays(wavelength, flux)
    continuum = estimate_continuum(x, y, degree=degree)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_flux = y / continuum
    return x, norm_flux, continuum


def detect_line_center(wavelength: ArrayLike, flux: ArrayLike) -> float:
    """Detecta el centro de la linea como el minimo de flujo observado."""
    x, y = _validate_arrays(wavelength, flux)
    return float(x[np.argmin(y)])


def fit_absorption_line(
    wavelength: ArrayLike,
    flux: ArrayLike,
    center_guess: float | None = None,
    window: float = 5.0,
) -> dict:
    """Ajusta una linea de absorcion con perfil gaussiano.

    Parametros:
    - wavelength: eje espectral.
    - flux: flujo observado.
    - center_guess: centro inicial en unidades de wavelength.
    - window: semiancho del segmento a ajustar.

    Retorna un diccionario con parametros ajustados y metricas.
    """
    x, y = _validate_arrays(wavelength, flux)

    if center_guess is None:
        center_guess = detect_line_center(x, y)

    mask = (x >= center_guess - window) & (x <= center_guess + window)
    xw = x[mask]
    yw = y[mask]

    if xw.size < 8:
        raise ValueError("No hay suficientes puntos dentro de la ventana de ajuste")

    continuum_arr = estimate_continuum(xw, yw, degree=1)
    baseline = float(np.nanmedian(continuum_arr))

    min_flux = float(np.nanmin(yw))
    depth0 = max(0.01, min(0.9, 1.0 - min_flux / baseline))
    sigma0 = max(1e-3, window / 6.0)

    bounds_lower = [0.0, 0.0, center_guess - window, 1e-5]
    bounds_upper = [np.inf, 1.5, center_guess + window, window]
    p0 = [baseline, depth0, center_guess, sigma0]

    params, covariance = curve_fit(
        gaussian_absorption,
        xw,
        yw,
        p0=p0,
        bounds=(bounds_lower, bounds_upper),
        maxfev=10000,
    )

    continuum_fit, depth_fit, center_fit, sigma_fit = params
    model_flux = gaussian_absorption(xw, continuum_fit, depth_fit, center_fit, sigma_fit)

    fwhm = float(2.354820045 * sigma_fit)
    with np.errstate(divide="ignore", invalid="ignore"):
        ew_profile = 1.0 - (model_flux / continuum_fit)
    # Integramos el perfil normalizado con la regla trapezoidal para obtener EW.
    equivalent_width = float(np.trapezoid(ew_profile, xw))

    rms = float(np.sqrt(np.mean((yw - model_flux) ** 2)))
    param_errors = np.sqrt(np.diag(covariance)) if covariance.size else np.full(4, np.nan)

    return {
        "continuum": float(continuum_fit),
        "depth": float(depth_fit),
        "center": float(center_fit),
        "sigma": float(sigma_fit),
        "fwhm": fwhm,
        "equivalent_width": equivalent_width,
        "rms": rms,
        "parameter_errors": {
            "continuum": float(param_errors[0]),
            "depth": float(param_errors[1]),
            "center": float(param_errors[2]),
            "sigma": float(param_errors[3]),
        },
        "wavelength_fit": xw,
        "flux_fit": yw,
        "model_flux": model_flux,
    }

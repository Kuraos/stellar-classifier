"""Funciones para manipular distancias bayesianas (Bailer-Jones 2021).

Referencia:
Bailer-Jones, C. A. L., et al. 2021, AJ, 161, 147.

Este modulo implementa utilidades puras para seleccionar la mejor
estimacion de distancia entre las columnas entregadas por el catalogo
`external.gaiaedr3_distance` y para convertir distancias en magnitudes
absultas y errores asimetricos.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def best_distance_bayesian(
    r_med_photogeo: ArrayLike,
    r_med_geo: ArrayLike,
    parallax_mas: ArrayLike,
) -> np.ndarray:
    """Devuelve la mejor estimacion de distancia en parsec.

    Prioridad: usar `r_med_photogeo` cuando este disponible (no-NaN),
    fallback a `r_med_geo` si `photogeo` es NaN, y fallback final a la
    distancia geométrica `1000/parallax` cuando ambas estimaciones bayesianas
    son NaN y el paralaje es positivo.

    Maneja entradas vectorizadas y paralajes negativos (devuelve NaN
    cuando no se puede inferir una distancia positiva).

    Referencia: Bailer-Jones et al. (2021), AJ, 161, 147.
    """
    r_phot = np.atleast_1d(np.asarray(r_med_photogeo, dtype=float))
    r_geo = np.atleast_1d(np.asarray(r_med_geo, dtype=float))
    parallax = np.atleast_1d(np.asarray(parallax_mas, dtype=float))

    n = max(r_phot.size, r_geo.size, parallax.size)
    # Broadcast to same length
    if r_phot.size != n:
        r_phot = np.broadcast_to(r_phot, (n,))
    if r_geo.size != n:
        r_geo = np.broadcast_to(r_geo, (n,))
    if parallax.size != n:
        parallax = np.broadcast_to(parallax, (n,))

    result = np.full(n, np.nan, dtype=float)

    # Use photogeo when finite
    mask_phot = np.isfinite(r_phot)
    result[mask_phot] = r_phot[mask_phot]

    # Use geo when photogeo not available
    mask_geo = ~mask_phot & np.isfinite(r_geo)
    result[mask_geo] = r_geo[mask_geo]

    # Fallback to simple geometric distance when both bayesian are missing
    mask_geom = ~np.isfinite(result) & (parallax > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        geom = 1000.0 / parallax
    result[mask_geom] = geom[mask_geom]

    # Any remaining non-positive or non-finite distances -> NaN
    result[~np.isfinite(result)] = np.nan
    result[result <= 0] = np.nan

    return result


def distance_uncertainty_asymmetric(
    r_lo: ArrayLike, r_med: ArrayLike, r_hi: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (sigma_lower, sigma_upper) en parsec.

    sigma_lower = r_med - r_lo
    sigma_upper = r_hi - r_med

    Las salidas conservan NaN cuando cualquiera de las entradas no es finita.

    Referencia: Bailer-Jones et al. (2021), AJ, 161, 147.
    """
    r_lo_a = np.atleast_1d(np.asarray(r_lo, dtype=float))
    r_med_a = np.atleast_1d(np.asarray(r_med, dtype=float))
    r_hi_a = np.atleast_1d(np.asarray(r_hi, dtype=float))

    n = max(r_lo_a.size, r_med_a.size, r_hi_a.size)
    if r_lo_a.size != n:
        r_lo_a = np.broadcast_to(r_lo_a, (n,))
    if r_med_a.size != n:
        r_med_a = np.broadcast_to(r_med_a, (n,))
    if r_hi_a.size != n:
        r_hi_a = np.broadcast_to(r_hi_a, (n,))

    sigma_lower = r_med_a - r_lo_a
    sigma_upper = r_hi_a - r_med_a

    sigma_lower[~np.isfinite(sigma_lower)] = np.nan
    sigma_upper[~np.isfinite(sigma_upper)] = np.nan

    return sigma_lower, sigma_upper


def absolute_magnitude_bayesian(g_mag: ArrayLike, distance_pc: ArrayLike) -> np.ndarray:
    """Calcula la magnitud absoluta M_G usando distancia en parsec.

    M_G = m_G + 5 - 5*log10(d)

    Valida que `distance_pc` > 0; devuelve NaN para valores no fisicos.

    Referencia: formula fotometrica estandar; ver Bailer-Jones et al. (2021)
    para contexto sobre las distancias usadas.
    """
    g = np.atleast_1d(np.asarray(g_mag, dtype=float))
    d = np.atleast_1d(np.asarray(distance_pc, dtype=float))

    n = max(g.size, d.size)
    if g.size != n:
        g = np.broadcast_to(g, (n,))
    if d.size != n:
        d = np.broadcast_to(d, (n,))

    with np.errstate(divide="ignore", invalid="ignore"):
        m_abs = g + 5.0 - 5.0 * np.log10(d)

    m_abs[~np.isfinite(m_abs)] = np.nan
    m_abs[d <= 0] = np.nan
    return m_abs

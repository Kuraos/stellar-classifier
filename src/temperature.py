"""Conversiones fisicas para clasificacion estelar.

Este modulo implementa relaciones empiricas para pasar de fotometria Gaia
(BP-RP y magnitud G) a parametros fisicos aproximados.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike


def _as_array(values: ArrayLike) -> np.ndarray:
    """Convierte cualquier entrada numerica en un arreglo 1D de float."""
    return np.atleast_1d(np.asarray(values, dtype=float))


def bv_from_bprp(bp_rp: ArrayLike) -> np.ndarray:
    """Convierte color Gaia BP-RP a color Johnson B-V.

    Referencia:
    Evans et al. (2018), A&A, 616, A4.
    B-V = 0.0981 + 0.7119*(BP-RP) + 0.0718*(BP-RP)^2
    """
    bp_rp_arr = _as_array(bp_rp)
    return 0.0981 + 0.7119 * bp_rp_arr + 0.0718 * bp_rp_arr**2


def teff_from_bv(bv: ArrayLike) -> np.ndarray:
    """Estima temperatura efectiva T_eff a partir de B-V.

    Referencia:
    Ballesteros (2012), EPL, 97, 34008.
    T_eff = 4600 * [1/(0.92*(B-V)+1.7) + 1/(0.92*(B-V)+0.62)]

    La aproximacion es valida principalmente para 0.0 < B-V < 2.0.
    """
    bv_arr = _as_array(bv)
    with np.errstate(divide="ignore", invalid="ignore"):
        teff = 4600.0 * (
            1.0 / (0.92 * bv_arr + 1.7) + 1.0 / (0.92 * bv_arr + 0.62)
        )
    teff[~np.isfinite(teff)] = np.nan
    return teff


def absolute_magnitude(g_mag: ArrayLike, parallax_mas: ArrayLike) -> np.ndarray:
    """Calcula magnitud absoluta M_G desde magnitud aparente y paralaje.

    Referencia:
    Relacion fotometrica estandar usada en Gaia DR3 para fuentes con
    paralaje positiva y de buena calidad.

    M_G = m_G + 5 + 5*log10(parallax_mas / 1000)
    """
    g_arr = _as_array(g_mag)
    parallax_arr = _as_array(parallax_mas)

    with np.errstate(divide="ignore", invalid="ignore"):
        m_abs = g_arr + 5.0 + 5.0 * np.log10(parallax_arr / 1000.0)

    m_abs[parallax_arr <= 0] = np.nan
    m_abs[~np.isfinite(m_abs)] = np.nan
    return m_abs


def luminosity_solar(m_abs: ArrayLike, m_sun: float = 4.74) -> np.ndarray:
    """Convierte magnitud absoluta a luminosidad en unidades solares.

    Referencia:
    Relacion bolometrica simplificada usada de forma comun en didactica
    astronimica para estimar L/L_sun a partir de magnitud absoluta.

    L/L_sun = 10^((M_sun - M_G)/2.5)
    """
    m_arr = _as_array(m_abs)
    with np.errstate(over="ignore", invalid="ignore"):
        lum = 10.0 ** ((m_sun - m_arr) / 2.5)
    lum[~np.isfinite(lum)] = np.nan
    return lum


def spectral_type(teff: Union[float, ArrayLike]) -> Union[str, np.ndarray]:
    """Asigna tipo espectral de Harvard a partir de T_eff.

    Referencia:
    Clasificacion OBAFGKM basada en rangos de temperatura efectiva usados
    en astronomia estelar introductoria.
    """
    teff_arr = np.asarray(teff, dtype=float)
    is_scalar = teff_arr.ndim == 0
    values = np.atleast_1d(teff_arr)

    types = np.full(values.shape, "M", dtype="<U1")
    types[values >= 3700.0] = "K"
    types[values >= 5200.0] = "G"
    types[values >= 6000.0] = "F"
    types[values >= 7500.0] = "A"
    types[values >= 10000.0] = "B"
    types[values >= 30000.0] = "O"
    types[~np.isfinite(values)] = "?"

    if is_scalar:
        return str(types[0])
    return types

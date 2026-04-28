"""Corrección de extinción interestelar para muestras Gaia.

La implementación consulta Bayestar19 a través de ``dustmaps`` para estimar
``E(B-V)`` y aplica las relaciones de Casagrande et al. (2018) para derivar
las correcciones fotométricas usadas por la GUI.
"""

from __future__ import annotations

from functools import lru_cache
from collections.abc import Callable

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from src.temperature import (
    absolute_magnitude,
    bv_from_bprp,
    luminosity_solar,
    spectral_type,
    teff_from_bv,
)

RV_DEFAULT = 3.1
AG_PER_AV = 0.789
EBR_PER_AV = 0.415


@lru_cache(maxsize=1)
def _get_bayestar_query():
    """Construye y reutiliza una instancia de BayestarQuery.

    Cargar Bayestar2019 desde disco es costoso, así que se cachea para evitar
    volver a leer el HDF5 completo en cada corrección.
    """
    from dustmaps.bayestar import BayestarQuery

    return BayestarQuery(version="bayestar2019")


def prime_bayestar_cache() -> object:
    """Precarga Bayestar2019 y deja la instancia en la caché local.

    Esta funcion se usa al arrancar la GUI para que la primera correccion de
    extincion no pague el coste de cargar el mapa desde disco.
    """
    return _get_bayestar_query()


def _as_array(values: ArrayLike) -> np.ndarray:
    """Convierte la entrada en un arreglo NumPy de tipo float."""
    return np.atleast_1d(np.asarray(values, dtype=float))


def _build_galactic_coordinates(df: pd.DataFrame, distance_col: str = "distance_pc") -> SkyCoord:
    """Construye coordenadas galacticas usando la columna de distancia indicada.

    Si `distance_col` no existe en el DataFrame, se intenta calcular `distance_pc`
    como `1000/parallax` para mantener compatibilidad hacia atras.
    """
    ra = df["ra"].to_numpy(dtype=float)
    dec = df["dec"].to_numpy(dtype=float)

    if distance_col in df.columns:
        distance_pc = df[distance_col].to_numpy(dtype=float)
    else:
        parallax = df["parallax"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            distance_pc = 1000.0 / parallax
        distance_pc[parallax <= 0] = np.nan

    return SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=distance_pc * u.pc,
        frame="icrs",
    ).galactic


def _query_bayestar_reddening(coords: SkyCoord) -> np.ndarray:
    """Consulta Bayestar19 y devuelve ``E(B-V)`` en magnitudes.

    Referencia:
    Green et al. (2019), Bayestar19, usado desde ``dustmaps``.
    """
    try:
        query = _get_bayestar_query()
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError(
            "La correccion de extincion requiere instalar dustmaps."
        ) from exc
    try:
        reddening = query(coords, mode="median")
    except Exception as exc:  # pragma: no cover - depende del mapa local
        raise RuntimeError("No fue posible consultar Bayestar19.") from exc

    return np.ma.asarray(reddening, dtype=float).filled(np.nan)


def _broadcast_to_length(values: ArrayLike, length: int) -> np.ndarray:
    """Normaliza una salida escalar o vectorial al tamano esperado."""
    array = _as_array(values)
    if array.size == length:
        return array
    if array.size == 1 and length > 1:
        return np.full(length, float(array[0]), dtype=float)
    if length == 1 and array.size == 1:
        return array
    raise RuntimeError(
        "La consulta de Bayestar no devolvio un tamano compatible con el DataFrame."
    )


def apply_extinction_correction(
    df: pd.DataFrame,
    reddening_query: Callable[[SkyCoord], ArrayLike] | None = None,
    distance_col: str = "distance_pc",
) -> pd.DataFrame:
    """Aplica correccion de extincion interestelar a una muestra Gaia.

    La funcion conserva las columnas originales y agrega:
    ``A_V``, ``A_G``, ``E_BR``, ``BP_RP_corr``, ``B_V_corr``, ``teff_corr``,
    ``M_G_corr``, ``luminosity_solar_corr`` y ``spectral_type_corr``.
    Cuando la correccion esta activa, las columnas canonicas ``B_V``,
    ``teff``, ``M_G``, ``luminosity_solar`` y ``spectral_type`` se reemplazan
    por sus valores corregidos para que la GUI reutilice el flujo existente.

    Referencias:
    - Green et al. (2019), Bayestar19, consultado via ``dustmaps``.
    - Casagrande et al. (2018), conversiones ``A_G = 0.789 A_V`` y
      ``E(BP-RP) = 0.415 A_V``.
    """
    required_columns = {"ra", "dec", "parallax", "bp_rp", "phot_g_mean_mag"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas para corregir extincion: {sorted(missing)}")

    output = df.copy()
    if distance_col not in output.columns:
        # Mantener compatibilidad: rellenar distance_col con 1000/parallax si es necesario
        with np.errstate(divide="ignore", invalid="ignore"):
            parallax = output["parallax"].to_numpy(dtype=float)
            fallback = 1000.0 / parallax
        fallback[parallax <= 0] = np.nan
        output[distance_col] = fallback

    coords = _build_galactic_coordinates(output, distance_col=distance_col)
    query = reddening_query or _query_bayestar_reddening
    reddening_ebv = _broadcast_to_length(query(coords), len(output))

    a_v = RV_DEFAULT * reddening_ebv
    a_g = AG_PER_AV * a_v
    e_br = EBR_PER_AV * a_v

    bp_rp_corr = output["bp_rp"].to_numpy(dtype=float) - e_br
    b_v_corr = bv_from_bprp(bp_rp_corr)
    teff_corr = teff_from_bv(b_v_corr)

    if "M_G" in output.columns:
        m_g_raw = output["M_G"].to_numpy(dtype=float)
    else:
        # Si no hay M_G, construir desde la magnitud aparente y la distancia geometrica
        m_g_raw = absolute_magnitude(
            output["phot_g_mean_mag"].to_numpy(dtype=float),
            output["parallax"].to_numpy(dtype=float),
        )

    m_g_corr = m_g_raw - a_g
    luminosity_corr = luminosity_solar(m_g_corr)
    spectral_corr = spectral_type(teff_corr)

    output["A_V"] = a_v
    output["A_G"] = a_g
    output["E_BR"] = e_br
    output["BP_RP_corr"] = bp_rp_corr
    output["B_V_corr"] = b_v_corr
    output["teff_corr"] = teff_corr
    output["M_G_corr"] = m_g_corr
    output["luminosity_solar_corr"] = luminosity_corr
    output["spectral_type_corr"] = spectral_corr

    output["B_V"] = b_v_corr
    output["teff"] = teff_corr
    output["M_G"] = m_g_corr
    output["luminosity_solar"] = luminosity_corr
    output["spectral_type"] = spectral_corr

    return output
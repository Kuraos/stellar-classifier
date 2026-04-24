"""Descarga de una muestra estelar desde Gaia DR3 usando ADQL."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

DATA_OUTPUT = Path(__file__).resolve().parent / "gaia_sample.csv"
REQUIRED_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "teff_gspphot",
    "lum_flame",
    "radius_flame",
]


def _query_once_with_fallback(Gaia, query: str) -> pd.DataFrame:
    """Intenta primero consulta asincrona y si falla usa consulta sincronica."""
    async_error: Exception | None = None

    try:
        print("[Gaia] Metodo asincrono...")
        job = Gaia.launch_job_async(query, dump_to_file=False)
        return job.get_results().to_pandas()
    except Exception as exc:
        async_error = exc
        print(f"[Gaia] Fallo asincrono: {exc}")

    if not hasattr(Gaia, "launch_job"):
        raise RuntimeError(f"Fallo asincrono y no hay fallback sincronico: {async_error}")

    try:
        print("[Gaia] Intentando fallback sincronico...")
        job = Gaia.launch_job(query, dump_to_file=False)
        return job.get_results().to_pandas()
    except Exception as sync_exc:
        raise RuntimeError(
            "Fallo consulta asincrona y sincronica "
            f"(asincrona: {async_error}; sincronica: {sync_exc})"
        ) from sync_exc


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Faltan columnas esperadas en Gaia DR3: {missing}")


def _build_query(n_stars: int, max_dist_pc: float) -> str:
    min_parallax = 1000.0 / max_dist_pc
    return f"""
    SELECT TOP {int(n_stars)}
        source_id,
        ra,
        dec,
        parallax,
        parallax_error,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag,
        bp_rp,
        teff_gspphot,
        lum_flame,
        radius_flame,
        ruwe,
        phot_bp_rp_excess_factor
    FROM gaiadr3.gaia_source
    WHERE parallax > {min_parallax}
      AND parallax_error / parallax < 0.1
      AND ruwe < 1.4
      AND phot_bp_rp_excess_factor < 1.5
      AND bp_rp IS NOT NULL
      AND parallax IS NOT NULL
      AND phot_g_mean_mag IS NOT NULL
      AND phot_bp_mean_mag IS NOT NULL
      AND phot_rp_mean_mag IS NOT NULL
    """.strip()


def query_gaia_sample(n_stars: int = 5000, max_dist_pc: float = 100) -> pd.DataFrame:
    """Consulta Gaia DR3 y devuelve un DataFrame con una muestra de estrellas.

    Parametros:
    - n_stars: numero maximo de filas a solicitar.
    - max_dist_pc: distancia maxima del volumen de consulta en parsec.

    Salida:
    - DataFrame con columnas astrometricas y fotometricas.

    Efecto lateral:
    - Guarda una copia de la muestra en data/gaia_sample.csv.
    """
    if n_stars <= 0:
        raise ValueError("n_stars debe ser mayor a 0")
    if max_dist_pc <= 0:
        raise ValueError("max_dist_pc debe ser mayor a 0")

    from astroquery.gaia import Gaia

    query = _build_query(n_stars=n_stars, max_dist_pc=max_dist_pc)
    print(f"[Gaia] Preparando consulta para TOP {n_stars} y distancia <= {max_dist_pc} pc")

    Gaia.ROW_LIMIT = -1
    max_retries = 3
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Gaia] Ejecutando consulta (intento {attempt}/{max_retries})...")
            df = _query_once_with_fallback(Gaia, query)

            if df.empty:
                raise RuntimeError("La consulta devolvio 0 filas")

            _validate_required_columns(df)

            DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_OUTPUT, index=False)
            print(f"[Gaia] Muestra descargada: {len(df)} filas")
            print(f"[Gaia] CSV guardado en: {DATA_OUTPUT}")
            return df
        except Exception as exc:  # pragma: no cover - cobertura via test con mocks
            last_error = exc
            print(f"[Gaia] Error en intento {attempt}: {exc}")
            if attempt < max_retries:
                wait_seconds = min(2 ** (attempt - 1), 8)
                print(f"[Gaia] Reintentando en {wait_seconds} segundos...")
                time.sleep(wait_seconds)

    raise RuntimeError(f"No fue posible descargar datos de Gaia DR3: {last_error}")

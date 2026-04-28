"""Descarga de una muestra estelar desde Gaia DR3 usando ADQL.

La consulta combina `gaiadr3.gaia_source` con tablas auxiliares de Gaia para
recuperar magnitudes, paralaje, parametros estelares derivados y columnas de
variabilidad cuando estan disponibles.
"""

from __future__ import annotations

from io import StringIO
import time
from pathlib import Path

import pandas as pd
import requests

DATA_OUTPUT = Path(__file__).resolve().parent / "gaia_sample.csv"
GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_REQUEST_TIMEOUT_SECONDS = 300
GAIA_MAX_RETRIES = 3
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
    "r_med_geo",
    "r_lo_geo",
    "r_hi_geo",
    "r_med_photogeo",
    "r_lo_photogeo",
    "r_hi_photogeo",
]


def _query_gaia_tap_csv(query: str, timeout_seconds: int = GAIA_REQUEST_TIMEOUT_SECONDS) -> pd.DataFrame:
    """Consulta Gaia TAP en modo sincronico y parsea el CSV devuelto.

    La llamada usa un timeout amplio porque Gaia puede tardar bastante en
    procesar consultas con filtros y joins sobre DR3.
    """
    payload = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": query,
    }

    try:
        print(f"[Gaia] Enviando consulta TAP sincronica con timeout {timeout_seconds}s...")
        response = requests.post(GAIA_TAP_SYNC_URL, data=payload, timeout=timeout_seconds)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Fallo la descarga desde Gaia TAP: {exc}") from exc

    text = response.text.strip()
    if not text:
        raise RuntimeError("La consulta de Gaia devolvio una respuesta vacia")

    try:
        return pd.read_csv(StringIO(text))
    except Exception as exc:
        raise RuntimeError(f"No fue posible interpretar el CSV devuelto por Gaia: {exc}") from exc


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Verifica que el resultado contenga las columnas minimas del proyecto."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Faltan columnas esperadas en Gaia DR3: {missing}")


def _build_query(n_stars: int, max_dist_pc: float) -> str:
    """Construye la consulta ADQL con filtros de calidad y el join astrofisico."""
    min_parallax = 1000.0 / max_dist_pc
    return f"""
    SELECT TOP {int(n_stars)}
        gs.source_id,
        gs.ra,
        gs.dec,
        gs.parallax,
        gs.parallax_error,
        gs.phot_g_mean_mag,
        gs.phot_bp_mean_mag,
        gs.phot_rp_mean_mag,
        gs.bp_rp,
        ap.teff_gspphot,
        ap.lum_flame,
        ap.radius_flame,
        gs.ruwe,
        gs.phot_bp_rp_excess_factor,
        d.r_med_geo,
        d.r_lo_geo,
        d.r_hi_geo,
        d.r_med_photogeo,
        d.r_lo_photogeo,
        d.r_hi_photogeo,
        vs.in_vari_classification_result,
        vcr.best_class_name,
        vcr.best_class_score,
        vc.pf AS cepheid_period,
        vr.pf AS rrlyrae_period
    FROM gaiadr3.gaia_source AS gs
    LEFT JOIN gaiadr3.astrophysical_parameters AS ap
        ON gs.source_id = ap.source_id
    LEFT JOIN external.gaiaedr3_distance AS d
        ON gs.source_id = d.source_id
    LEFT JOIN gaiadr3.vari_summary AS vs
        ON gs.source_id = vs.source_id
    LEFT JOIN gaiadr3.vari_classifier_result AS vcr
        ON gs.source_id = vcr.source_id
    LEFT JOIN gaiadr3.vari_cepheid AS vc
        ON gs.source_id = vc.source_id
    LEFT JOIN gaiadr3.vari_rrlyrae AS vr
        ON gs.source_id = vr.source_id
    WHERE gs.parallax > {min_parallax}
        AND gs.parallax_error / gs.parallax < 0.1
        AND gs.ruwe < 1.4
        AND gs.phot_bp_rp_excess_factor < 1.5
        AND gs.bp_rp IS NOT NULL
        AND gs.parallax IS NOT NULL
        AND gs.phot_g_mean_mag IS NOT NULL
        AND gs.phot_bp_mean_mag IS NOT NULL
        AND gs.phot_rp_mean_mag IS NOT NULL
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

    query = _build_query(n_stars=n_stars, max_dist_pc=max_dist_pc)
    print(f"[Gaia] Preparando consulta para TOP {n_stars} y distancia <= {max_dist_pc} pc")

    last_error: Exception | None = None

    # Gaia puede devolver errores transitorios, asi que damos algunos reintentos cortos.
    for attempt in range(1, GAIA_MAX_RETRIES + 1):
        try:
            print(f"[Gaia] Ejecutando consulta (intento {attempt}/{GAIA_MAX_RETRIES})...")
            df = _query_gaia_tap_csv(query)

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
            if attempt < GAIA_MAX_RETRIES:
                wait_seconds = min(2 ** (attempt - 1), 8)
                print(f"[Gaia] Reintentando en {wait_seconds} segundos...")
                time.sleep(wait_seconds)

    raise RuntimeError(f"No fue posible descargar datos de Gaia DR3: {last_error}")

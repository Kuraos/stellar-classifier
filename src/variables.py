"""Herramientas para clasificación y validación de estrellas variables.

Este módulo ofrece funciones vectorizadas y seguras para:
- normalizar etiquetas Gaia DR3 de variabilidad,
- estimar distancias por relaciones período-luminosidad para Cepheids y
  RR Lyrae con coeficientes de referencia sencillos pero más realistas,
- enriquecer un DataFrame con columnas `variable_type`, `is_variable`,
  `pl_period_days` y `distance_pc_PL`.

Las relaciones y constantes aquí usadas son aproximadas y pensadas para
validación rápida en GUI; no sustituyen una calibración científica detallada.
"""

from __future__ import annotations

import re
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


VARIABLE_MARKERS: dict[str, str] = {
    "T2CEP": "T2CEP",
    "DCEPS": "DCEP",
    "DCEP": "DCEP",
    "CEPHEID": "DCEP",
    "RRAB": "RRAB",
    "RRC": "RRC",
    "RRD": "RRC",
    "ECL": "ECL",
    "EA": "ECL",
    "EB": "ECL",
    "EW": "ECL",
    "MIRA": "MIRA",
    "SRV": "MIRA",
    "SR": "MIRA",
    "BY_DRA": "ROT",
    "ROT": "ROT",
    "SPOT": "ROT",
}

VARIABLE_LABELS: dict[str, str] = {
    "DCEP": "Cefeida clásica",
    "T2CEP": "Cefeida tipo II",
    "RRAB": "RR Lyrae ab",
    "RRC": "RR Lyrae c",
    "ECL": "Eclipsante",
    "MIRA": "Mira/Semirregular",
    "ROT": "Rotacional",
    "OTHER": "Otras",
}

VARIABLE_PLOT_STYLE: dict[str, dict[str, object]] = {
    "DCEP": {"marker": "*", "color": "gold", "zorder": 6, "size": 80},
    "T2CEP": {"marker": "P", "color": "darkorange", "zorder": 6, "size": 70},
    "RRAB": {"marker": "s", "color": "crimson", "zorder": 6, "size": 50},
    "RRC": {"marker": "D", "color": "salmon", "zorder": 6, "size": 50},
    "ECL": {"marker": "o", "color": "lightblue", "zorder": 5, "size": 40},
    "MIRA": {"marker": "v", "color": "darkred", "zorder": 6, "size": 60},
    "ROT": {"marker": "^", "color": "mediumpurple", "zorder": 5, "size": 40},
    "OTHER": {"marker": "X", "color": "gray", "zorder": 4, "size": 40},
}

PERIOD_LUMINOSITY_TYPES: set[str] = {"DCEP", "T2CEP", "RRAB", "RRC"}


def _as_array(values: ArrayLike) -> np.ndarray:
    """Convierte una entrada escalar o vectorial a ndarray float 1D."""
    return np.atleast_1d(np.asarray(values, dtype=float))


def _normalize_text(value: object) -> str:
    """Normaliza una etiqueta Gaia para comparaciones robustas."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _finite_period(period_days: object) -> float:
    """Devuelve periodo positivo o NaN si no es util."""
    if period_days is None:
        return float("nan")
    if isinstance(period_days, (int, float, np.integer, np.floating)):
        period = float(period_days)
    elif isinstance(period_days, str):
        try:
            period = float(period_days)
        except Exception:
            return float("nan")
    else:
        return float("nan")
    if not np.isfinite(period) or period <= 0:
        return float("nan")
    return period


def classify_variable_type(
    class_name: object,
    period_days: object | None = None,
    classification_result: object | None = None,
) -> str:
    """Normaliza una clase Gaia a una categoría estable.

    Devuelve 'non_variable' si no hay evidencia de variabilidad.
    """
    cleaned = _normalize_text(class_name)
    if cleaned:
        for marker, normalized in VARIABLE_MARKERS.items():
            if (
                cleaned == marker
                or cleaned.startswith(f"{marker}_")
                or cleaned.endswith(f"_{marker}")
            ):
                return normalized

    if classification_result is not None:
        try:
            if not bool(classification_result):
                return "non_variable"
        except Exception:
            pass

    if _finite_period(period_days) > 0:
        return "OTHER"

    return "non_variable"


def cepheid_distance(g_mag: ArrayLike, period_days: ArrayLike, is_type2: bool = False) -> np.ndarray:
    """Estima distancia para Cepheids con una relación P-L vectorizada.

    Coeficientes aproximados (G band) basados en valores típicos:
    - classical Cepheid: M_G = -2.76 * logP - 1.40
    - type II Cepheid: M_G = -1.95 * logP - 0.90
    """
    g = _as_array(g_mag)
    period = _as_array(period_days)
    n = max(g.size, period.size)
    if g.size != n:
        g = np.broadcast_to(g, (n,))
    if period.size != n:
        period = np.broadcast_to(period, (n,))

    with np.errstate(divide="ignore", invalid="ignore"):
        log_period = np.log10(period)

    m_abs = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(g) & np.isfinite(log_period) & (period > 0)
    # Classical Cepheid coefficients (slope and zero-point) adopted from
    # Freedman et al. (2001) / Benedict et al. (2007) style calibrations
    a = -2.76
    b = -1.40
    if is_type2:
        # Type II Cepheid approximate coefficients (Matsunaga et al. 2006, adapted)
        m_abs[valid] = -1.95 * log_period[valid] - 0.90
    else:
        # Coefficients (classical Cepheid) from Freedman/Benedict-style calibrations
        m_abs[valid] = a * log_period[valid] + b

    with np.errstate(divide="ignore", invalid="ignore"):
        distance_pc = 10.0 ** ((g - m_abs + 5.0) / 5.0)
    distance_pc[~np.isfinite(distance_pc)] = np.nan
    distance_pc[~valid] = np.nan
    distance_pc[distance_pc <= 0] = np.nan
    return cast(np.ndarray, distance_pc)


def rrlyrae_distance(
    g_mag: ArrayLike,
    period_days: ArrayLike,
    metallicity: ArrayLike | float = -1.5,
) -> np.ndarray:
    """Estima distancia para RR Lyrae con calibración óptica aproximada.

    M_G ~ 0.32*[Fe/H] + 1.11 + 0.18*logP  (coeficientes aproximados)
    """
    g = _as_array(g_mag)
    period = _as_array(period_days)
    feh = _as_array(metallicity)
    n = max(g.size, period.size, feh.size)
    if g.size != n:
        g = np.broadcast_to(g, (n,))
    if period.size != n:
        period = np.broadcast_to(period, (n,))
    if feh.size != n:
        feh = np.broadcast_to(feh, (n,))

    # Use a MV-[Fe/H] relation and approximate MV ~ MG for lack of band conversion
    valid = np.isfinite(g) & np.isfinite(period) & np.isfinite(feh) & (period > 0)
    m_abs = np.full(n, np.nan, dtype=float)
    # M_V = alpha * [Fe/H] + beta  (adopted alpha=0.214, beta=0.88)
    alpha = 0.214
    beta = 0.88
    m_abs[valid] = alpha * feh[valid] + beta

    with np.errstate(divide="ignore", invalid="ignore"):
        distance_pc = 10.0 ** ((g - m_abs + 5.0) / 5.0)
    distance_pc[~np.isfinite(distance_pc)] = np.nan
    distance_pc[~valid] = np.nan
    distance_pc[distance_pc <= 0] = np.nan
    return cast(np.ndarray, distance_pc)


def _apparent_g_magnitude(df: pd.DataFrame) -> np.ndarray:
    """Elige la mejor magnitud G disponible para la estimación P-L."""
    if "phot_g_mean_mag_corr" in df.columns:
        return df["phot_g_mean_mag_corr"].to_numpy(dtype=float)
    # TODO: aplicar transformación G→V de Evans et al. (2018) para precisión publicable.
    # Actualmente usamos G como aproximación.
    return df["phot_g_mean_mag"].to_numpy(dtype=float)


def add_variability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas de variabilidad sin modificar el original."""
    output = df.copy()
    n_rows = len(output)
    if n_rows == 0:
        output["variable_type"] = pd.Series(dtype=object)
        output["is_variable"] = pd.Series(dtype=bool)
        output["pl_period_days"] = pd.Series(dtype=float)
        output["distance_pc_PL"] = pd.Series(dtype=float)
        return output

    class_names = output.get("best_class_name", pd.Series([None] * n_rows)).tolist()
    classification_flag = output.get("in_vari_classification_result", pd.Series([False] * n_rows)).tolist()
    cepheid_periods = output.get("cepheid_period", pd.Series([np.nan] * n_rows)).tolist()
    rrlyrae_periods = output.get("rrlyrae_period", pd.Series([np.nan] * n_rows)).tolist()

    variable_types: list[str] = []
    periods: list[float] = []
    distances: list[float] = []

    g_mag = _apparent_g_magnitude(output)
    metallicity = output["mh"].to_numpy(dtype=float) if "mh" in output.columns else np.full(n_rows, -1.5, dtype=float)

    for idx, class_name in enumerate(class_names):
        cepheid_period = _finite_period(cepheid_periods[idx] if idx < len(cepheid_periods) else None)
        rrlyrae_period = _finite_period(rrlyrae_periods[idx] if idx < len(rrlyrae_periods) else None)

        variable_type = classify_variable_type(
            class_name,
            period_days=cepheid_period if np.isfinite(cepheid_period) else rrlyrae_period,
            classification_result=classification_flag[idx] if idx < len(classification_flag) else None,
        )

        # Gaia puede devolver clases genéricas como SOLAR_LIKE aunque la fila
        # venga de tablas específicas de Cepheids o RR Lyrae. En ese caso,
        # usamos el periodo disponible para fijar un subtipo utilizable para P-L.
        if variable_type in {"non_variable", "OTHER"}:
            if np.isfinite(cepheid_period):
                variable_type = "DCEP"
            elif np.isfinite(rrlyrae_period):
                variable_type = "RRAB"
            elif bool(classification_flag[idx] if idx < len(classification_flag) else False):
                variable_type = "OTHER"

        period_value = np.nan
        distance_value = np.nan
        if variable_type in {"DCEP", "T2CEP"}:
            period_value = cepheid_period
            if np.isfinite(period_value):
                distance_value = float(
                    cepheid_distance(
                        g_mag[idx],
                        period_value,
                        is_type2=variable_type == "T2CEP",
                    )[0]
                )
        elif variable_type in {"RRAB", "RRC"}:
            period_value = rrlyrae_period
            if np.isfinite(period_value):
                distance_value = float(
                    rrlyrae_distance(
                        g_mag[idx],
                        period_value,
                        metallicity=metallicity[idx] if idx < len(metallicity) else -1.5,
                    )[0]
                )
        elif np.isfinite(cepheid_period):
            period_value = cepheid_period
        elif np.isfinite(rrlyrae_period):
            period_value = rrlyrae_period

        variable_types.append(variable_type)
        periods.append(period_value)
        distances.append(distance_value)

    output["variable_type"] = variable_types
    output["is_variable"] = [value != "non_variable" for value in variable_types]
    output["pl_period_days"] = periods
    output["distance_pc_PL"] = distances
    return output


def compare_distances(
    df: pd.DataFrame,
    distance_col: str = "distance_pc",
    pl_col: str = "distance_pc_PL",
) -> dict[str, object]:
    """Compara distancias P-L contra una columna de referencia.

    Devuelve un resumen con número de objetos comparados y estadísticas de la
    diferencia absoluta y fraccional.
    """
    if df is None or df.empty:
        return {"n_compared": 0}

    if distance_col not in df.columns or pl_col not in df.columns:
        return {"n_compared": 0}

    mask = np.isfinite(df[distance_col].to_numpy(dtype=float)) & np.isfinite(df[pl_col].to_numpy(dtype=float))
    if not mask.any():
        return {"n_compared": 0}

    ref = df.loc[mask, distance_col].to_numpy(dtype=float)
    pl = df.loc[mask, pl_col].to_numpy(dtype=float)
    abs_diff = np.abs(pl - ref)
    frac = abs_diff / np.maximum(ref, 1.0)
    return {
        "n_compared": int(mask.sum()),
        "median_abs_diff_pc": float(np.median(abs_diff)),
        "median_frac_diff": float(np.median(frac)),
        "max_abs_diff_pc": float(np.nanmax(abs_diff)),
    }

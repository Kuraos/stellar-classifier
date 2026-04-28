"""Calculos estadisticos para la interfaz de stellar-classifier."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

SPECTRAL_ORDER = ["O", "B", "A", "F", "G", "K", "M"]


def _nan_summary(values: pd.Series) -> Dict[str, float]:
    """Resume una serie numérica ignorando valores no finitos."""
    arr = values.to_numpy(dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "std": float(np.nanstd(arr)),
    }


def compute_statistics(df: pd.DataFrame) -> dict:
    """Calcula metricas globales que se muestran en paneles de la GUI.

    Espera un DataFrame procesado con al menos estas columnas:
    - teff
    - M_G
    - distance_pc
    - luminosity_solar
    - spectral_type
    """
    required_cols = {
        "teff",
        "M_G",
        "distance_pc",
        "luminosity_solar",
        "spectral_type",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas para estadisticas: {sorted(missing)}")

    teff_stats = _nan_summary(df["teff"])
    teff_stats["min"] = float(np.nanmin(df["teff"].to_numpy(dtype=float)))
    teff_stats["max"] = float(np.nanmax(df["teff"].to_numpy(dtype=float)))

    mg_stats = _nan_summary(df["M_G"])

    dist_stats = {
        "mean": float(np.nanmean(df["distance_pc"].to_numpy(dtype=float))),
        "median": float(np.nanmedian(df["distance_pc"].to_numpy(dtype=float))),
        "max": float(np.nanmax(df["distance_pc"].to_numpy(dtype=float))),
    }

    lum_stats = {
        "mean": float(np.nanmean(df["luminosity_solar"].to_numpy(dtype=float))),
        "median": float(np.nanmedian(df["luminosity_solar"].to_numpy(dtype=float))),
        "min": float(np.nanmin(df["luminosity_solar"].to_numpy(dtype=float))),
        "max": float(np.nanmax(df["luminosity_solar"].to_numpy(dtype=float))),
    }

    # Conservamos el orden OBAFGKM para que el panel sea estable aunque falten clases.
    counts = (
        df["spectral_type"]
        .astype(str)
        .str.upper()
        .value_counts(dropna=False)
        .to_dict()
    )
    spectral_distribution = {key: int(counts.get(key, 0)) for key in SPECTRAL_ORDER}

    # Comparacion entre distancia simple (1000/parallax) y bayesiana si existe
    distance_comparison = None
    if "distance_pc_bayesian" in df.columns:
        simple = df["distance_pc"].to_numpy(dtype=float)
        bay = df["distance_pc_bayesian"].to_numpy(dtype=float)
        # diferencia absoluta donde ambos son finitos
        mask = np.isfinite(simple) & np.isfinite(bay)
        diffs = np.abs(bay[mask] - simple[mask])
        median_diff = float(np.nanmedian(diffs)) if diffs.size > 0 else 0.0
        max_diff = float(np.nanmax(diffs)) if diffs.size > 0 else 0.0
        n_negative_parallax = int(np.sum(df["parallax"].to_numpy(dtype=float) <= 0))
        recovered = int(np.sum((df["parallax"].to_numpy(dtype=float) <= 0) & np.isfinite(bay)))
        distance_comparison = {
            "median_diff_pc": median_diff,
            "max_diff_pc": max_diff,
            "n_negative_parallax": n_negative_parallax,
            "n_recovered_by_bayesian": recovered,
        }

    return {
        "n_stars": int(len(df)),
        "teff": teff_stats,
        "M_G": mg_stats,
        "distance_pc": dist_stats,
        "luminosity_solar": lum_stats,
        "spectral_distribution": spectral_distribution,
        "distance_comparison": distance_comparison,
    }

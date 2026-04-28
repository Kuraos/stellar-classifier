"""Generacion del diagrama Hertzsprung-Russell (HR).

El grafico invierte los ejes por convencion astronomica y colorea los puntos
segun la temperatura efectiva.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

SPECTRAL_BOUNDS = [30000, 10000, 7500, 6000, 5200, 3700]


def plot_hr(df: pd.DataFrame, ax: Optional[plt.Axes] = None, use_bayesian: bool = False) -> plt.Figure:
    """Dibuja un diagrama HR usando T_eff y M_G.

    Si se pasa ax, dibuja sobre ese eje para permitir embebido en Tkinter y
    devuelve la figura asociada al grafico.
    """
    required_cols = {"teff", "M_G"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas para graficar HR: {sorted(missing)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    else:
        fig = ax.figure
        ax.clear()

    # Seleccionar la columna de magnitud absoluta a usar
    mg_col = "M_G_bayesian" if use_bayesian and "M_G_bayesian" in df.columns else "M_G"

    scatter = ax.scatter(
        df["teff"],
        df[mg_col],
        c=df["teff"],
        cmap="RdYlBu_r",
        s=12,
        alpha=0.8,
        linewidths=0,
    )

    for bound in SPECTRAL_BOUNDS:
        ax.axvline(bound, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlabel("T_eff [K]")
    ax.set_ylabel("M_G [mag]")
    ax.set_title("Diagrama Hertzsprung-Russell")
    ax.grid(alpha=0.25)

    # Convencion astronomica: temperatura decrece hacia la derecha.
    ax.invert_xaxis()
    # Magnitud absoluta creciente hacia abajo.
    ax.invert_yaxis()

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("T_eff [K]")

    fig.tight_layout()
    return fig

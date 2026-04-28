"""Generacion del diagrama Hertzsprung-Russell (HR).

El grafico invierte los ejes por convencion astronomica y colorea los puntos
segun la temperatura efectiva.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.isochrones import filter_evolutionary_phases, isochrone_to_observables

SPECTRAL_BOUNDS = [30000, 10000, 7500, 6000, 5200, 3700]


def plot_hr(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    use_corrected: bool = False,
    use_bayesian: bool = False,
    isochrones_to_overlay: list[dict] | None = None,
) -> plt.Figure:
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

    previous_colorbar = getattr(fig, "_hr_colorbar", None)
    if previous_colorbar is not None:
        try:
            previous_colorbar.ax.remove()
        except Exception:
            try:
                previous_colorbar.remove()
            except Exception:
                pass
        fig._hr_colorbar = None

    # Seleccionar columnas activas según los toggles disponibles.
    teff_col = "teff"
    mg_col = "M_G"
    if use_corrected and {"teff_corr", "M_G_corr"}.issubset(df.columns):
        teff_col = "teff_corr"
        mg_col = "M_G_corr"
    elif use_bayesian and "M_G_bayesian" in df.columns:
        mg_col = "M_G_bayesian"

    scatter = ax.scatter(
        df[teff_col],
        df[mg_col],
        c=df[teff_col],
        cmap="RdYlBu_r",
        s=12,
        alpha=0.8,
        linewidths=0,
        picker=5,
    )
    fig._hr_scatter = scatter

    for bound in SPECTRAL_BOUNDS:
        ax.axvline(bound, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlabel("T_eff [K]")
    ax.set_ylabel("M_G [mag]")
    ax.set_title("Diagrama Hertzsprung-Russell")
    ax.grid(alpha=0.25)

    if isochrones_to_overlay:
        for overlay in isochrones_to_overlay:
            iso = overlay.get("isochrone")
            if iso is None or not isinstance(iso, pd.DataFrame) or iso.empty:
                continue
            color = overlay.get("color", "tab:red")
            label = overlay.get("label", f"log_age={overlay.get('log_age', '?')}")

            filtered = filter_evolutionary_phases(iso)
            observables = isochrone_to_observables(filtered)
            ax.plot(
                observables["log_T_eff"],
                observables["M_G_iso"],
                color=color,
                linewidth=1.8,
                label=label,
                alpha=0.95,
            )

        if any(overlay.get("label") for overlay in isochrones_to_overlay):
            ax.legend(loc="upper left", fontsize=8, frameon=True)

    # Convencion astronomica: temperatura decrece hacia la derecha.
    ax.invert_xaxis()
    # Ajustar limites del eje x para recortar espacio vacío a la izquierda
    # (se hace DESPUÉS de invertir para evitar conflictos con gráficas repetidas)
    try:
        teff_vals = df[teff_col].dropna()
        if not teff_vals.empty:
            tmin = float(teff_vals.min())
            tmax = float(teff_vals.max())
            trange = max(tmax - tmin, 1.0)
            pad = trange * 0.05
            # Para eje invertido: xlim(max, min)
            ax.set_xlim(tmax + pad, tmin - pad)
    except Exception:
        pass
    # Magnitud absoluta creciente hacia abajo.
    ax.invert_yaxis()

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("T_eff [K]")
    fig._hr_colorbar = colorbar

    fig.tight_layout()
    return fig

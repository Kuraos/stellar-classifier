"""Generacion del diagrama Hertzsprung-Russell (HR).

El grafico invierte los ejes por convencion astronomica y colorea los puntos
segun la temperatura efectiva.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes

from src.isochrones import filter_evolutionary_phases, isochrone_to_observables
from src.variables import VARIABLE_LABELS, VARIABLE_PLOT_STYLE

SPECTRAL_BOUNDS: list[int] = [30000, 10000, 7500, 6000, 5200, 3700]


def plot_hr(
    df: pd.DataFrame,
    ax: Optional["matplotlib.axes.Axes"] = None,
    use_corrected: bool = False,
    use_bayesian: bool = False,
    isochrones_to_overlay: list[dict[str, object]] | None = None,
    highlight_variables: bool = False,
    variable_types_to_show: set[str] | None = None,
) -> "matplotlib.figure.Figure":
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
        fig = cast("matplotlib.figure.Figure", ax.figure)
        ax.clear()

    fig_state = cast(Any, fig)
    previous_colorbar = getattr(fig_state, "_hr_colorbar", None)
    if previous_colorbar is not None:
        try:
            previous_colorbar.ax.remove()
        except Exception:
            try:
                previous_colorbar.remove()
            except Exception:
                pass
        fig_state._hr_colorbar = None

    # Seleccionar columnas activas según los toggles disponibles.
    teff_col = "teff"
    mg_col = "M_G"
    if use_corrected and {"teff_corr", "M_G_corr"}.issubset(df.columns):
        teff_col = "teff_corr"
        mg_col = "M_G_corr"
    elif use_bayesian and "M_G_bayesian" in df.columns:
        mg_col = "M_G_bayesian"

    scatter = None
    if highlight_variables and "variable_type" in df.columns and not df.empty:
        variable_series = df["variable_type"].fillna("non_variable").astype(str).str.strip()
        normalized_types = variable_series.str.upper()
        mask_variable = ~normalized_types.isin({"NON_VARIABLE", "", "NAN", "NONE"})

        if mask_variable.any():
            df_non_var = df.loc[~mask_variable]
            df_var = df.loc[mask_variable]
            visible_types = None if variable_types_to_show is None else set(variable_types_to_show)

            if not df_non_var.empty:
                scatter = ax.scatter(
                    df_non_var[teff_col],
                    df_non_var[mg_col],
                    c=df_non_var[teff_col],
                    cmap="RdYlBu_r",
                    s=8,
                    alpha=0.25,
                    linewidths=0,
                    picker=5,
                    label="_nolegend_",
                )
            else:
                scatter = ax.scatter(
                    df[teff_col],
                    df[mg_col],
                    c=df[teff_col],
                    cmap="RdYlBu_r",
                    s=1,
                    alpha=0.0,
                    linewidths=0,
                    picker=5,
                    label="_nolegend_",
                )

            visible_variable_types: list[str] = []
            variable_subset_types = normalized_types.loc[df_var.index]
            for variable_type, style in VARIABLE_PLOT_STYLE.items():
                if visible_types is not None and variable_type not in visible_types:
                    continue

                subset = df_var.loc[variable_subset_types.eq(variable_type)]
                if subset.empty:
                    continue

                marker = str(style.get("marker", "o"))
                color = str(style.get("color", "gray"))
                size_raw = style.get("size", 40.0)
                if isinstance(size_raw, (int, float)):
                    size = float(size_raw)
                elif isinstance(size_raw, str):
                    try:
                        size = float(size_raw)
                    except Exception:
                        size = 40.0
                else:
                    size = 40.0

                zorder_raw = style.get("zorder", 5)
                if isinstance(zorder_raw, (int, float)):
                    zorder = int(zorder_raw)
                elif isinstance(zorder_raw, str):
                    try:
                        zorder = int(zorder_raw)
                    except Exception:
                        zorder = 5
                else:
                    zorder = 5

                ax.scatter(
                    subset[teff_col],
                    subset[mg_col],
                    marker=marker,
                    color=color,
                    s=size,
                    zorder=zorder,
                    linewidths=0.5,
                    label=VARIABLE_LABELS.get(variable_type, variable_type),
                )
                visible_variable_types.append(variable_type)

            if visible_variable_types:
                ax.legend(loc="upper left", fontsize=8)
        else:
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
    else:
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

    fig_state._hr_scatter = scatter

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
            color = str(overlay.get("color", "tab:red"))
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

    if highlight_variables and any(label for label in VARIABLE_LABELS):
        handles, labels = ax.get_legend_handles_labels()
        visible_labels = [label for label in labels if label and not label.startswith("_")]
        if visible_labels:
            ax.legend(loc="upper left", fontsize=8)

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
    fig_state._hr_colorbar = colorbar

    fig.tight_layout()
    return fig

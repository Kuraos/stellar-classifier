"""Widgets de graficacion matplotlib embebidos en Tkinter."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd


class MatplotlibPanel(ttk.Frame):
    """Panel reutilizable para embebido de matplotlib en Tkinter."""

    def __init__(self, master: tk.Misc, width: float = 6.5, height: float = 5.5, dpi: int = 100):
        super().__init__(master)

        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        self.current_df: pd.DataFrame | None = None
        self.current_scatter = None
        self._view_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.mode_var = tk.StringVar(value="Modo: bruto")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        controls = ttk.Frame(self)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(2, 4))

        self.reset_btn = ttk.Button(controls, text="Restablecer zoom", command=self.reset_view)
        self.reset_btn.pack(side=tk.LEFT)

        ttk.Label(controls, textvariable=self.mode_var, foreground="dimgray").pack(side=tk.RIGHT)

        self.canvas.mpl_connect("pick_event", self._on_pick_event)
        # Callback invoked when a point is selected: receives a dict with row data
        self.on_point_selected = None

        self.clear(message="Presiona 'Graficar' para mostrar el diagrama HR")

    def clear(self, message: str = "") -> None:
        """Limpia el eje actual y opcionalmente muestra un mensaje central."""
        self.ax.clear()
        self._view_limits = None
        if message:
            self.ax.text(
                0.5,
                0.5,
                message,
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="dimgray",
            )
            self.ax.set_axis_off()
        self.canvas.draw_idle()

    def set_point_context(self, df: pd.DataFrame | None, scatter=None) -> None:
        """Asocia el DataFrame y el scatter actual para inspeccion por clic."""
        self.current_df = df
        self.current_scatter = scatter

    def set_display_mode(self, mode: str) -> None:
        """Muestra el modo de visualizacion activo."""
        self.mode_var.set(f"Modo: {mode}")

    def capture_view_limits(self) -> None:
        """Guarda los limites visibles actuales para restaurarlos luego."""
        self._view_limits = (self.ax.get_xlim(), self.ax.get_ylim())

    def reset_view(self) -> None:
        """Restaura los limites guardados del grafico actual."""
        if self._view_limits is None:
            self.toolbar.home()
        else:
            xlim, ylim = self._view_limits
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.canvas.draw_idle()

    def _build_point_details(self, row: pd.Series) -> str:
        """Construye un resumen corto para inspeccion de un punto."""
        fields = [
            ("source_id", "Source ID", "{}"),
            ("ra", "RA", "{:.4f}"),
            ("dec", "DEC", "{:.4f}"),
            ("bp_rp", "BP-RP", "{:.3f}"),
            ("teff", "T_eff", "{:.0f} K"),
            ("M_G", "M_G", "{:.2f}"),
            ("distance_display", "Distancia", "{:.2f} pc"),
            ("distance_pc", "Distancia", "{:.2f} pc"),
        ]
        lines: list[str] = []
        for column, label, fmt in fields:
            value = row.get(column)
            if value is None or pd.isna(value):
                continue
            try:
                lines.append(f"{label}: {fmt.format(float(value))}")
            except Exception:
                lines.append(f"{label}: {value}")
        return "\n".join(lines) if lines else "No hay datos disponibles para este punto."

    def _on_pick_event(self, event) -> None:
        """Muestra un detalle breve al hacer clic sobre un punto del HR."""
        if self.current_df is None or self.current_scatter is None:
            return
        if event.artist is not self.current_scatter:
            return
        indices = getattr(event, "ind", None)
        if not indices:
            return

        index = int(indices[0])
        if index < 0 or index >= len(self.current_df):
            return

        row = self.current_df.iloc[index]
        # Keep the popup for compatibility, and emit structured details to callback
        details = self._build_point_details(row)
        try:
            messagebox.showinfo("Detalle del punto", details)
        finally:
            if callable(self.on_point_selected):
                try:
                    # pass a plain dict for easier consumption
                    self.on_point_selected(row.to_dict())
                except Exception:
                    pass

    def draw(self) -> None:
        """Fuerza repintado del canvas."""
        self.canvas.draw_idle()

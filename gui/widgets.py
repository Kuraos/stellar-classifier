"""Componentes reutilizables para la interfaz de stellar-classifier."""

from __future__ import annotations

from datetime import datetime
import tkinter as tk
from tkinter import ttk

import pandas as pd


class StatisticsPanel(ttk.LabelFrame):
    """Panel con estadisticas generales y distribucion espectral."""

    def __init__(self, master: tk.Misc):
        super().__init__(master, text="Estadisticas")

        style = ttk.Style(self)
        style.configure("Mono.TLabel", font=("Consolas", 10))

        self.general_frame = ttk.LabelFrame(self, text="Estadisticas generales")
        self.general_frame.pack(fill=tk.X, padx=8, pady=8)

        self.spectral_frame = ttk.LabelFrame(self, text="Distribucion espectral")
        self.spectral_frame.pack(fill=tk.X, padx=8, pady=8)

        self.general_vars = {
            "n_stars": tk.StringVar(value="N estrellas:      -"),
            "teff_mean": tk.StringVar(value="T_eff media:     -"),
            "teff_median": tk.StringVar(value="T_eff mediana:   -"),
            "distance_mean": tk.StringVar(value="Distancia media: -"),
            "luminosity_mean": tk.StringVar(value="Luminosidad med: -"),
        }

        for var in self.general_vars.values():
            ttk.Label(self.general_frame, textvariable=var, style="Mono.TLabel").pack(
                anchor="w", padx=8, pady=2
            )

        self.spectral_vars = {
            "O": tk.StringVar(value="O: 0"),
            "B": tk.StringVar(value="B: 0"),
            "A": tk.StringVar(value="A: 0"),
            "F": tk.StringVar(value="F: 0"),
            "G": tk.StringVar(value="G: 0"),
            "K": tk.StringVar(value="K: 0"),
            "M": tk.StringVar(value="M: 0"),
        }

        row1 = ttk.Frame(self.spectral_frame)
        row1.pack(fill=tk.X, padx=8, pady=2)
        row2 = ttk.Frame(self.spectral_frame)
        row2.pack(fill=tk.X, padx=8, pady=2)
        row3 = ttk.Frame(self.spectral_frame)
        row3.pack(fill=tk.X, padx=8, pady=2)

        ttk.Label(row1, textvariable=self.spectral_vars["O"], style="Mono.TLabel").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(row1, textvariable=self.spectral_vars["A"], style="Mono.TLabel").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(row1, textvariable=self.spectral_vars["G"], style="Mono.TLabel").pack(side=tk.LEFT)

        ttk.Label(row2, textvariable=self.spectral_vars["B"], style="Mono.TLabel").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(row2, textvariable=self.spectral_vars["F"], style="Mono.TLabel").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(row2, textvariable=self.spectral_vars["K"], style="Mono.TLabel").pack(side=tk.LEFT)

        ttk.Label(row3, textvariable=self.spectral_vars["M"], style="Mono.TLabel").pack(side=tk.LEFT)

    def clear(self) -> None:
        """Reinicia valores del panel a estado vacio."""
        self.general_vars["n_stars"].set("N estrellas:      -")
        self.general_vars["teff_mean"].set("T_eff media:     -")
        self.general_vars["teff_median"].set("T_eff mediana:   -")
        self.general_vars["distance_mean"].set("Distancia media: -")
        self.general_vars["luminosity_mean"].set("Luminosidad med: -")
        for key in self.spectral_vars:
            self.spectral_vars[key].set(f"{key}: 0")

    def update_from_stats(self, stats: dict) -> None:
        """Actualiza labels a partir del diccionario de metricas."""
        self.general_vars["n_stars"].set(f"N estrellas:      {stats['n_stars']:>6d}")
        self.general_vars["teff_mean"].set(f"T_eff media:     {stats['teff']['mean']:>8.1f} K")
        self.general_vars["teff_median"].set(f"T_eff mediana:   {stats['teff']['median']:>8.1f} K")
        self.general_vars["distance_mean"].set(
            f"Distancia media: {stats['distance_pc']['mean']:>8.2f} pc"
        )
        self.general_vars["luminosity_mean"].set(
            f"Luminosidad med: {stats['luminosity_solar']['mean']:>8.3f} L_sun"
        )

        distribution = stats.get("spectral_distribution", {})
        for key in self.spectral_vars:
            self.spectral_vars[key].set(f"{key}: {int(distribution.get(key, 0))}")


class DataTable(ttk.Frame):
    """Tabla de datos con ordenacion por click en encabezados."""

    def __init__(self, master: tk.Misc, columns: list[tuple[str, str, int]]):
        super().__init__(master)
        self.columns = columns
        self.current_df = pd.DataFrame()
        self.sort_states: dict[str, bool] = {}

        keys = [key for key, _, _ in columns]
        self.tree = ttk.Treeview(self, columns=keys, show="headings", height=10)

        for key, title, width in columns:
            self.tree.heading(key, text=title, command=lambda c=key: self._sort_by_column(c))
            self.tree.column(key, width=width, anchor="center", stretch=True)

        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _format_value(self, value: object, column: str) -> str:
        if value is None:
            return ""
        if pd.isna(value):
            return ""
        if column in {"ra", "dec"}:
            return f"{float(value):.4f}"
        if column in {"bp_rp", "B_V", "M_G", "luminosity_solar"}:
            return f"{float(value):.3f}"
        if column in {"teff", "distance_pc"}:
            return f"{float(value):.2f}"
        return str(value)

    def _populate_rows(self, df: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        col_keys = [key for key, _, _ in self.columns]
        for _, row in df.iterrows():
            values = [self._format_value(row.get(col), col) for col in col_keys]
            self.tree.insert("", tk.END, values=values)

    def _sort_by_column(self, column: str) -> None:
        if self.current_df.empty:
            return

        ascending = self.sort_states.get(column, True)
        sorted_df = self.current_df.sort_values(
            by=column,
            ascending=ascending,
            kind="mergesort",
            na_position="last",
        )
        self.sort_states[column] = not ascending
        self.current_df = sorted_df.reset_index(drop=True)
        self._populate_rows(self.current_df)

    def set_dataframe(self, df: pd.DataFrame, max_rows: int = 500) -> tuple[int, int]:
        """Carga un DataFrame y devuelve (filas_mostradas, total_filas)."""
        total = len(df)
        shown_df = df.head(max_rows).copy()
        self.current_df = shown_df.reset_index(drop=True)
        self.sort_states.clear()
        self._populate_rows(self.current_df)
        return len(self.current_df), total


class StatusBar(ttk.Frame):
    """Barra de estado inferior con timestamp de ultima actualizacion."""

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.status_var = tk.StringVar(value="Estado: listo")
        self.time_var = tk.StringVar(value="Ultima actualizacion: -")

        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.time_label = ttk.Label(self, textvariable=self.time_var)

        self.status_label.pack(side=tk.LEFT, padx=8, pady=4)
        self.time_label.pack(side=tk.RIGHT, padx=8, pady=4)

    def set_status(self, message: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_var.set(f"Estado: {message}")
        self.time_var.set(f"Ultima actualizacion: {now}")

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
            "distance_rescued": tk.StringVar(value="Rescatadas BJ:   -"),
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
        self.general_vars["distance_rescued"].set("Rescatadas BJ:   -")
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

        comparison = stats.get("distance_comparison") or {}
        if comparison:
            self.general_vars["distance_rescued"].set(
                f"Rescatadas BJ: {int(comparison.get('n_recovered_by_bayesian', 0)):>6d}"
            )
        else:
            self.general_vars["distance_rescued"].set("Rescatadas BJ:   -")

        distribution = stats.get("spectral_distribution", {})
        for key in self.spectral_vars:
            self.spectral_vars[key].set(f"{key}: {int(distribution.get(key, 0))}")


class IsochronePanel(ttk.LabelFrame):
    """Panel para elegir, sobreponer y ajustar isócronas PARSEC."""

    def __init__(self, parent: tk.Misc, on_overlay, on_clear, on_fit_age):
        super().__init__(parent, text="Isócronas PARSEC")
        self.on_overlay = on_overlay
        self.on_clear = on_clear
        self.on_fit_age = on_fit_age
        self.available_isochrones: list[dict] = []
        self._label_to_isochrone: dict[str, dict] = {}

        self.selected_var = tk.StringVar(value="Cargando isócronas...")
        self.status_var = tk.StringVar(value="Buscando archivos en data/isochrones/")

        self.columnconfigure(0, weight=1)

        self.combo = ttk.Combobox(self, textvariable=self.selected_var, state="disabled")
        self.combo.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 4))

        self.overlay_btn = ttk.Button(self, text="Sobreponer isócrona", command=self._handle_overlay, state="disabled")
        self.overlay_btn.grid(row=1, column=0, sticky="ew", padx=(8, 4), pady=4)

        self.clear_btn = ttk.Button(self, text="Limpiar isócronas", command=self._handle_clear, state="disabled")
        self.clear_btn.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        self.fit_btn = ttk.Button(self, text="Ajustar edad", command=self._handle_fit_age, state="disabled")
        self.fit_btn.grid(row=1, column=2, sticky="ew", padx=(4, 8), pady=4)

        ttk.Label(self, textvariable=self.status_var, wraplength=320, justify="left").grid(
            row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(2, 8)
        )

    def set_loading(self) -> None:
        """Muestra el estado de carga inicial."""
        self.selected_var.set("Cargando isócronas...")
        self.status_var.set("Leyendo archivos CMD 3.7 desde data/isochrones/")
        self.combo.configure(values=[], state="disabled")
        self.overlay_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.fit_btn.configure(state="disabled")

    def set_isochrones(self, isochrones: list[dict]) -> None:
        """Carga las isócronas disponibles en el combo y habilita acciones."""
        self.available_isochrones = list(isochrones)
        self._label_to_isochrone = {
            item["label_humano"]: item for item in self.available_isochrones
        }

        if not self.available_isochrones:
            self.selected_var.set("No hay isócronas disponibles")
            self.status_var.set("Coloca archivos PARSEC en data/isochrones/")
            self.combo.configure(values=[], state="disabled")
            self.overlay_btn.configure(state="disabled")
            self.clear_btn.configure(state="disabled")
            self.fit_btn.configure(state="disabled")
            return

        values = [item["label_humano"] for item in self.available_isochrones]
        self.combo.configure(values=values, state="readonly")
        self.selected_var.set(values[0])
        self.status_var.set(f"{len(values)} isócronas disponibles")
        self.overlay_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        self.fit_btn.configure(state="normal")

    def set_status(self, message: str) -> None:
        """Actualiza el mensaje descriptivo del panel."""
        self.status_var.set(message)

    def set_enabled(self, enabled: bool) -> None:
        """Activa o desactiva los controles del panel."""
        state = "normal" if enabled else "disabled"
        combo_state = "readonly" if enabled and self.available_isochrones else "disabled"
        self.combo.configure(state=combo_state)
        self.overlay_btn.configure(state=state if enabled and self.available_isochrones else "disabled")
        self.clear_btn.configure(state=state if enabled and self.available_isochrones else "disabled")
        self.fit_btn.configure(state=state if enabled and self.available_isochrones else "disabled")

    def get_selected_isochrone(self) -> dict | None:
        """Devuelve el metadato de la isócrona seleccionada."""
        label = self.selected_var.get()
        return self._label_to_isochrone.get(label)

    def _handle_overlay(self) -> None:
        selected = self.get_selected_isochrone()
        if selected is not None:
            self.on_overlay(selected)

    def _handle_clear(self) -> None:
        self.on_clear()

    def _handle_fit_age(self) -> None:
        self.on_fit_age()


class VariablesPanel(ttk.LabelFrame):
    """Panel para filtrar y validar estrellas variables.

    Provee checkboxes por grupos y un boton para lanzar la validacion P-L.
    """

    def __init__(self, parent: tk.Misc, on_validate):
        super().__init__(parent, text="Estrellas variables")
        self.on_validate = on_validate
        self.status_var = tk.StringVar(value="No hay columnas de variabilidad")

        self.columnconfigure(0, weight=1)

        # Master toggle para mostrar variables en el HR
        self.show_vars_var = tk.BooleanVar(value=False)
        self.chk_show = ttk.Checkbutton(self, text="Mostrar variables en HR", variable=self.show_vars_var)
        self.chk_show.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

        # Simple checkboxes por tipo (colapsado en primera iteracion)
        types_frame = ttk.Frame(self)
        types_frame.grid(row=1, column=0, sticky="ew", padx=8)
        self.type_vars = {}
        for i, label in enumerate(["DCEP", "RRAB", "RRC", "MIRA", "ECL", "OTHER"]):
            v = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(types_frame, text=label, variable=v)
            cb.grid(row=0, column=i, sticky="w", padx=(0, 6))
            self.type_vars[label] = v

        self.validate_btn = ttk.Button(self, text="Validar P-L", command=self._handle_validate, state="disabled")
        self.validate_btn.grid(row=2, column=0, sticky="ew", padx=8, pady=8)

        ttk.Label(self, textvariable=self.status_var, wraplength=320, justify="left").grid(
            row=3, column=0, sticky="ew", padx=8, pady=(2, 8)
        )

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        combo_state = state
        self.chk_show.configure(state=state)
        for v in self.type_vars.values():
            # toggle the underlying widget state
            # tkinter BooleanVar remains usable; we just enable/disable the button
            pass
        self.validate_btn.configure(state=state)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _handle_validate(self) -> None:
        self.on_validate()


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
        if column in {"teff", "distance_pc", "distance_display"}:
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

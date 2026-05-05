"""Componentes reutilizables para la interfaz de stellar-classifier."""

from __future__ import annotations

from datetime import datetime
from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.lamost import SPECTRAL_LINES as SPECTRAL_LINES_GUI
from src.variables import VARIABLE_LABELS


def _safe_isfinite(value: object) -> bool:
    """Devuelve True solo para floats finitos; tolera None y strings."""
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


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

        self.variables_frame = ttk.LabelFrame(self, text="Variables detectadas")
        self.variables_frame.pack(fill=tk.X, padx=8, pady=8)
        self.variables_frame.pack_forget()

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

        self.var_summary_var = tk.StringVar(value="")
        self.var_count_label = ttk.Label(
            self.variables_frame,
            textvariable=self.var_summary_var,
            style="Mono.TLabel",
            wraplength=260,
            justify="left",
        )
        self.var_count_label.pack(anchor="w", padx=8, pady=4)

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
        self.var_summary_var.set("")
        self.variables_frame.pack_forget()

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

        variability = stats.get("variability")
        if variability and variability.get("n_variables", 0) > 0:
            counts = variability.get("counts_by_type", {})
            n_vars = int(variability.get("n_variables", 0))
            n_total = int(stats.get("n_stars", 0))
            fraction = 100.0 * n_vars / n_total if n_total > 0 else 0.0
            n_pl = int(variability.get("n_with_pl_distance", 0))

            lines = [f"Total: {n_vars} ({fraction:.1f}%)"]
            for tipo, count in counts.items():
                tipo_str = str(tipo)
                if tipo_str.lower() in {"non_variable", "nan", "none", ""}:
                    continue
                if int(count) <= 0:
                    continue
                label = VARIABLE_LABELS.get(tipo_str, tipo_str)
                lines.append(f"  {label}: {int(count)}")
            if n_pl > 0:
                lines.append(f"Con dist. P-L: {n_pl}")

            self.var_summary_var.set("\n".join(lines))
            self.variables_frame.pack(fill=tk.X, padx=8, pady=8)
        else:
            self.var_summary_var.set("")
            self.variables_frame.pack_forget()


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
        has_isochrones = bool(self.available_isochrones)
        combo_state = "readonly" if (enabled and has_isochrones) else "disabled"
        btn_state = "normal" if (enabled and has_isochrones) else "disabled"
        self.combo.configure(state=combo_state)
        self.overlay_btn.configure(state=btn_state)
        self.clear_btn.configure(state=btn_state)
        self.fit_btn.configure(state=btn_state)

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

    def __init__(
        self,
        parent: tk.Misc,
        on_validate: Callable[[], None],
        on_filter_change: Callable[[set[str] | None], None] | None = None,
    ):
        super().__init__(parent, text="Estrellas variables")
        self.on_validate = on_validate
        self.on_filter_change = on_filter_change
        self.status_var = tk.StringVar(value="No hay columnas de variabilidad")

        self.columnconfigure(0, weight=1)

        # Master toggle para mostrar variables en el HR
        self.show_vars_var = tk.BooleanVar(value=False)
        self.chk_show = ttk.Checkbutton(
            self,
            text="Mostrar variables en HR",
            variable=self.show_vars_var,
            command=self._on_show_vars_changed,
        )
        self.chk_show.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

        # Simple checkboxes por tipo (colapsado en primera iteracion)
        types_frame = ttk.Frame(self)
        types_frame.grid(row=1, column=0, sticky="ew", padx=8)
        self.type_vars: dict[str, tk.BooleanVar] = {}
        self.type_checks: dict[str, ttk.Checkbutton] = {}
        for i, label in enumerate(["DCEP", "RRAB", "RRC", "MIRA", "ECL", "OTHER"]):
            v = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(
                types_frame,
                text=label,
                variable=v,
                command=self._on_type_filter_changed,
            )
            cb.grid(row=0, column=i, sticky="w", padx=(0, 6))
            self.type_vars[label] = v
            self.type_checks[label] = cb

        self.validate_btn = ttk.Button(self, text="Validar P-L", command=self._handle_validate, state="disabled")
        self.validate_btn.grid(row=2, column=0, sticky="ew", padx=8, pady=8)

        ttk.Label(self, textvariable=self.status_var, wraplength=320, justify="left").grid(
            row=3, column=0, sticky="ew", padx=8, pady=(2, 8)
        )

    def set_enabled(self, enabled: bool) -> None:
        """Activa o desactiva todos los controles del panel."""
        state = "normal" if enabled else "disabled"
        self.chk_show.configure(state=state)
        for cb in self.type_checks.values():
            cb.configure(state=state)
        self.validate_btn.configure(state=state)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def get_active_types(self) -> set[str]:
        """Devuelve el conjunto de tipos activos según los checkboxes."""
        return {
            label
            for label, var in self.type_vars.items()
            if var.get()
        }

    def get_show_variables(self) -> bool:
        """Devuelve True si el toggle principal está activo."""
        return self.show_vars_var.get()

    def _on_type_filter_changed(self) -> None:
        """Notifica al app cuando cambia el filtro de tipos."""
        if self.on_filter_change is not None:
            self.on_filter_change(self.get_active_types())

    def _on_show_vars_changed(self) -> None:
        """Notifica al app cuando cambia el toggle principal."""
        if self.on_filter_change is not None:
            if self.show_vars_var.get():
                self.on_filter_change(self.get_active_types())
            else:
                self.on_filter_change(None)

    def _handle_validate(self) -> None:
        self.on_validate()


class DetailPanel(ttk.LabelFrame):
    """Panel lateral para mostrar detalles de una estrella seleccionada."""

    def __init__(self, parent: tk.Misc):
        super().__init__(parent, text="Detalle de estrella")
        self.columnconfigure(0, weight=1)
        self.text = tk.Text(self, height=8, wrap="word")
        self.text.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.clear()

    def set_details(self, info: dict) -> None:
        """Rellena el panel con un diccionario de campos.

        Espera un diccionario simple {col: value}.
        """
        self.text.config(state="normal")
        self.text.delete("1.0", tk.END)
        if not info:
            self.text.insert(tk.END, "Sin selección")
        else:
            for key, val in info.items():
                self.text.insert(tk.END, f"{key}: {val}\n")
        self.text.config(state="disabled")

    def clear(self) -> None:
        self.text.config(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "Selecciona un punto en el diagrama HR para ver detalles.")
        self.text.config(state="disabled")


class SpectroscopyPanel(ttk.Frame):
    """Panel de espectroscopia con visor, tabla de lineas y estado."""

    def __init__(
        self,
        master: tk.Misc,
        on_crossmatch: Callable[[], None],
        on_batch_analyse: Callable[[], None],
        on_prev_spectrum: Callable[[], None] | None = None,
        on_next_spectrum: Callable[[], None] | None = None,
        on_focus_hr: Callable[[], None] | None = None,
    ):
        super().__init__(master)
        self.on_crossmatch = on_crossmatch
        self.on_batch_analyse = on_batch_analyse
        self.on_prev_spectrum = on_prev_spectrum
        self.on_next_spectrum = on_next_spectrum
        self.on_focus_hr = on_focus_hr
        self._crossmatch_df: pd.DataFrame | None = None

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        left = ttk.LabelFrame(self, text="Espectro")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(6, 3.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        right = ttk.LabelFrame(self, text="Lineas espectrales")
        right.grid(row=0, column=1, sticky="nsew", pady=(0, 6))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        cols = ("line", "lambda", "ew", "fitted")
        self.lines_tree = ttk.Treeview(right, columns=cols, show="headings", height=8)
        self.lines_tree.heading("line", text="Linea")
        self.lines_tree.heading("lambda", text="lambda (A)")
        self.lines_tree.heading("ew", text="EW (A)")
        self.lines_tree.heading("fitted", text="Ajustada")
        self.lines_tree.column("line", width=90, anchor="w")
        self.lines_tree.column("lambda", width=85, anchor="center")
        self.lines_tree.column("ew", width=85, anchor="center")
        self.lines_tree.column("fitted", width=80, anchor="center")
        self.lines_tree.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        summary = ttk.Frame(right)
        summary.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        summary.columnconfigure(0, weight=1)

        self.type_var = tk.StringVar(value="Tipo espectral (EW): -")
        self.teff_spec_var = tk.StringVar(value="T_eff espectrosc.: -")
        self.teff_phot_var = tk.StringVar(value="T_eff fotometrica: -")
        self.teff_diff_var = tk.StringVar(value="Diferencia: -")
        self.obsid_var = tk.StringVar(value="LAMOST obsid: -")
        self.snr_var = tk.StringVar(value="S/N (G band): -")
        self.class_var = tk.StringVar(value="Clase LAMOST: -")

        ttk.Label(summary, textvariable=self.type_var).grid(row=0, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.teff_spec_var).grid(row=1, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.teff_phot_var).grid(row=2, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.teff_diff_var).grid(row=3, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.obsid_var).grid(row=4, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.snr_var).grid(row=5, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.class_var).grid(row=6, column=0, sticky="w")

        bottom = ttk.Frame(self)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        bottom.columnconfigure(2, weight=1)

        self.btn_crossmatch = ttk.Button(bottom, text="Buscar espectros LAMOST", command=self.on_crossmatch)
        self.btn_crossmatch.grid(row=0, column=0, padx=(0, 8), pady=4)

        self.btn_batch = ttk.Button(bottom, text="Analizar muestra", command=self.on_batch_analyse)
        self.btn_batch.grid(row=0, column=1, padx=(0, 12), pady=4)

        self.btn_prev = ttk.Button(bottom, text="Anterior", command=self._handle_prev, state="disabled")
        self.btn_prev.grid(row=0, column=2, padx=(0, 6), pady=4)

        self.btn_next = ttk.Button(bottom, text="Siguiente", command=self._handle_next, state="disabled")
        self.btn_next.grid(row=0, column=3, padx=(0, 6), pady=4)

        self.btn_focus_hr = ttk.Button(bottom, text="Ir a estrella en HR", command=self._handle_focus_hr, state="disabled")
        self.btn_focus_hr.grid(row=0, column=4, padx=(0, 10), pady=4)

        self.nav_var = tk.StringVar(value="Espectros: 0/0")
        ttk.Label(bottom, textvariable=self.nav_var).grid(row=0, column=5, sticky="w", padx=(0, 12))

        self.status_var = tk.StringVar(value="Estado: listo")
        ttk.Label(bottom, textvariable=self.status_var).grid(row=0, column=6, sticky="w")

        self.clear_spectrum()

    def set_status(self, message: str) -> None:
        """Actualiza el mensaje de estado del panel."""
        self.status_var.set(f"Estado: {message}")

    def set_crossmatch_results(self, df: pd.DataFrame) -> None:
        """Guarda internamente resultados de cross-match."""
        self._crossmatch_df = df.copy() if df is not None else None

    def set_navigation_state(self, index: int | None, total: int, has_selection: bool) -> None:
        """Actualiza controles de navegacion para recorrer espectros."""
        if total <= 0 or index is None:
            self.nav_var.set("Espectros: 0/0")
            self.btn_prev.configure(state="disabled")
            self.btn_next.configure(state="disabled")
            self.btn_focus_hr.configure(state="disabled")
            return

        idx = int(index)
        self.nav_var.set(f"Espectros: {idx + 1}/{int(total)}")
        self.btn_prev.configure(state="normal" if idx > 0 else "disabled")
        self.btn_next.configure(state="normal" if idx < int(total) - 1 else "disabled")
        self.btn_focus_hr.configure(state="normal" if has_selection else "disabled")

    def get_crossmatch_df(self) -> pd.DataFrame | None:
        """Devuelve el DataFrame de cross-match almacenado."""
        return self._crossmatch_df

    def clear_spectrum(self) -> None:
        """Limpia grafico, tabla y resumen del panel espectroscopico."""
        self.ax.clear()
        self.ax.text(
            0.5,
            0.5,
            "Selecciona una estrella del HR con espectro LAMOST",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            color="dimgray",
        )
        self.ax.set_axis_off()
        self.canvas.draw_idle()

        self.lines_tree.delete(*self.lines_tree.get_children())
        self.type_var.set("Tipo espectral (EW): -")
        self.teff_spec_var.set("T_eff espectrosc.: -")
        self.teff_phot_var.set("T_eff fotometrica: -")
        self.teff_diff_var.set("Diferencia: -")
        self.obsid_var.set("LAMOST obsid: -")
        self.snr_var.set("S/N (G band): -")
        self.class_var.set("Clase LAMOST: -")

    def _handle_prev(self) -> None:
        if callable(self.on_prev_spectrum):
            self.on_prev_spectrum()

    def _handle_next(self) -> None:
        if callable(self.on_next_spectrum):
            self.on_next_spectrum()

    def _handle_focus_hr(self) -> None:
        if callable(self.on_focus_hr):
            self.on_focus_hr()

    def show_spectrum(self, result: dict) -> None:
        """Dibuja espectro, llena tabla EW y actualiza resumen textual."""
        if not result or not result.get("success"):
            self.clear_spectrum()
            err = result.get("error") if isinstance(result, dict) else "error desconocido"
            self.set_status(f"error: {err}")
            return

        wave = np.asarray(result.get("wavelength", []), dtype=float)
        flux = np.asarray(result.get("flux", []), dtype=float)
        ew_dict = result.get("equivalent_widths", {}) or {}

        self.ax.clear()
        self.ax.set_axis_on()
        self.ax.plot(wave, flux, color="black", linewidth=0.9, label="Flujo normalizado")
        self.ax.set_xlabel("Longitud de onda [A]")
        self.ax.set_ylabel("Flujo relativo")
        self.ax.set_title("Espectro LAMOST")
        self.ax.grid(alpha=0.25)

        line_colors = {
            "H_alpha": "red",
            "H_beta": "royalblue",
            "Ca_II_K": "green",
        }
        for name, wl in SPECTRAL_LINES_GUI.items():
            color = line_colors.get(name, "gray")
            self.ax.axvline(wl, color=color, linestyle="--", alpha=0.55, linewidth=0.9)

        self.canvas.draw_idle()

        self.lines_tree.delete(*self.lines_tree.get_children())
        for name, wl in SPECTRAL_LINES_GUI.items():
            item = ew_dict.get(name) or {}
            ew = item.get("EW", np.nan)
            fitted = bool(item.get("fitted", False))
            ew_str = f"{float(ew):.3f}" if _safe_isfinite(ew) else "NaN"
            self.lines_tree.insert(
                "",
                tk.END,
                values=(name, f"{wl:.1f}", ew_str, "si" if fitted else "no"),
            )

        spt = result.get("spectral_type_spec", "?")
        teff_spec = result.get("teff_spectroscopic")
        teff_phot = result.get("teff_photometric")
        teff_diff_k = result.get("teff_diff_K")
        teff_diff_pct = result.get("teff_diff_pct")

        self.type_var.set(f"Tipo espectral (EW): {spt}")
        if _safe_isfinite(teff_spec):
            self.teff_spec_var.set(f"T_eff espectrosc.: {float(teff_spec):.0f} K")
        else:
            self.teff_spec_var.set("T_eff espectrosc.: NaN")

        if _safe_isfinite(teff_phot):
            self.teff_phot_var.set(f"T_eff fotometrica: {float(teff_phot):.0f} K")
        else:
            self.teff_phot_var.set("T_eff fotometrica: -")

        if teff_diff_k is None or teff_diff_pct is None:
            self.teff_diff_var.set("Diferencia: -")
        else:
            self.teff_diff_var.set(f"Diferencia: {float(teff_diff_k):+.0f} K ({float(teff_diff_pct):+.1f}%)")

        obsid = result.get("obsid", "-")
        self.obsid_var.set(f"LAMOST obsid: {obsid}")

        snrg = result.get("snrg", np.nan)
        if _safe_isfinite(snrg):
            self.snr_var.set(f"S/N (G band): {float(snrg):.1f}")
        else:
            self.snr_var.set("S/N (G band): -")

        class_lamost = result.get("class_lamost")
        subclass_lamost = result.get("subclass_lamost")
        if class_lamost is None and subclass_lamost is None:
            self.class_var.set("Clase LAMOST: -")
        else:
            self.class_var.set(f"Clase LAMOST: {class_lamost or '-'} / {subclass_lamost or '-'}")


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

"""Aplicacion principal Tkinter para stellar-classifier.

La ventana coordina tres etapas: descarga desde Gaia, procesamiento fisico de
las estrellas y visualizacion en tabla, panel de estadisticas y diagrama HR.
"""

from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd

from data.download import query_gaia_sample
from src.extinction import apply_extinction_correction, prime_bayestar_cache
from src.distances import best_distance_bayesian, absolute_magnitude_bayesian
from gui.plots import MatplotlibPanel
from gui.widgets import DataTable, IsochronePanel, StatisticsPanel, StatusBar, VariablesPanel
from src.hr_diagram import plot_hr
from src.isochrones import fit_best_age, list_available_isochrones, load_isochrone
from src.statistics import compute_statistics
from src.temperature import (
    absolute_magnitude,
    bv_from_bprp,
    luminosity_solar,
    spectral_type,
    teff_from_bv,
)
from src.variables import add_variability_columns


ISOCHRONES_DIR = Path(__file__).resolve().parent.parent / "data" / "isochrones"


class StellarClassifierApp:
    """Ventana principal y coordinadora del flujo de trabajo interactivo."""

    def __init__(self, root: tk.Tk, preload_bayestar: bool = True):
        """Construye la ventana, el estado interno y todos los widgets."""
        self.root = root
        self.root.title("stellar-classifier")
        self.root.geometry("1200x800")
        self.root.minsize(1024, 720)

        # Estado interno compartido entre descarga, procesamiento, tabla y grafico.
        self.df_raw: pd.DataFrame | None = None
        self.df_processed: pd.DataFrame | None = None
        self.stats: dict | None = None
        self.fig = None
        self.ax = None

        self._download_thread: threading.Thread | None = None
        self._bayestar_preload_thread: threading.Thread | None = None
        self._isochrones_thread: threading.Thread | None = None
        self._bayestar_ready = not preload_bayestar
        self._bayestar_error: str | None = None
        self._isochrones_error: str | None = None
        self._preload_bayestar = preload_bayestar

        self.available_isochrones: list[dict] = []
        self.active_isochrones: list[dict] = []
        self._isochrone_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

        self.n_stars_var = tk.IntVar(value=5000)
        self.max_dist_var = tk.IntVar(value=100)
        self.extinction_var = tk.BooleanVar(value=False)
        self.bayesian_var = tk.BooleanVar(value=False)

        self._build_layout()

        if self._preload_bayestar:
            self._start_bayestar_preload()
        self._start_isochrones_load()

    def _build_layout(self) -> None:
        """Construye la disposicion general de la interfaz."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=3)
        self.root.rowconfigure(2, weight=2)

        self._build_action_bar()

        content_frame = ttk.Frame(self.root)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)
        content_frame.columnconfigure(0, weight=3)
        content_frame.columnconfigure(1, weight=2)
        content_frame.rowconfigure(0, weight=1)

        left_frame = ttk.LabelFrame(content_frame, text="Diagrama HR")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        self.plot_panel = MatplotlibPanel(left_frame)
        self.plot_panel.grid(row=0, column=0, sticky="nsew")
        self.fig = self.plot_panel.figure
        self.ax = self.plot_panel.ax

        right_frame = ttk.Frame(content_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=0)

        self.stats_panel = StatisticsPanel(right_frame)
        self.stats_panel.grid(row=0, column=0, sticky="nsew")

        self.isochrone_panel = IsochronePanel(
            right_frame,
            on_overlay=self._overlay_selected_isochrone,
            on_clear=self._clear_isochrones,
            on_fit_age=self._fit_best_age,
        )
        self.isochrone_panel.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.isochrone_panel.set_loading()

        self.variables_panel = VariablesPanel(right_frame, on_validate=self._validate_pl)
        self.variables_panel.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.variables_panel.set_status("Panel de variables: esperando descarga")

        table_frame = ttk.LabelFrame(self.root, text="Tabla de datos")
        table_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        table_columns = [
            ("source_id", "source_id", 160),
            ("ra", "ra", 90),
            ("dec", "dec", 90),
            ("distance_display", "Distancia [pc]", 100),
            ("bp_rp", "BP-RP", 80),
            ("B_V", "B-V", 80),
            ("teff", "T_eff [K]", 100),
            ("M_G", "M_G", 80),
            ("luminosity_solar", "L/L_sun", 90),
            ("spectral_type", "Tipo", 70),
        ]
        self.data_table = DataTable(table_frame, columns=table_columns)
        self.data_table.grid(row=0, column=0, sticky="nsew")

        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(row=3, column=0, sticky="ew")

    def _build_action_bar(self) -> None:
        """Crea la barra superior con acciones y parametros de consulta."""
        action_bar = ttk.Frame(self.root)
        action_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 2))
        action_bar.columnconfigure(99, weight=1)

        self.download_btn = ttk.Button(action_bar, text="Descargar datos", command=self._start_download)
        self.download_btn.grid(row=0, column=0, padx=(0, 6))

        self.process_btn = ttk.Button(action_bar, text="Procesar", command=self._process_data)
        self.process_btn.grid(row=0, column=1, padx=(0, 6))

        self.plot_btn = ttk.Button(action_bar, text="Graficar", command=self._plot_data)
        self.plot_btn.grid(row=0, column=2, padx=(0, 6))

        self.export_btn = ttk.Button(action_bar, text="Exportar CSV", command=self._export_csv)
        self.export_btn.grid(row=0, column=3, padx=(0, 12))

        self.extinction_check = ttk.Checkbutton(
            action_bar,
            text="Corregir extinción",
            variable=self.extinction_var,
        )
        self.extinction_check.grid(row=0, column=4, padx=(0, 12))

        self.bayesian_check = ttk.Checkbutton(
            action_bar,
            text="Distancias bayesianas",
            variable=self.bayesian_var,
        )
        self.bayesian_check.grid(row=0, column=5, padx=(0, 12))
        self.bayesian_check.configure(state="disabled")

        ttk.Label(action_bar, text="N:").grid(row=0, column=6, padx=(0, 4))
        self.n_stars_entry = ttk.Entry(action_bar, textvariable=self.n_stars_var, width=8)
        self.n_stars_entry.grid(row=0, column=7, padx=(0, 8))

        ttk.Label(action_bar, text="Max pc:").grid(row=0, column=8, padx=(0, 4))
        self.max_dist_entry = ttk.Entry(action_bar, textvariable=self.max_dist_var, width=8)
        self.max_dist_entry.grid(row=0, column=9)

    def _set_status(self, text: str) -> None:
        """Actualiza la barra inferior con un mensaje de estado."""
        self.status_bar.set_status(text)

    def _set_download_controls(self, enabled: bool) -> None:
        """Habilita o deshabilita los controles que afectan a la descarga."""
        state = "normal" if enabled else "disabled"
        self.download_btn.configure(state=state)
        self.n_stars_entry.configure(state=state)
        self.max_dist_entry.configure(state=state)

    def _start_bayestar_preload(self) -> None:
        """Inicia la carga en segundo plano del mapa Bayestar2019."""
        if self._bayestar_preload_thread and self._bayestar_preload_thread.is_alive():
            return

        self._bayestar_ready = False
        self._bayestar_error = None
        self._set_status("cargando mapa Bayestar2019 en segundo plano...")

        self._bayestar_preload_thread = threading.Thread(
            target=self._bayestar_preload_worker,
            daemon=True,
        )
        self._bayestar_preload_thread.start()
        self.root.after(0, self._poll_bayestar_preload_thread)

    def _start_isochrones_load(self) -> None:
        """Inicia la carga en segundo plano de la lista de isócronas."""
        if self._isochrones_thread and self._isochrones_thread.is_alive():
            return

        self._isochrones_error = None
        self.isochrone_panel.set_loading()
        self._isochrones_thread = threading.Thread(target=self._isochrones_loader_worker, daemon=True)
        self._isochrones_thread.start()
        self.root.after(0, self._poll_isochrones_thread)

    def _isochrones_loader_worker(self) -> None:
        """Carga la lista de isócronas fuera del hilo principal."""
        try:
            self.available_isochrones = list_available_isochrones(str(ISOCHRONES_DIR))
        except Exception as exc:
            self._isochrones_error = str(exc)

    def _poll_isochrones_thread(self) -> None:
        """Comprueba si terminó la carga de isócronas y actualiza la GUI."""
        if not self._isochrones_thread:
            return

        if self._isochrones_thread.is_alive():
            self.root.after(100, self._poll_isochrones_thread)
            return

        if self._isochrones_error is None:
            self._on_isochrones_loaded()
        else:
            self._on_isochrones_load_error(self._isochrones_error)

    def _on_isochrones_loaded(self) -> None:
        """Puebla el panel de isócronas cuando la carga termina bien."""
        if not self.available_isochrones:
            self.isochrone_panel.set_isochrones([])
            self.isochrone_panel.set_status("No hay isócronas en data/isochrones/")
            return

        self.isochrone_panel.set_isochrones(self.available_isochrones)
        self.isochrone_panel.set_status(f"{len(self.available_isochrones)} isócronas listas")

    def _on_isochrones_load_error(self, message: str) -> None:
        """Notifica un fallo leyendo isócronas sin romper la GUI."""
        self.available_isochrones = []
        self.isochrone_panel.set_isochrones([])
        self.isochrone_panel.set_status(f"Error al leer isócronas: {message}")

    def _refresh_hr_plot(self) -> None:
        """Redibuja el HR si ya hay datos procesados."""
        if self.df_processed is not None and not self.df_processed.empty:
            self._plot_data()

    def _overlay_selected_isochrone(self, selection: dict) -> None:
        """Carga la isócrona seleccionada y la añade a la superposición activa."""
        if not selection:
            return

        try:
            loaded = load_isochrone(
                selection["log_age"],
                metallicity=selection.get("metallicity", 0.0),
                isochrones_dir=str(ISOCHRONES_DIR),
            )
        except Exception as exc:
            messagebox.showerror("Error de isócrona", str(exc))
            return

        label = selection.get("label_humano", f"log_age={selection.get('log_age', '?')}")
        if any(item.get("label") == label for item in self.active_isochrones):
            self.isochrone_panel.set_status(f"Ya está sobrepuesta: {label}")
            self._refresh_hr_plot()
            return

        color = self._isochrone_colors[len(self.active_isochrones) % len(self._isochrone_colors)]
        self.active_isochrones.append(
            {
                "isochrone": loaded,
                "log_age": selection["log_age"],
                "color": color,
                "label": label,
            }
        )
        self.isochrone_panel.set_status(f"Isócrona añadida: {label}")
        self._refresh_hr_plot()

    def _clear_isochrones(self) -> None:
        """Limpia todas las isócronas activas y refresca el HR."""
        self.active_isochrones.clear()
        self.isochrone_panel.set_status("Isócronas limpiadas")
        self._refresh_hr_plot()

    def _fit_best_age(self) -> None:
        """Ajusta la mejor edad en segundo plano y sobrepone la isócrona ganadora."""
        if self.df_processed is None or self.df_processed.empty:
            self._set_status("error: primero procesa datos")
            return

        if self._isochrones_thread and self._isochrones_thread.is_alive():
            self._set_status("espera a que termine la carga de isócronas")
            return

        self.isochrone_panel.set_status("Ajustando edad en segundo plano...")

        def worker() -> None:
            try:
                result = fit_best_age(
                    self.df_processed,
                    age_grid=np.arange(7.0, 10.1, 0.1),
                    metallicity=0.0,
                    use_corrected=self.extinction_var.get(),
                    use_bayesian=self.bayesian_var.get(),
                    isochrones_dir=str(ISOCHRONES_DIR),
                )
                self.root.after(0, lambda: self._on_fit_best_age_success(result))
            except Exception as exc:
                self.root.after(0, lambda: self._on_fit_best_age_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_fit_best_age_success(self, result: dict) -> None:
        """Muestra el resultado del ajuste y sobrepone la mejor isócrona."""
        best_log_age = float(result["best_log_age"])
        best_label = str(result["best_age_label"])
        chi2 = float(result["min_chi2"])
        best_isochrone = result.get("best_isochrone")

        if isinstance(best_isochrone, pd.DataFrame) and not best_isochrone.empty:
            color = self._isochrone_colors[len(self.active_isochrones) % len(self._isochrone_colors)]
            label = f"Ajuste {best_label}"
            self.active_isochrones.append(
                {
                    "isochrone": best_isochrone,
                    "log_age": best_log_age,
                    "color": color,
                    "label": label,
                }
            )
            self._refresh_hr_plot()

        self.isochrone_panel.set_status(f"Edad mejor: {best_label} (chi2={chi2:.3f})")
        messagebox.showinfo(
            "Ajuste de edad",
            f"Edad mejor: {best_label}\nlog_age={best_log_age:.2f}\nchi2={chi2:.3f}",
        )

    def _on_fit_best_age_error(self, message: str) -> None:
        """Muestra un error si el ajuste de edad falla."""
        self.isochrone_panel.set_status(f"Error en ajuste: {message}")
        messagebox.showerror("Error de ajuste", message)

    def _validate_pl(self) -> None:
        """Valida relaciones P-L en background y muestra un resumen simple."""
        if self.df_processed is None or self.df_processed.empty:
            self._set_status("error: primero procesa datos")
            return

        self.variables_panel.set_status("Validando P-L en segundo plano...")

        def worker() -> None:
            df = self.df_processed
            # Preferir distance_pc_bayesian cuando esté activada
            dist_col = (
                "distance_pc_bayesian" if self.bayesian_var.get() and "distance_pc_bayesian" in df.columns else "distance_pc"
            )
            if "distance_pc_PL" not in df.columns:
                self.root.after(0, lambda: messagebox.showinfo("Validación P-L", "No hay distancias P-L calculadas para esta muestra."))
                self.root.after(0, lambda: self.variables_panel.set_status("Sin distancias P-L en la muestra"))
                return

            mask = np.isfinite(df["distance_pc_PL"].to_numpy(dtype=float)) & np.isfinite(df[dist_col].to_numpy(dtype=float))
            if not mask.any():
                self.root.after(0, lambda: messagebox.showinfo("Validación P-L", "No hay objetos con ambas distancias (P-L y geom/bayes) finitas."))
                self.root.after(0, lambda: self.variables_panel.set_status("Sin pares de distancias finitas"))
                return

            pl = df.loc[mask, "distance_pc_PL"].to_numpy(dtype=float)
            ref = df.loc[mask, dist_col].to_numpy(dtype=float)
            frac = np.abs(pl - ref) / np.maximum(ref, 1.0)
            median_frac = float(np.median(frac))
            n = int(mask.sum())
            msg = f"Objetos comparados: {n}\nMediana diferencia fraccional: {median_frac:.3f}"
            self.root.after(0, lambda: messagebox.showinfo("Validación P-L", msg))
            self.root.after(0, lambda: self.variables_panel.set_status(f"Validación completa: {n} objetos"))

        threading.Thread(target=worker, daemon=True).start()

    def _bayestar_preload_worker(self) -> None:
        """Carga Bayestar2019 fuera del hilo principal de Tkinter."""
        try:
            prime_bayestar_cache()
        except Exception as exc:
            self._bayestar_error = str(exc)

    def _poll_bayestar_preload_thread(self) -> None:
        """Comprueba el estado de la precarga y actualiza la GUI cuando termina."""
        if not self._bayestar_preload_thread:
            return

        if self._bayestar_preload_thread.is_alive():
            self.root.after(100, self._poll_bayestar_preload_thread)
            return

        if self._bayestar_error is None:
            self._on_bayestar_preload_success()
        else:
            self._on_bayestar_preload_error(self._bayestar_error)

    def _on_bayestar_preload_success(self) -> None:
        """Marca Bayestar como listo para correccion y actualiza el estado."""
        self._bayestar_ready = True
        self._bayestar_error = None
        current_status = self.status_bar.status_var.get().lower()
        if "bayestar" in current_status or "cargando" in current_status:
            self._set_status("Bayestar2019 listo para corrección")

    def _on_bayestar_preload_error(self, message: str) -> None:
        """Registra el error de precarga sin bloquear el resto de la GUI."""
        self._bayestar_ready = False
        self._bayestar_error = message
        current_status = self.status_bar.status_var.get().lower()
        if "bayestar" in current_status or "cargando" in current_status:
            self._set_status(f"advertencia: no se pudo precargar Bayestar2019: {message}")

    def _start_download(self) -> None:
        """Valida la entrada y lanza la descarga en un hilo secundario."""
        if self._download_thread and self._download_thread.is_alive():
            self._set_status("descargando... espera a que termine")
            return

        try:
            n_stars = int(self.n_stars_var.get())
            max_dist = float(self.max_dist_var.get())
            if n_stars <= 0 or max_dist <= 0:
                raise ValueError
        except Exception:
            self._set_status("error: N y Max pc deben ser positivos")
            return

        self._set_status("descargando datos Gaia DR3...")
        self._set_download_controls(False)

        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=(n_stars, max_dist),
            daemon=True,
        )
        self._download_thread.start()
        self.root.after(100, self._poll_download_thread)

    def _poll_download_thread(self) -> None:
        """Rehabilita los controles cuando termina la descarga en background."""
        if self._download_thread and self._download_thread.is_alive():
            self.root.after(100, self._poll_download_thread)
            return
        self._set_download_controls(True)

    def _download_worker(self, n_stars: int, max_dist: float) -> None:
        """Ejecuta la consulta Gaia fuera del hilo principal de Tkinter."""
        try:
            df = query_gaia_sample(n_stars=n_stars, max_dist_pc=max_dist)
            # Tkinter no es thread-safe: la actualizacion de la GUI vuelve al hilo principal.
            self.root.after(100, lambda: self._on_download_success(df))
        except Exception as exc:
            # Igual para la rama de error, que debe mostrar el dialogo desde Tk.
            # Bind the message into the lambda default to avoid referencing the
            # exception variable after the except block (it gets cleared).
            msg = str(exc)
            self.root.after(100, lambda m=msg: self._on_download_error(m))

    def _on_download_success(self, df: pd.DataFrame) -> None:
        self.df_raw = df
        self.df_processed = None
        self.stats = None
        self.stats_panel.clear()
        self.data_table.set_dataframe(pd.DataFrame(columns=[
            "source_id",
            "ra",
            "dec",
            "distance_display",
            "bp_rp",
            "B_V",
            "teff",
            "M_G",
            "luminosity_solar",
            "spectral_type",
        ]))
        self.plot_panel.clear(message="Datos descargados. Presiona 'Procesar' y luego 'Graficar'.")
        self._set_status(f"descarga completada: {len(df)} filas")
        # Habilitar o deshabilitar el checkbox bayesiano según existan las columnas Bailer-Jones
        if {"r_med_photogeo", "r_med_geo"}.intersection(df.columns):
            self.bayesian_check.configure(state="normal")
        else:
            self.bayesian_check.configure(state="disabled")
        # Habilitar panel de variables si la descarga incluyo columnas de variabilidad
        if {"best_class_name", "in_vari_classification_result"}.intersection(df.columns):
            try:
                self.variables_panel.set_status("Columnas de variabilidad detectadas")
                self.variables_panel.set_enabled(True)
            except Exception:
                pass
        else:
            try:
                self.variables_panel.set_status("No hay columnas de variabilidad en la descarga")
                self.variables_panel.set_enabled(False)
            except Exception:
                pass

    def _on_download_error(self, message: str) -> None:
        self._set_status(f"error: {message}")
        messagebox.showerror("Error de descarga", message)

    def _process_data(self) -> None:
        """Deriva magnitudes fisicas y actualiza la tabla y las estadisticas."""
        if self.df_raw is None or self.df_raw.empty:
            self._set_status("error: primero descarga datos")
            return

        status_message = (
            "procesando datos con correccion de extincion..."
            if self.extinction_var.get()
            else "procesando datos..."
        )
        self._set_status(status_message)
        try:
            df = self.df_raw.copy()

            # Convertimos las columnas base en arreglos NumPy para vectorizar los calculos fisicos.
            parallax = df["parallax"].to_numpy(dtype=float)
            bp_rp = df["bp_rp"].to_numpy(dtype=float)
            g_mag = df["phot_g_mean_mag"].to_numpy(dtype=float)

            with np.errstate(divide="ignore", invalid="ignore"):
                distance_pc = 1000.0 / parallax
            distance_pc[parallax <= 0] = np.nan

            df["distance_pc"] = distance_pc
            df["B_V"] = bv_from_bprp(bp_rp)
            df["teff"] = teff_from_bv(df["B_V"].to_numpy(dtype=float))
            df["M_G"] = absolute_magnitude(g_mag, parallax)
            df["luminosity_solar"] = luminosity_solar(df["M_G"].to_numpy(dtype=float))
            df["spectral_type"] = spectral_type(df["teff"].to_numpy(dtype=float))

            # Construir columnas bayesianas si las columnas de Bailer-Jones existen
            if {"r_med_photogeo", "r_med_geo"}.intersection(df.columns):
                try:
                    r_med_photogeo = df.get("r_med_photogeo", np.full_like(distance_pc, np.nan, dtype=float))
                    r_med_geo = df.get("r_med_geo", np.full_like(distance_pc, np.nan, dtype=float))
                    df["distance_pc_bayesian"] = best_distance_bayesian(r_med_photogeo, r_med_geo, parallax)

                    # incertidumbres asimetricas: preferir photogeo, fallback geo
                    r_lo_phot = df.get("r_lo_photogeo", np.full_like(distance_pc, np.nan, dtype=float))
                    r_hi_phot = df.get("r_hi_photogeo", np.full_like(distance_pc, np.nan, dtype=float))
                    r_lo_geo = df.get("r_lo_geo", np.full_like(distance_pc, np.nan, dtype=float))
                    r_hi_geo = df.get("r_hi_geo", np.full_like(distance_pc, np.nan, dtype=float))
                    lo = np.where(np.isfinite(r_lo_phot), r_lo_phot, r_lo_geo)
                    hi = np.where(np.isfinite(r_hi_phot), r_hi_phot, r_hi_geo)
                    df["distance_lo_bayesian"] = lo
                    df["distance_hi_bayesian"] = hi

                    # magnitud absoluta y luminosidad bayesiana
                    df["M_G_bayesian"] = absolute_magnitude_bayesian(g_mag, df["distance_pc_bayesian"].to_numpy(dtype=float))
                    df["luminosity_solar_bayesian"] = luminosity_solar(df["M_G_bayesian"].to_numpy(dtype=float))
                except Exception:
                    # No bloquear el flujo de procesamiento por errores en columnas opcionales
                    pass

            if self.extinction_var.get():
                if not self._bayestar_ready:
                    if self._bayestar_preload_thread and self._bayestar_preload_thread.is_alive():
                        self._set_status(
                            "Bayestar2019 sigue cargando en segundo plano. "
                            "Espera unos segundos y vuelve a procesar."
                        )
                    elif self._bayestar_error is not None:
                        self._set_status(f"error: no se pudo precargar Bayestar2019: {self._bayestar_error}")
                    else:
                        self._set_status("Bayestar2019 no está precargado todavía.")
                    return
                # Seleccionar columna de distancia para la correccion (bayesiana si esta activa)
                dist_col = (
                    "distance_pc_bayesian"
                    if self.bayesian_var.get() and "distance_pc_bayesian" in df.columns
                    else "distance_pc"
                )
                df = apply_extinction_correction(df, distance_col=dist_col)

            # Enriquecemos el DataFrame con columnas de variabilidad si están presentes
            try:
                df = add_variability_columns(df)
            except Exception:
                # No bloquear por errores en columnas opcionales de variabilidad
                pass

            self.df_processed = df
            self.stats = compute_statistics(df)

            self.stats_panel.update_from_stats(self.stats)

            # La tabla muestra solo las columnas derivadas mas utiles para revision rapida.
            # La columna de distancia mostrada depende del toggle bayesiano.
            if self.bayesian_var.get() and "distance_pc_bayesian" in df.columns:
                df["distance_display"] = df["distance_pc_bayesian"]
            else:
                df["distance_display"] = df["distance_pc"]

            table_cols = [
                "source_id",
                "ra",
                "dec",
                "distance_display",
                "bp_rp",
                "B_V",
                "teff",
                "M_G",
                "luminosity_solar",
                "spectral_type",
            ]
            shown, total = self.data_table.set_dataframe(df[table_cols], max_rows=500)

            suffix = " con correccion de extincion" if self.extinction_var.get() else ""
            if total > shown:
                self._set_status(f"procesado listo{suffix}: mostrando {shown}/{total} filas")
            else:
                self._set_status(f"procesado listo{suffix}: {total} filas")
        except Exception as exc:
            self._set_status(f"error: {exc}")
            messagebox.showerror("Error de procesamiento", str(exc))

    def _plot_data(self) -> None:
        """Dibuja o actualiza el diagrama HR con los datos ya procesados."""
        if self.df_processed is None or self.df_processed.empty:
            self._set_status("error: primero procesa datos")
            return

        self._set_status("graficando diagrama HR...")
        try:
            plot_hr(
                self.df_processed,
                ax=self.ax,
                use_corrected=self.extinction_var.get(),
                use_bayesian=self.bayesian_var.get(),
                isochrones_to_overlay=self.active_isochrones,
            )
            self.plot_panel.draw()
            self._set_status("grafica lista")
        except Exception as exc:
            self._set_status(f"error: {exc}")
            messagebox.showerror("Error de grafica", str(exc))

    def _export_csv(self) -> None:
        """Guarda el DataFrame procesado en results/stars_processed.csv."""
        if self.df_processed is None or self.df_processed.empty:
            self._set_status("error: no hay datos procesados para exportar")
            return

        output = Path(__file__).resolve().parent.parent / "results" / "stars_processed.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        self.df_processed.to_csv(output, index=False)
        self._set_status(f"csv exportado: {output}")

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
from scipy.spatial import cKDTree
from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes

from data.download import query_gaia_sample
from src.extinction import apply_extinction_correction, prime_bayestar_cache
from src.distances import best_distance_bayesian, absolute_magnitude_bayesian
from gui.plots import MatplotlibPanel
from gui.widgets import (
    DataTable,
    DetailPanel,
    IsochronePanel,
    SpectroscopyPanel,
    StatisticsPanel,
    StatusBar,
    VariablesPanel,
)
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


def _to_float(value: object) -> float | None:
    """Convierte valores heterogeneos a float si es posible."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None
    return None


class StellarClassifierApp:
    """Ventana principal y coordinadora del flujo de trabajo interactivo."""

    HR_CLICK_TOLERANCE = 0.08
    """Tolerancia para asociar click a estrella en el HR.

    Las coordenadas se normalizan por la desviacion estandar de cada
    eje, asi 0.08 corresponde a ~8% de una sigma. Suficientemente
    laxo para muestras compactas pero estricto en muestras dispersas.
    Empirico; ajustar si el click se siente "perdido".
    """

    def __init__(self, root: tk.Tk, preload_bayestar: bool = True):
        """Construye la ventana, el estado interno y todos los widgets."""
        self.root = root
        self.root.title("stellar-classifier")
        self.root.geometry("1200x800")
        self.root.minsize(1024, 720)

        # Estado interno compartido entre descarga, procesamiento, tabla y grafico.
        self.df_raw: pd.DataFrame | None = None
        self.df_processed: pd.DataFrame | None = None
        self.stats: dict[str, object] | None = None
        self.fig: "matplotlib.figure.Figure | None" = None
        self.ax: "matplotlib.axes.Axes | None" = None

        self._download_thread: threading.Thread | None = None
        self._bayestar_preload_thread: threading.Thread | None = None
        self._isochrones_thread: threading.Thread | None = None
        self._bayestar_ready = not preload_bayestar
        self._bayestar_error: str | None = None
        self._isochrones_error: str | None = None
        self._preload_bayestar = preload_bayestar

        self.available_isochrones: list[dict[str, object]] = []
        self.active_isochrones: list[dict[str, object]] = []
        self._isochrone_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
        self.df_crossmatch: pd.DataFrame | None = None
        self.df_spectra_results: pd.DataFrame | None = None
        self._mpl_click_cid: int | None = None
        self._hr_kdtree: cKDTree | None = None
        self._hr_kdtree_scale: np.ndarray | None = None
        self._hr_kdtree_source_ids: np.ndarray | None = None
        self._hr_kdtree_cols: tuple[str, str] | None = None
        self._hr_selected_marker: Any | None = None
        self._spectra_source_order: list[str] = []
        self._selected_spectrum_source_id: str | None = None

        self.n_stars_var = tk.IntVar(value=5000)
        self.max_dist_var = tk.IntVar(value=100)
        self.extinction_var = tk.BooleanVar(value=False)
        self.bayesian_var = tk.BooleanVar(value=False)
        self.only_variables_var = tk.BooleanVar(value=False)
        self.hr_only_lamost_var = tk.BooleanVar(value=False)

        self._build_layout()

        if self._preload_bayestar:
            self._start_bayestar_preload()
        self._start_isochrones_load()

    def _build_layout(self) -> None:
        """Construye la disposicion general de la interfaz."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_action_bar()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)

        tab_hr = ttk.Frame(self.notebook)
        tab_spec = ttk.Frame(self.notebook)
        self.notebook.add(tab_hr, text="Diagrama HR")
        self.notebook.add(tab_spec, text="Espectroscopia")

        tab_hr.columnconfigure(0, weight=1)
        tab_hr.rowconfigure(0, weight=3)
        tab_hr.rowconfigure(1, weight=2)

        content_frame = ttk.Frame(tab_hr)
        content_frame.grid(row=0, column=0, sticky="nsew")
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
        # point selection will be wired once detail panel exists

        right_frame = ttk.Frame(content_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=0)
        right_frame.rowconfigure(2, weight=0)
        right_frame.rowconfigure(3, weight=1)

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

        self.variables_panel = VariablesPanel(
            right_frame,
            on_validate=self._validate_pl,
            on_filter_change=self._on_variable_filter_changed,
        )
        self.variables_panel.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.variables_panel.set_status("Panel de variables: esperando descarga")
        self.variables_panel.set_enabled(False)

        # Panel lateral de detalle para la estrella seleccionada
        self.detail_panel = DetailPanel(right_frame)
        self.detail_panel.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        # Wire the plot selection callback to update the detail panel
        self.plot_panel.on_point_selected = self._on_point_selected

        table_frame = ttk.LabelFrame(tab_hr, text="Tabla de datos")
        table_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
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
            ("variable_type", "Tipo var.", 90),
        ]
        self.data_table = DataTable(table_frame, columns=table_columns)
        self.data_table.grid(row=0, column=0, sticky="nsew")

        tab_spec.columnconfigure(0, weight=1)
        tab_spec.rowconfigure(0, weight=1)
        self.spectroscopy_panel = SpectroscopyPanel(
            tab_spec,
            on_crossmatch=self._start_crossmatch,
            on_batch_analyse=self._start_batch_analyse,
            on_prev_spectrum=self._show_previous_spectrum,
            on_next_spectrum=self._show_next_spectrum,
            on_focus_hr=self._focus_selected_spectrum_in_hr,
        )
        self.spectroscopy_panel.grid(row=0, column=0, sticky="nsew")
        self.spectroscopy_panel.set_navigation_state(index=None, total=0, has_selection=False)

        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(row=2, column=0, sticky="ew")

    def _get_hr_columns(self) -> tuple[str, str]:
        """Selecciona columnas activas de HR segun toggles y disponibilidad."""
        if self.df_processed is None or self.df_processed.empty:
            return "teff", "M_G"

        teff_col = "teff"
        mg_col = "M_G"
        if self.extinction_var.get() and {"teff_corr", "M_G_corr"}.issubset(self.df_processed.columns):
            teff_col = "teff_corr"
            mg_col = "M_G_corr"
        elif self.bayesian_var.get() and "M_G_bayesian" in self.df_processed.columns:
            mg_col = "M_G_bayesian"
        return teff_col, mg_col

    def _get_crossmatched_source_ids(self) -> set[str]:
        """Devuelve source_id de estrellas con espectro LAMOST."""
        if self.df_crossmatch is None or self.df_crossmatch.empty:
            return set()
        return set(self.df_crossmatch["source_id"].astype(str).tolist())

    def _get_hr_dataframe_for_plot(self) -> pd.DataFrame:
        """Devuelve el DataFrame que se dibuja en HR segun filtros activos."""
        if self.df_processed is None:
            return pd.DataFrame()
        if not self.hr_only_lamost_var.get():
            return self.df_processed

        source_ids = self._get_crossmatched_source_ids()
        if not source_ids:
            return self.df_processed.iloc[0:0].copy()
        mask = self.df_processed["source_id"].astype(str).isin(source_ids)
        return self.df_processed.loc[mask].copy()

    def _update_spectrum_navigation_state(self) -> None:
        """Sincroniza el estado de navegacion espectral en el panel."""
        total = len(self._spectra_source_order)
        if total == 0 or self._selected_spectrum_source_id is None:
            self.spectroscopy_panel.set_navigation_state(index=None, total=total, has_selection=False)
            return

        try:
            idx = self._spectra_source_order.index(str(self._selected_spectrum_source_id))
        except ValueError:
            idx = None
        self.spectroscopy_panel.set_navigation_state(index=idx, total=total, has_selection=idx is not None)

    def _show_spectrum_for_source(self, source_id: str, switch_to_spectro: bool = True) -> None:
        """Carga y muestra el espectro de una estrella seleccionada por source_id."""
        if self.df_crossmatch is None or self.df_crossmatch.empty:
            self.spectroscopy_panel.set_status("error: primero busca espectros LAMOST")
            return

        source_str = str(source_id)
        match = self.df_crossmatch.loc[self.df_crossmatch["source_id"].astype(str) == source_str]
        if match.empty:
            if switch_to_spectro:
                self.notebook.select(1)
            self.spectroscopy_panel.clear_spectrum()
            self.spectroscopy_panel.set_status("la estrella seleccionada no tiene espectro LAMOST")
            return

        obs_row = match.iloc[0]
        obsid = obs_row.get("obsid")
        snrg = obs_row.get("snrg")
        class_lamost = obs_row.get("class_lamost")
        subclass_lamost = obs_row.get("subclass_lamost")

        teff_phot = None
        if self.df_processed is not None and not self.df_processed.empty:
            row_src = self.df_processed.loc[self.df_processed["source_id"].astype(str) == source_str]
            if not row_src.empty and "teff" in row_src.columns:
                teff_raw = row_src.iloc[0].get("teff")
                teff_phot = _to_float(teff_raw)

        self._selected_spectrum_source_id = source_str
        self._update_spectrum_navigation_state()

        if switch_to_spectro:
            self.notebook.select(1)
        self.spectroscopy_panel.set_status("analizando espectro seleccionado...")

        def worker() -> None:
            from src.lamost import analyse_star_spectrum

            obsid_safe: int | str = obsid if isinstance(obsid, (int, str)) else str(obsid)

            result = analyse_star_spectrum(
                source_id=source_str,
                obsid=obsid_safe,
                teff_photometric=teff_phot,
            )
            result["snrg"] = snrg
            result["class_lamost"] = class_lamost
            result["subclass_lamost"] = subclass_lamost
            self.root.after(0, lambda r=result: self.spectroscopy_panel.show_spectrum(r))  # type: ignore[misc]
            if result.get("success"):
                self.root.after(0, lambda: self.spectroscopy_panel.set_status("espectro cargado"))
            else:
                self.root.after(0, lambda: self.spectroscopy_panel.set_status(f"error: {result.get('error') or 'sin detalle'}"))

        threading.Thread(target=worker, daemon=True).start()

    def _show_previous_spectrum(self) -> None:
        """Muestra el espectro anterior de la lista de coincidencias."""
        if not self._spectra_source_order:
            self.spectroscopy_panel.set_status("error: no hay espectros para navegar")
            return
        if self._selected_spectrum_source_id not in self._spectra_source_order:
            self._show_spectrum_for_source(self._spectra_source_order[0], switch_to_spectro=True)
            return
        idx = self._spectra_source_order.index(str(self._selected_spectrum_source_id))
        if idx <= 0:
            return
        self._show_spectrum_for_source(self._spectra_source_order[idx - 1], switch_to_spectro=True)

    def _show_next_spectrum(self) -> None:
        """Muestra el espectro siguiente de la lista de coincidencias."""
        if not self._spectra_source_order:
            self.spectroscopy_panel.set_status("error: no hay espectros para navegar")
            return
        if self._selected_spectrum_source_id not in self._spectra_source_order:
            self._show_spectrum_for_source(self._spectra_source_order[0], switch_to_spectro=True)
            return
        idx = self._spectra_source_order.index(str(self._selected_spectrum_source_id))
        if idx >= len(self._spectra_source_order) - 1:
            return
        self._show_spectrum_for_source(self._spectra_source_order[idx + 1], switch_to_spectro=True)

    def _focus_selected_spectrum_in_hr(self) -> None:
        """Lleva la vista al HR y resalta la estrella del espectro seleccionado."""
        if self._selected_spectrum_source_id is None:
            self.spectroscopy_panel.set_status("error: no hay espectro seleccionado")
            return
        if self.df_processed is None or self.df_processed.empty:
            self.spectroscopy_panel.set_status("error: primero procesa y grafica datos")
            return
        self.notebook.select(0)
        self._highlight_source_in_hr(self._selected_spectrum_source_id)

    def _highlight_source_in_hr(self, source_id: str) -> None:
        """Marca visualmente una estrella en el diagrama HR."""
        if self.df_processed is None or self.df_processed.empty:
            return
        if self.ax is None:
            return

        teff_col, mg_col = self._get_hr_columns()
        row = self.df_processed.loc[self.df_processed["source_id"].astype(str) == str(source_id)]
        if row.empty:
            return
        x = _to_float(row.iloc[0].get(teff_col))
        y = _to_float(row.iloc[0].get(mg_col))
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            return

        try:
            if self._hr_selected_marker is not None:
                self._hr_selected_marker.remove()
        except Exception:
            pass
        self._hr_selected_marker = self.ax.scatter(
            [float(x)],
            [float(y)],
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            zorder=12,
        )
        self.plot_panel.draw()

    def _rebuild_hr_kdtree(self) -> None:
        """Reconstruye el KDTree de HR para habilitar click cercano a estrellas."""
        self._hr_kdtree = None
        self._hr_kdtree_scale = None
        self._hr_kdtree_source_ids = None
        self._hr_kdtree_cols = None

        hr_df = self._get_hr_dataframe_for_plot()
        if hr_df.empty:
            return

        teff_col, mg_col = self._get_hr_columns()
        if teff_col not in hr_df.columns or mg_col not in hr_df.columns:
            return

        x = pd.to_numeric(hr_df[teff_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(hr_df[mg_col], errors="coerce").to_numpy(dtype=float)
        sid = hr_df.get("source_id", pd.Series(np.arange(len(hr_df))))
        source_ids = sid.astype(str).to_numpy()

        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            return

        pts = np.column_stack([x[mask], y[mask]])
        scale = np.nanstd(pts, axis=0)
        scale[~np.isfinite(scale)] = 1.0
        scale[scale == 0] = 1.0
        pts_norm = pts / scale

        self._hr_kdtree = cKDTree(pts_norm)
        self._hr_kdtree_scale = scale
        self._hr_kdtree_source_ids = source_ids[mask]
        self._hr_kdtree_cols = (teff_col, mg_col)

    def _connect_hr_click(self) -> None:
        """Conecta el evento de click sobre el HR."""
        if self.fig is None or self.df_crossmatch is None:
            return
        if self._mpl_click_cid is not None and self.fig.canvas is not None:
            self.fig.canvas.mpl_disconnect(self._mpl_click_cid)
        self._mpl_click_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_hr_click)

    def _on_hr_click(self, event: object) -> None:
        """Atiende click en HR y muestra espectro si la estrella tiene match LAMOST."""
        if self.df_processed is None or self.df_processed.empty:
            return
        if self.df_crossmatch is None or self.df_crossmatch.empty:
            return
        event_any = cast(Any, event)
        if event_any is None or event_any.xdata is None or event_any.ydata is None:
            return

        if self._hr_kdtree is None or self._hr_kdtree_scale is None or self._hr_kdtree_source_ids is None:
            self._rebuild_hr_kdtree()
        if self._hr_kdtree is None or self._hr_kdtree_scale is None or self._hr_kdtree_source_ids is None:
            return

        click_norm = np.array([float(event_any.xdata), float(event_any.ydata)], dtype=float) / self._hr_kdtree_scale
        dist, idx = self._hr_kdtree.query(click_norm, k=1)
        if not np.isfinite(dist) or dist > self.HR_CLICK_TOLERANCE:
            return

        source_id = str(self._hr_kdtree_source_ids[int(idx)])
        self._show_spectrum_for_source(source_id, switch_to_spectro=True)

    def _start_crossmatch(self) -> None:
        """Lanza el cross-match LAMOST en background."""
        if self.df_processed is None or self.df_processed.empty:
            self.spectroscopy_panel.set_status("error: primero descarga y procesa datos")
            return
        processed_df = self.df_processed

        self.spectroscopy_panel.set_status("buscando espectros en LAMOST...")

        def worker() -> None:
            from src.lamost import crossmatch_lamost

            try:
                df_match = crossmatch_lamost(processed_df, radius_arcsec=2.0, max_stars=500)
                self.df_crossmatch = df_match
                n = len(df_match)
                source_order = [str(v) for v in df_match.get("source_id", pd.Series(dtype=object)).astype(str).tolist()]
                # Mantener orden y evitar duplicados para navegacion estable.
                self._spectra_source_order = list(dict.fromkeys(source_order))
                self._selected_spectrum_source_id = self._spectra_source_order[0] if self._spectra_source_order else None
                def _set_crossmatch_results() -> None:
                    self.spectroscopy_panel.set_crossmatch_results(df_match)

                self.root.after(0, _set_crossmatch_results)
                self.root.after(0, self._update_spectrum_navigation_state)
                def _set_crossmatch_status() -> None:
                    self.spectroscopy_panel.set_status(f"cross-match listo: {n} estrellas con espectro LAMOST")

                self.root.after(0, _set_crossmatch_status)
                self.root.after(0, self._connect_hr_click)
                self.root.after(0, self._refresh_hr_plot)
            except Exception as exc:
                msg = str(exc)

                def _set_crossmatch_error() -> None:
                    self.spectroscopy_panel.set_status(f"error en cross-match: {msg}")

                self.root.after(0, _set_crossmatch_error)

        threading.Thread(target=worker, daemon=True).start()

    def _start_batch_analyse(self) -> None:
        """Lanza el analisis en lote en background con progreso."""
        if self.df_crossmatch is None or self.df_crossmatch.empty:
            self.spectroscopy_panel.set_status("error: primero busca espectros LAMOST")
            return
        crossmatch_df = self.df_crossmatch
        processed_df = self.df_processed
        if processed_df is None:
            self.spectroscopy_panel.set_status("error: primero procesa datos")
            return

        def worker() -> None:
            from src.lamost import batch_analyse_spectra

            def progress(n_done: int, n_total: int) -> None:
                self.root.after(0, lambda: self.spectroscopy_panel.set_status(
                    f"analizando espectros: {n_done}/{n_total}..."
                ))

            try:
                df_results = batch_analyse_spectra(
                    crossmatch_df,
                    processed_df,
                    max_spectra=100,
                    progress_callback=progress,
                )
                self.df_spectra_results = df_results
                n_ok = int(df_results["success"].sum()) if "success" in df_results.columns else 0
                def _set_batch_status() -> None:
                    self.spectroscopy_panel.set_status(f"analisis listo: {n_ok} espectros procesados correctamente")

                self.root.after(0, _set_batch_status)
            except Exception as exc:
                msg = str(exc)

                def _set_batch_error() -> None:
                    self.spectroscopy_panel.set_status(f"error en analisis: {msg}")

                self.root.after(0, _set_batch_error)

        threading.Thread(target=worker, daemon=True).start()

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

        self.variables_check = ttk.Checkbutton(
            action_bar,
            text="Solo variables",
            variable=self.only_variables_var,
        )
        self.variables_check.grid(row=0, column=6, padx=(0, 12))

        self.hr_lamost_check = ttk.Checkbutton(
            action_bar,
            text="HR: solo con espectro",
            variable=self.hr_only_lamost_var,
            command=self._refresh_hr_plot,
        )
        self.hr_lamost_check.grid(row=0, column=7, padx=(0, 12))

        ttk.Label(action_bar, text="N:").grid(row=0, column=8, padx=(0, 4))
        self.n_stars_entry = ttk.Entry(action_bar, textvariable=self.n_stars_var, width=8)
        self.n_stars_entry.grid(row=0, column=9, padx=(0, 8))

        ttk.Label(action_bar, text="Max pc:").grid(row=0, column=10, padx=(0, 4))
        self.max_dist_entry = ttk.Entry(action_bar, textvariable=self.max_dist_var, width=8)
        self.max_dist_entry.grid(row=0, column=11)

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

    def _on_variable_filter_changed(self, active_types: set[str] | None) -> None:
        """Refresca el HR cuando cambia el filtro de variables."""
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
        processed_df = self.df_processed

        if self._isochrones_thread and self._isochrones_thread.is_alive():
            self._set_status("espera a que termine la carga de isócronas")
            return

        self.isochrone_panel.set_status("Ajustando edad en segundo plano...")

        def worker() -> None:
            try:
                result = fit_best_age(
                    processed_df,
                    age_grid=np.arange(7.0, 10.1, 0.1),
                    metallicity=0.0,
                    use_corrected=self.extinction_var.get(),
                    use_bayesian=self.bayesian_var.get(),
                    isochrones_dir=str(ISOCHRONES_DIR),
                )
                def _on_success() -> None:
                    self._on_fit_best_age_success(result)

                self.root.after(0, _on_success)
            except Exception as exc:
                msg = str(exc)
                def _on_error() -> None:
                    self._on_fit_best_age_error(msg)

                self.root.after(0, _on_error)

        threading.Thread(target=worker, daemon=True).start()

    def _on_fit_best_age_success(self, result: dict[str, object]) -> None:
        """Muestra el resultado del ajuste y sobrepone la mejor isócrona."""
        best_log_age_val = _to_float(result.get("best_log_age"))
        best_log_age = best_log_age_val if best_log_age_val is not None else float("nan")
        best_label = str(result["best_age_label"])
        chi2_val = _to_float(result.get("min_chi2"))
        chi2 = chi2_val if chi2_val is not None else float("nan")
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
        processed_df = self.df_processed

        self.variables_panel.set_status("Validando P-L en segundo plano...")

        def worker() -> None:
            try:
                df = processed_df
                dist_col = (
                    "distance_pc_bayesian" if self.bayesian_var.get() and "distance_pc_bayesian" in df.columns else "distance_pc"
                )
                if "distance_pc_PL" not in df.columns:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Validación P-L",
                        "No hay distancias P-L calculadas para esta muestra."
                    ))
                    self.root.after(0, lambda: self.variables_panel.set_status(
                        "Sin distancias P-L en la muestra"
                    ))
                    return

                try:
                    pl_vals = pd.to_numeric(df["distance_pc_PL"], errors="coerce").to_numpy(dtype=float)
                    ref_vals = pd.to_numeric(df[dist_col], errors="coerce").to_numpy(dtype=float)
                except Exception as cast_exc:
                    msg = str(cast_exc)
                    def _show_cast_error() -> None:
                        messagebox.showerror("Error de validación", f"No se pudo convertir distancias: {msg}")

                    self.root.after(0, _show_cast_error)
                    return

                mask = np.isfinite(pl_vals) & np.isfinite(ref_vals)
                if not mask.any():
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Validación P-L",
                        "No hay objetos con ambas distancias (P-L y geom/bayes) finitas."
                    ))
                    self.root.after(0, lambda: self.variables_panel.set_status(
                        "Sin pares de distancias finitas"
                    ))
                    return

                frac = np.abs(pl_vals[mask] - ref_vals[mask]) / np.maximum(ref_vals[mask], 1.0)
                median_frac = float(np.median(frac))
                n = int(mask.sum())
                msg = f"Objetos comparados: {n}\nMediana diferencia fraccional: {median_frac:.3f}"
                self.root.after(0, lambda: messagebox.showinfo("Validación P-L", msg))
                self.root.after(0, lambda: self.variables_panel.set_status(
                    f"Validación completa: {n} objetos, mediana={median_frac:.3f}"
                ))
            except Exception as exc:
                err_msg = str(exc)
                self.root.after(0, lambda: messagebox.showerror("Error de validación", err_msg))
                self.root.after(0, lambda: self.variables_panel.set_status(f"Error: {err_msg}"))

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
            args=(n_stars, max_dist, self.only_variables_var.get()),
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

    def _download_worker(self, n_stars: int, max_dist: float, only_variables: bool = False) -> None:
        """Ejecuta la consulta Gaia fuera del hilo principal de Tkinter."""
        try:
            df = query_gaia_sample(n_stars=n_stars, max_dist_pc=max_dist, only_variables=only_variables)
            # Tkinter no es thread-safe: la actualizacion de la GUI vuelve al hilo principal.
            self.root.after(100, lambda: self._on_download_success(df))
        except Exception as exc:
            # Igual para la rama de error, que debe mostrar el dialogo desde Tk.
            # Bind the message into the lambda default to avoid referencing the
            # exception variable after the except block (it gets cleared).
            msg = str(exc)
            self.root.after(100, lambda m=msg: self._on_download_error(m))  # type: ignore[misc]

    def _on_download_success(self, df: pd.DataFrame) -> None:
        self.df_raw = df
        self.df_processed = None
        self.stats = None
        self.df_crossmatch = None
        self.df_spectra_results = None
        self._spectra_source_order = []
        self._selected_spectrum_source_id = None
        self._hr_kdtree = None
        self._hr_kdtree_scale = None
        self._hr_kdtree_source_ids = None
        self._hr_selected_marker = None
        if hasattr(self, "spectroscopy_panel"):
            self.spectroscopy_panel.set_crossmatch_results(pd.DataFrame())
            self.spectroscopy_panel.set_navigation_state(index=None, total=0, has_selection=False)
            self.spectroscopy_panel.clear_spectrum()
            self.spectroscopy_panel.set_status("esperando cross-match LAMOST")
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
        try:
            self.variables_panel.set_status("Descarga lista. Procesa para detectar variables.")
            self.variables_panel.set_enabled(False)
        except Exception:
            pass

    def _on_download_error(self, message: str) -> None:
        self._set_status(f"error: {message}")
        messagebox.showerror("Error de descarga", message)

    def _on_point_selected(self, info: dict[str, object]) -> None:
        """Actualiza el panel lateral con info de la estrella seleccionada."""
        try:
            if hasattr(self, "detail_panel") and self.detail_panel is not None:
                self.detail_panel.set_details(info)
        except Exception:
            pass

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
                n_vars = int(df["is_variable"].sum()) if "is_variable" in df.columns else 0
                def _set_variables_status() -> None:
                    self.variables_panel.set_status(
                        f"{n_vars} estrellas variables detectadas en la muestra"
                        if n_vars > 0 else "Sin variables detectadas en esta muestra"
                    )

                self.root.after(
                    0,
                    _set_variables_status,
                )
                self.variables_panel.set_enabled(True)
            except Exception as var_exc:
                self.variables_panel.set_status(f"Error en variables: {var_exc}")
                self.variables_panel.set_enabled(False)

            self.df_processed = df
            self._rebuild_hr_kdtree()
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
            if "variable_type" in df.columns:
                table_cols.append("variable_type")
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

        df_plot = self._get_hr_dataframe_for_plot()
        if df_plot.empty:
            self.plot_panel.clear(message="No hay estrellas con espectro LAMOST para mostrar en HR")
            self._set_status("sin estrellas con espectro LAMOST para el filtro activo")
            return

        self._set_status("graficando diagrama HR...")
        try:
            if self.fig is None:
                self._set_status("error: figura HR no inicializada")
                return
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            if hasattr(self.plot_panel, "update_ax"):
                self.plot_panel.update_ax(self.ax)
            else:
                self.plot_panel.ax = self.ax

            show_vars = self.variables_panel.get_show_variables()
            active_types = self.variables_panel.get_active_types() if show_vars else None
            plot_hr(
                df_plot,
                ax=self.ax,
                use_corrected=self.extinction_var.get(),
                use_bayesian=self.bayesian_var.get(),
                isochrones_to_overlay=self.active_isochrones,
                highlight_variables=show_vars,
                variable_types_to_show=active_types if show_vars else None,
            )

            # Marcar estrellas con espectro para facilitar seleccion visual en HR.
            if self.df_crossmatch is not None and not self.df_crossmatch.empty:
                teff_col, mg_col = self._get_hr_columns()
                ids_with_spec = self._get_crossmatched_source_ids()
                mask_spec = df_plot["source_id"].astype(str).isin(ids_with_spec)
                if mask_spec.any() and teff_col in df_plot.columns and mg_col in df_plot.columns:
                    x_spec = pd.to_numeric(df_plot.loc[mask_spec, teff_col], errors="coerce").to_numpy(dtype=float)
                    y_spec = pd.to_numeric(df_plot.loc[mask_spec, mg_col], errors="coerce").to_numpy(dtype=float)
                    valid = np.isfinite(x_spec) & np.isfinite(y_spec)
                    if valid.any():
                        self.ax.scatter(
                            x_spec[valid],
                            y_spec[valid],
                            s=36,
                            facecolors="none",
                            edgecolors="gold",
                            linewidths=1.0,
                            alpha=0.9,
                            zorder=9,
                            label="Con espectro LAMOST",
                        )

            self.plot_panel.set_point_context(df_plot, getattr(self.fig, "_hr_scatter", None))
            modes = []
            if self.extinction_var.get():
                modes.append("extinción corregida")
            if self.bayesian_var.get():
                modes.append("bayesiano")
            if show_vars:
                modes.append("variables")
            if self.hr_only_lamost_var.get():
                modes.append("solo LAMOST")
            mode = " + ".join(modes) if modes else "bruto"
            self.plot_panel.set_display_mode(mode)
            self.plot_panel.capture_view_limits()
            self.plot_panel.draw()
            if self.df_crossmatch is not None and not self.df_crossmatch.empty:
                self._connect_hr_click()
            if self._selected_spectrum_source_id is not None:
                self._highlight_source_in_hr(self._selected_spectrum_source_id)
            if hasattr(self.plot_panel, "toolbar") and self.plot_panel.toolbar is not None:
                self.plot_panel.toolbar.update()
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

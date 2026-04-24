"""Aplicacion principal Tkinter para stellar-classifier."""

from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd

from data.download import query_gaia_sample
from gui.plots import MatplotlibPanel
from gui.widgets import DataTable, StatisticsPanel, StatusBar
from src.hr_diagram import plot_hr
from src.statistics import compute_statistics
from src.temperature import (
    absolute_magnitude,
    bv_from_bprp,
    luminosity_solar,
    spectral_type,
    teff_from_bv,
)


class StellarClassifierApp:
    """Ventana principal de la aplicacion."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("stellar-classifier")
        self.root.geometry("1200x800")
        self.root.minsize(1024, 720)

        # Estado interno requerido por el diseno.
        self.df_raw: pd.DataFrame | None = None
        self.df_processed: pd.DataFrame | None = None
        self.stats: dict | None = None
        self.fig = None
        self.ax = None

        self._download_thread: threading.Thread | None = None

        self.n_stars_var = tk.IntVar(value=5000)
        self.max_dist_var = tk.IntVar(value=100)

        self._build_layout()

    def _build_layout(self) -> None:
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

        self.stats_panel = StatisticsPanel(right_frame)
        self.stats_panel.grid(row=0, column=0, sticky="nsew")

        table_frame = ttk.LabelFrame(self.root, text="Tabla de datos")
        table_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        table_columns = [
            ("source_id", "source_id", 160),
            ("ra", "ra", 90),
            ("dec", "dec", 90),
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

        ttk.Label(action_bar, text="N:").grid(row=0, column=4, padx=(0, 4))
        self.n_stars_entry = ttk.Entry(action_bar, textvariable=self.n_stars_var, width=8)
        self.n_stars_entry.grid(row=0, column=5, padx=(0, 8))

        ttk.Label(action_bar, text="Max pc:").grid(row=0, column=6, padx=(0, 4))
        self.max_dist_entry = ttk.Entry(action_bar, textvariable=self.max_dist_var, width=8)
        self.max_dist_entry.grid(row=0, column=7)

    def _set_status(self, text: str) -> None:
        self.status_bar.set_status(text)

    def _set_download_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.download_btn.configure(state=state)
        self.n_stars_entry.configure(state=state)
        self.max_dist_entry.configure(state=state)

    def _start_download(self) -> None:
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
        if self._download_thread and self._download_thread.is_alive():
            self.root.after(100, self._poll_download_thread)
            return
        self._set_download_controls(True)

    def _download_worker(self, n_stars: int, max_dist: float) -> None:
        try:
            df = query_gaia_sample(n_stars=n_stars, max_dist_pc=max_dist)
            self.root.after(100, lambda: self._on_download_success(df))
        except Exception as exc:
            self.root.after(100, lambda: self._on_download_error(str(exc)))

    def _on_download_success(self, df: pd.DataFrame) -> None:
        self.df_raw = df
        self.df_processed = None
        self.stats = None
        self.stats_panel.clear()
        self.data_table.set_dataframe(pd.DataFrame(columns=[
            "source_id",
            "ra",
            "dec",
            "bp_rp",
            "B_V",
            "teff",
            "M_G",
            "luminosity_solar",
            "spectral_type",
        ]))
        self.plot_panel.clear(message="Datos descargados. Presiona 'Procesar' y luego 'Graficar'.")
        self._set_status(f"descarga completada: {len(df)} filas")

    def _on_download_error(self, message: str) -> None:
        self._set_status(f"error: {message}")
        messagebox.showerror("Error de descarga", message)

    def _process_data(self) -> None:
        if self.df_raw is None or self.df_raw.empty:
            self._set_status("error: primero descarga datos")
            return

        self._set_status("procesando datos...")
        try:
            df = self.df_raw.copy()

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

            self.df_processed = df
            self.stats = compute_statistics(df)

            self.stats_panel.update_from_stats(self.stats)

            table_cols = [
                "source_id",
                "ra",
                "dec",
                "bp_rp",
                "B_V",
                "teff",
                "M_G",
                "luminosity_solar",
                "spectral_type",
            ]
            shown, total = self.data_table.set_dataframe(df[table_cols], max_rows=500)

            if total > shown:
                self._set_status(f"procesado listo: mostrando {shown}/{total} filas")
            else:
                self._set_status(f"procesado listo: {total} filas")
        except Exception as exc:
            self._set_status(f"error: {exc}")
            messagebox.showerror("Error de procesamiento", str(exc))

    def _plot_data(self) -> None:
        if self.df_processed is None or self.df_processed.empty:
            self._set_status("error: primero procesa datos")
            return

        self._set_status("graficando diagrama HR...")
        try:
            plot_hr(self.df_processed, ax=self.ax)
            self.plot_panel.draw()
            self._set_status("grafica lista")
        except Exception as exc:
            self._set_status(f"error: {exc}")
            messagebox.showerror("Error de grafica", str(exc))

    def _export_csv(self) -> None:
        if self.df_processed is None or self.df_processed.empty:
            self._set_status("error: no hay datos procesados para exportar")
            return

        output = Path(__file__).resolve().parent.parent / "results" / "stars_processed.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        self.df_processed.to_csv(output, index=False)
        self._set_status(f"csv exportado: {output}")

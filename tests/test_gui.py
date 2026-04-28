from __future__ import annotations

from pathlib import Path
import shutil
import tkinter as tk

import numpy as np
import pytest

from gui.app import StellarClassifierApp


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"Tkinter no disponible en entorno de test: {exc}")
    yield root
    root.update_idletasks()
    root.destroy()


def test_app_initializes(tk_root) -> None:
    app = StellarClassifierApp(tk_root, preload_bayestar=False)
    assert app.df_raw is None
    assert app.df_processed is None
    assert app.stats is None
    assert app.ax is not None


def test_app_process_and_plot_flow(tk_root, sample_processed_df) -> None:
    app = StellarClassifierApp(tk_root, preload_bayestar=False)

    app.df_raw = sample_processed_df[
        ["source_id", "ra", "dec", "parallax", "phot_g_mean_mag", "bp_rp"]
    ].copy()

    app._process_data()

    assert app.df_processed is not None
    assert app.stats is not None
    assert app.stats["n_stars"] == len(app.df_raw)

    app._plot_data()
    assert len(app.ax.collections) >= 1


def test_app_process_with_extinction(monkeypatch, tk_root, sample_processed_df) -> None:
    app = StellarClassifierApp(tk_root, preload_bayestar=False)

    app.df_raw = sample_processed_df[
        ["source_id", "ra", "dec", "parallax", "phot_g_mean_mag", "bp_rp"]
    ].copy()
    app.extinction_var.set(True)

    def fake_apply_extinction_correction(df, reddening_query=None, distance_col="distance_pc"):
        corrected = df.copy()
        corrected["A_V"] = 0.05
        corrected["A_G"] = 0.05 * 0.789
        corrected["E_BR"] = 0.05 * 0.415
        corrected["BP_RP_corr"] = corrected["bp_rp"] - corrected["E_BR"]
        corrected["B_V_corr"] = corrected["B_V"] * 0.98
        corrected["teff_corr"] = corrected["teff"] + 25.0
        corrected["M_G_corr"] = corrected["M_G"] - corrected["A_G"]
        corrected["luminosity_solar_corr"] = corrected["luminosity_solar"] * 1.02
        corrected["spectral_type_corr"] = corrected["spectral_type"]
        corrected["B_V"] = corrected["B_V_corr"]
        corrected["teff"] = corrected["teff_corr"]
        corrected["M_G"] = corrected["M_G_corr"]
        corrected["luminosity_solar"] = corrected["luminosity_solar_corr"]
        corrected["spectral_type"] = corrected["spectral_type_corr"]
        return corrected

    monkeypatch.setattr("gui.app.apply_extinction_correction", fake_apply_extinction_correction)

    app._process_data()

    assert app.df_processed is not None
    assert "A_V" in app.df_processed.columns
    assert np.isclose(app.df_processed["A_V"].iloc[0], 0.05)
    assert np.isclose(app.df_processed["teff"].iloc[0], app.df_processed["teff_corr"].iloc[0])


def test_app_preloads_bayestar_in_background(monkeypatch, tk_root) -> None:
    calls = {"count": 0}

    def fake_prime_bayestar_cache():
        calls["count"] += 1
        return object()

    monkeypatch.setattr("gui.app.prime_bayestar_cache", fake_prime_bayestar_cache)

    app = StellarClassifierApp(tk_root, preload_bayestar=True)
    assert app._bayestar_preload_thread is not None

    app._bayestar_preload_thread.join(timeout=1)
    app.root.update()

    assert calls["count"] == 1
    assert app._bayestar_ready is True
    assert "Bayestar2019 listo" in app.status_bar.status_var.get()


def test_bayesian_checkbox_disabled_when_no_bailer_columns(tk_root, sample_processed_df) -> None:
    app = StellarClassifierApp(tk_root, preload_bayestar=False)
    # Simular descarga sin columnas Bailer-Jones
    df = sample_processed_df[ ["source_id", "ra", "dec", "parallax", "phot_g_mean_mag", "bp_rp"] ].copy()
    app._on_download_success(df)
    assert str(app.bayesian_check.cget("state")) == "disabled"


def test_bayesian_toggle_enables_and_changes_table(tk_root, sample_processed_df) -> None:
    app = StellarClassifierApp(tk_root, preload_bayestar=False)
    # Añadir columnas Bailer-Jones plausibles
    df = sample_processed_df.copy()
    # usar la distancia geometrica como photogeo para test simple
    df["r_med_photogeo"] = df["distance_pc"].to_numpy()
    df["r_lo_photogeo"] = df["distance_pc"].to_numpy() * 0.95
    df["r_hi_photogeo"] = df["distance_pc"].to_numpy() * 1.05
    app._on_download_success(df)
    # ahora el checkbox debe estar habilitado
    assert str(app.bayesian_check.cget("state")) == "normal"
    # activar bayesiana y procesar
    app.bayesian_var.set(True)
    app.df_raw = df
    app._process_data()
    assert "distance_pc_bayesian" in app.df_processed.columns
    # cuando bayesiana ON, distance_display debe coincidir con la bayesiana
    assert np.allclose(app.df_processed["distance_display"].to_numpy(dtype=float), app.df_processed["distance_pc_bayesian"].to_numpy(dtype=float), equal_nan=True)


def test_bayesian_plus_extinction_no_errors(monkeypatch, tk_root, sample_processed_df) -> None:
    app = StellarClassifierApp(tk_root, preload_bayestar=False)
    df = sample_processed_df.copy()
    df["r_med_photogeo"] = df["distance_pc"].to_numpy()
    df["r_lo_photogeo"] = df["distance_pc"].to_numpy() * 0.95
    df["r_hi_photogeo"] = df["distance_pc"].to_numpy() * 1.05

    app._on_download_success(df)
    app.bayesian_var.set(True)
    app.extinction_var.set(True)

    # parcheamos la funcion de correccion para evitar depender de dustmaps en CI
    def fake_apply_extinction_correction(df_in, reddening_query=None, distance_col="distance_pc"):
        out = df_in.copy()
        out["A_V"] = 0.01
        # Simular que la funcion respeta distance_col (no alteraremos columnas aquí)
        return out

    monkeypatch.setattr("gui.app.apply_extinction_correction", fake_apply_extinction_correction)

    app.df_raw = df
    # No debe lanzar excepcion
    app._process_data()
    assert app.df_processed is not None


def _copy_isochrones_fixture_dir(tmp_path: Path) -> Path:
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "isochrones"
    for filepath in fixture_dir.glob("*.dat"):
        shutil.copy2(filepath, tmp_path / filepath.name)
    return tmp_path


def test_isochrone_panel_disabled_when_empty(monkeypatch, tk_root, tmp_path) -> None:
    monkeypatch.setattr("gui.app.ISOCHRONES_DIR", tmp_path)
    app = StellarClassifierApp(tk_root, preload_bayestar=False)
    if app._isochrones_thread is not None:
        app._isochrones_thread.join(timeout=1)
        app.root.update()
    assert str(app.isochrone_panel.combo.cget("state")) == "disabled"
    assert app.isochrone_panel.available_isochrones == []


def test_isochrone_dropdown_populates_and_overlays(monkeypatch, tk_root, tmp_path, sample_processed_df) -> None:
    isochrone_dir = _copy_isochrones_fixture_dir(tmp_path)
    monkeypatch.setattr("gui.app.ISOCHRONES_DIR", isochrone_dir)
    app = StellarClassifierApp(tk_root, preload_bayestar=False)
    if app._isochrones_thread is not None:
        app._isochrones_thread.join(timeout=1)
    for _ in range(20):
        app.root.update()
        if app.isochrone_panel.combo["values"]:
            break
    values = app.isochrone_panel.combo["values"]
    assert len(values) >= 1

    app.df_raw = sample_processed_df[["source_id", "ra", "dec", "parallax", "phot_g_mean_mag", "bp_rp"]].copy()
    app._process_data()

    selection = app.isochrone_panel.get_selected_isochrone()
    assert selection is not None
    app._overlay_selected_isochrone(selection)
    assert len(app.active_isochrones) == 1

    app._clear_isochrones()
    assert len(app.active_isochrones) == 0

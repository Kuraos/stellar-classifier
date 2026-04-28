from __future__ import annotations

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

    def fake_apply_extinction_correction(df, reddening_query=None):
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

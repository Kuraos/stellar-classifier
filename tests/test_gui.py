from __future__ import annotations

import tkinter as tk

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
    app = StellarClassifierApp(tk_root)
    assert app.df_raw is None
    assert app.df_processed is None
    assert app.stats is None
    assert app.ax is not None


def test_app_process_and_plot_flow(tk_root, sample_processed_df) -> None:
    app = StellarClassifierApp(tk_root)

    app.df_raw = sample_processed_df[
        ["source_id", "ra", "dec", "parallax", "phot_g_mean_mag", "bp_rp"]
    ].copy()

    app._process_data()

    assert app.df_processed is not None
    assert app.stats is not None
    assert app.stats["n_stars"] == len(app.df_raw)

    app._plot_data()
    assert len(app.ax.collections) >= 1

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


@pytest.fixture
def sample_processed_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            "source_id": np.arange(1, n + 1),
            "ra": rng.uniform(0.0, 360.0, n),
            "dec": rng.uniform(-90.0, 90.0, n),
            "parallax": rng.uniform(10.0, 50.0, n),
            "phot_g_mean_mag": rng.uniform(5.0, 14.0, n),
            "bp_rp": rng.uniform(0.3, 1.8, n),
        }
    )

    df["distance_pc"] = 1000.0 / df["parallax"]
    df["B_V"] = 0.0981 + 0.7119 * df["bp_rp"] + 0.0718 * df["bp_rp"] ** 2
    df["teff"] = 4600.0 * (
        1.0 / (0.92 * df["B_V"] + 1.7) + 1.0 / (0.92 * df["B_V"] + 0.62)
    )
    df["M_G"] = df["phot_g_mean_mag"] + 5.0 + 5.0 * np.log10(df["parallax"] / 1000.0)
    df["luminosity_solar"] = 10.0 ** ((4.74 - df["M_G"]) / 2.5)

    bins = [0, 3700, 5200, 6000, 7500, 10000, 30000, np.inf]
    labels = ["M", "K", "G", "F", "A", "B", "O"]
    df["spectral_type"] = pd.cut(df["teff"], bins=bins, labels=labels, right=False).astype(str)
    return df

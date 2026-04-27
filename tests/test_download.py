from __future__ import annotations

import os
import sys
import types

import pandas as pd
import pytest

from data import download


def _mock_gaia_modules(sample_df: pd.DataFrame):
    class FakeResults:
        def to_pandas(self):
            return sample_df

    class FakeJob:
        def get_results(self):
            return FakeResults()

    class FakeGaia:
        ROW_LIMIT = None
        last_query = ""

        @staticmethod
        def launch_job_async(query: str, dump_to_file: bool = False):
            FakeGaia.last_query = query
            return FakeJob()

    gaia_module = types.ModuleType("astroquery.gaia")
    gaia_module.Gaia = FakeGaia

    astroquery_module = types.ModuleType("astroquery")
    astroquery_module.gaia = gaia_module

    return astroquery_module, gaia_module, FakeGaia


def _mock_gaia_modules_async_fails_sync_succeeds(sample_df: pd.DataFrame):
    class FakeResults:
        def to_pandas(self):
            return sample_df

    class FakeJob:
        def get_results(self):
            return FakeResults()

    class FakeGaia:
        ROW_LIMIT = None

        @staticmethod
        def launch_job_async(query: str, dump_to_file: bool = False):
            raise RuntimeError("Error 500: null")

        @staticmethod
        def launch_job(query: str, dump_to_file: bool = False):
            return FakeJob()

    gaia_module = types.ModuleType("astroquery.gaia")
    gaia_module.Gaia = FakeGaia

    astroquery_module = types.ModuleType("astroquery")
    astroquery_module.gaia = gaia_module

    return astroquery_module, gaia_module


def _is_transient_gaia_error(message: str) -> bool:
    lowered = message.lower()
    transient_tokens = [
        "error 500",
        "error 502",
        "error 503",
        "error 504",
        "service unavailable",
        "read timed out",
        "connect timeout",
        "temporarily unavailable",
    ]
    return any(token in lowered for token in transient_tokens)


def test_query_gaia_sample_writes_csv(monkeypatch, tmp_path) -> None:
    sample_df = pd.DataFrame(
        {
            "source_id": [1, 2],
            "ra": [10.0, 20.0],
            "dec": [5.0, -2.0],
            "parallax": [20.0, 25.0],
            "parallax_error": [1.0, 1.5],
            "phot_g_mean_mag": [10.5, 11.0],
            "phot_bp_mean_mag": [11.2, 11.6],
            "phot_rp_mean_mag": [9.9, 10.2],
            "bp_rp": [1.3, 1.4],
            "teff_gspphot": [4800.0, 4600.0],
            "lum_flame": [0.5, 0.4],
            "radius_flame": [0.9, 0.85],
            "ruwe": [1.1, 1.2],
            "phot_bp_rp_excess_factor": [1.2, 1.3],
        }
    )

    astroquery_module, gaia_module, fake_gaia = _mock_gaia_modules(sample_df)
    monkeypatch.setitem(sys.modules, "astroquery", astroquery_module)
    monkeypatch.setitem(sys.modules, "astroquery.gaia", gaia_module)

    output_file = tmp_path / "gaia_sample.csv"
    monkeypatch.setattr(download, "DATA_OUTPUT", output_file)

    result = download.query_gaia_sample(n_stars=10, max_dist_pc=80)

    assert len(result) == 2
    assert output_file.exists()
    assert "ruwe < 1.4" in fake_gaia.last_query
    assert "LEFT JOIN gaiadr3.astrophysical_parameters" in fake_gaia.last_query
    assert "ap.lum_flame" in fake_gaia.last_query
    assert "ap.radius_flame" in fake_gaia.last_query

    for col in download.REQUIRED_COLUMNS:
        assert col in result.columns


def test_query_gaia_sample_uses_sync_fallback(monkeypatch, tmp_path) -> None:
    sample_df = pd.DataFrame(
        {
            "source_id": [1],
            "ra": [10.0],
            "dec": [5.0],
            "parallax": [20.0],
            "parallax_error": [1.0],
            "phot_g_mean_mag": [10.5],
            "phot_bp_mean_mag": [11.2],
            "phot_rp_mean_mag": [9.9],
            "bp_rp": [1.3],
            "teff_gspphot": [4800.0],
            "lum_flame": [0.5],
            "radius_flame": [0.9],
            "ruwe": [1.1],
            "phot_bp_rp_excess_factor": [1.2],
        }
    )

    astroquery_module, gaia_module = _mock_gaia_modules_async_fails_sync_succeeds(sample_df)
    monkeypatch.setitem(sys.modules, "astroquery", astroquery_module)
    monkeypatch.setitem(sys.modules, "astroquery.gaia", gaia_module)

    output_file = tmp_path / "gaia_sample.csv"
    monkeypatch.setattr(download, "DATA_OUTPUT", output_file)

    result = download.query_gaia_sample(n_stars=1, max_dist_pc=80)

    assert len(result) == 1
    assert output_file.exists()


@pytest.mark.online
def test_query_gaia_sample_online_small() -> None:
    if os.getenv("RUN_ONLINE_TESTS") != "1":
        pytest.skip("Set RUN_ONLINE_TESTS=1 para habilitar test en red real")

    try:
        df = download.query_gaia_sample(n_stars=20, max_dist_pc=25)
    except RuntimeError as exc:
        if _is_transient_gaia_error(str(exc)):
            pytest.skip(f"Servicio Gaia no disponible temporalmente: {exc}")
        raise

    assert not df.empty

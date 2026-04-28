from __future__ import annotations

import os

import pandas as pd
import pytest
import requests

from data import download


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


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


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
            "r_med_geo": [100.0, 120.0],
            "r_lo_geo": [90.0, 110.0],
            "r_hi_geo": [110.0, 130.0],
            "r_med_photogeo": [98.0, 118.0],
            "r_lo_photogeo": [88.0, 108.0],
            "r_hi_photogeo": [108.0, 128.0],
        }
    )

    captured = {}

    def fake_post(url, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["timeout"] = timeout
        return _FakeResponse(sample_df.to_csv(index=False))

    monkeypatch.setattr(download.requests, "post", fake_post)

    output_file = tmp_path / "gaia_sample.csv"
    monkeypatch.setattr(download, "DATA_OUTPUT", output_file)

    result = download.query_gaia_sample(n_stars=10, max_dist_pc=80)

    assert len(result) == 2
    assert output_file.exists()
    assert captured["url"] == download.GAIA_TAP_SYNC_URL
    assert "ruwe < 1.4" in captured["data"]["QUERY"]
    assert "LEFT JOIN gaiadr3.astrophysical_parameters" in captured["data"]["QUERY"]
    assert "LEFT JOIN external.gaiaedr3_distance" in captured["data"]["QUERY"]
    assert "ap.lum_flame" in captured["data"]["QUERY"]
    assert "ap.radius_flame" in captured["data"]["QUERY"]
    assert captured["timeout"] == download.GAIA_REQUEST_TIMEOUT_SECONDS

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
            "r_med_geo": [100.0],
            "r_lo_geo": [90.0],
            "r_hi_geo": [110.0],
            "r_med_photogeo": [98.0],
            "r_lo_photogeo": [88.0],
            "r_hi_photogeo": [108.0],
        }
    )

    calls = {"count": 0}

    def fake_post(url, data=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.Timeout("Read timed out")
        return _FakeResponse(sample_df.to_csv(index=False))

    monkeypatch.setattr(download.requests, "post", fake_post)

    output_file = tmp_path / "gaia_sample.csv"
    monkeypatch.setattr(download, "DATA_OUTPUT", output_file)

    result = download.query_gaia_sample(n_stars=1, max_dist_pc=80)

    assert len(result) == 1
    assert output_file.exists()
    assert calls["count"] == 2


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

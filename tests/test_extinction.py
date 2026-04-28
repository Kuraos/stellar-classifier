from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
from dustmaps.bayestar import BayestarQuery

from src.extinction import apply_extinction_correction


def _write_synthetic_bayestar_map(map_path: Path) -> None:
    """Crea un mapa Bayestar minimo para probar la integracion con dustmaps."""
    pixel_info_dtype = np.dtype(
        [
            ("nside", "i4"),
            ("healpix_index", "i8"),
            ("DM_reliable_min", "f4"),
            ("DM_reliable_max", "f4"),
            ("converged", "?"),
        ]
    )
    pixel_info = np.array([(1, 0, 0.0, 20.0, True)], dtype=pixel_info_dtype)
    samples = np.array([[[0.123]]], dtype="f4")
    best_fit = np.array([[0.123]], dtype="f4")

    with h5py.File(map_path, "w") as h5_file:
        pixel_info_dataset = h5_file.create_dataset("/pixel_info", data=pixel_info)
        pixel_info_dataset.attrs["DM_bin_edges"] = np.array([5.0], dtype="f8")
        h5_file.create_dataset("/samples", data=samples)
        h5_file.create_dataset("/best_fit", data=best_fit)


def test_apply_extinction_correction_adds_expected_columns(sample_processed_df) -> None:
    df = sample_processed_df[
        ["source_id", "ra", "dec", "parallax", "phot_g_mean_mag", "bp_rp", "M_G"]
    ].copy()

    corrected = apply_extinction_correction(df, reddening_query=lambda coords: np.full(len(coords), 0.1))

    expected_columns = {
        "A_V",
        "A_G",
        "E_BR",
        "BP_RP_corr",
        "B_V_corr",
        "teff_corr",
        "M_G_corr",
        "luminosity_solar_corr",
        "spectral_type_corr",
    }
    assert expected_columns.issubset(corrected.columns)

    assert np.allclose(corrected["A_V"], 0.31)
    assert np.allclose(corrected["A_G"], 0.31 * 0.789)
    assert np.allclose(corrected["E_BR"], 0.31 * 0.415)
    assert np.allclose(corrected["BP_RP_corr"], corrected["bp_rp"] - corrected["E_BR"])
    assert np.allclose(corrected["M_G_corr"], corrected["M_G"])
    assert np.allclose(corrected["B_V"], corrected["B_V_corr"])
    assert np.allclose(corrected["teff"], corrected["teff_corr"])
    assert np.allclose(corrected["luminosity_solar"], corrected["luminosity_solar_corr"])
    assert (corrected["spectral_type"] == corrected["spectral_type_corr"]).all()


def test_bayestar_query_executes_with_synthetic_map(tmp_path) -> None:
    map_path = tmp_path / "bayestar2019.h5"
    _write_synthetic_bayestar_map(map_path)

    query = BayestarQuery(map_fname=str(map_path), version="bayestar2019")
    healpix = HEALPix(nside=1, order="nested", frame="galactic")
    center = healpix.healpix_to_skycoord(0)
    coord = SkyCoord(l=center.l, b=center.b, distance=100 * u.pc, frame="galactic")

    reddening = np.atleast_1d(query(coord, mode="median"))
    assert np.isclose(reddening[0], 0.123, atol=1e-6)

    icrs = coord.icrs
    df = pd.DataFrame(
        {
            "source_id": [1],
            "ra": [icrs.ra.deg],
            "dec": [icrs.dec.deg],
            "parallax": [10.0],
            "phot_g_mean_mag": [10.0],
            "bp_rp": [1.0],
        }
    )

    corrected = apply_extinction_correction(
        df,
        reddening_query=lambda coords: query(coords, mode="median"),
    )

    assert np.isclose(corrected["A_V"].iloc[0], 0.123 * 3.1, atol=1e-6)
    assert corrected["spectral_type"].iloc[0] in {"G", "K"}
    assert np.isfinite(corrected["M_G_corr"].iloc[0])


def test_apply_extinction_correction_high_latitude_keeps_av_low() -> None:
    df = pd.DataFrame(
        {
            "source_id": [1],
            "ra": [192.85948],
            "dec": [27.12825],
            "parallax": [20.0],
            "phot_g_mean_mag": [10.0],
            "bp_rp": [0.9],
        }
    )

    def fake_query(coords):
        latitude = np.abs(np.asarray(coords.b.degree, dtype=float))
        return np.where(latitude > 60.0, 0.02, 0.2)

    corrected = apply_extinction_correction(df, reddening_query=fake_query)

    assert corrected["A_V"].iloc[0] < 0.1
    assert corrected["A_G"].iloc[0] < 0.1
    assert corrected["luminosity_solar"].iloc[0] > 0.0
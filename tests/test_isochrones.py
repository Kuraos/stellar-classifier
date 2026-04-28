from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src.isochrones import (
    chi_squared_isochrone,
    filter_evolutionary_phases,
    fit_best_age,
    isochrone_to_observables,
    list_available_isochrones,
    load_isochrone,
    parse_isochrone_file,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "isochrones"


def _copy_fixtures(tmp_path: Path, names: list[str]) -> Path:
    for name in names:
        shutil.copy2(FIXTURE_DIR / name, tmp_path / name)
    return tmp_path


def test_parse_isochrone_format_padova() -> None:
    df = parse_isochrone_file(str(FIXTURE_DIR / "iso_logage_8.0_9.0_combo.dat"))
    assert {"log_age", "logTe", "Gmag"}.issubset(df.columns)
    assert len(df) == 6


def test_parse_isochrone_handles_multiple_ages() -> None:
    df = parse_isochrone_file(str(FIXTURE_DIR / "iso_logage_8.0_9.0_combo.dat"))
    assert set(np.round(df["log_age"].unique(), 2)) == {8.0, 8.1}
    assert len(df) == 6


def test_load_isochrone_by_log_age(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path, ["iso_logage_8.0_mh_0.0.dat", "iso_logage_9.0_mh_0.0.dat"])
    df = load_isochrone(8.0, metallicity=0.0, isochrones_dir=str(tmp_path))
    assert not df.empty
    assert np.allclose(df["log_age"].astype(float).unique(), [8.0])


def test_load_isochrone_tolerance(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path, ["iso_logage_8.0_mh_0.0.dat", "iso_logage_9.0_mh_0.0.dat"])
    df = load_isochrone(8.03, metallicity=0.0, isochrones_dir=str(tmp_path))
    assert not df.empty
    assert np.allclose(df["log_age"].astype(float).unique(), [8.0])


def test_load_isochrone_not_found(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path, ["iso_logage_8.0_mh_0.0.dat"])
    with pytest.raises(FileNotFoundError) as exc:
        load_isochrone(11.0, metallicity=0.0, isochrones_dir=str(tmp_path))
    assert "Disponibles" in str(exc.value)


def test_filter_phases() -> None:
    df = pd.DataFrame({"label": [0, 1, 2, 3, 4, 5, 6, 7, 8], "logTe": np.linspace(3.6, 3.9, 9)})
    filtered = filter_evolutionary_phases(df, phases=(1, 2, 3))
    assert set(filtered["label"].tolist()) == {1, 2, 3}


def test_isochrone_to_observables_distance_modulus() -> None:
    df = pd.DataFrame({
        "logTe": [3.76],
        "Gmag": [5.0],
        "G_BPmag": [5.5],
        "G_RPmag": [4.7],
    })
    obs_10 = isochrone_to_observables(df, distance_pc=10.0)
    obs_100 = isochrone_to_observables(df, distance_pc=100.0)
    assert np.isclose(obs_10["m_G_iso"].iloc[0], 5.0)
    assert np.isclose(obs_100["m_G_iso"].iloc[0], 10.0)


def test_chi_squared_zero_for_perfect_match(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path, ["iso_logage_8.0_mh_0.0.dat"])
    iso = load_isochrone(8.0, metallicity=0.0, isochrones_dir=str(tmp_path))
    observed = pd.DataFrame({
        "teff": 10.0 ** iso["logTe"].to_numpy(dtype=float),
        "M_G": iso["Gmag"].to_numpy(dtype=float),
    })
    chi2 = chi_squared_isochrone(observed, iso, use_corrected=False, use_bayesian=False)
    assert np.isclose(chi2, 0.0)


def test_fit_best_age_recovers_input(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path, ["iso_logage_8.0_mh_0.0.dat", "iso_logage_9.0_mh_0.0.dat"])
    iso = load_isochrone(8.0, metallicity=0.0, isochrones_dir=str(tmp_path))
    observed = pd.DataFrame({
        "teff": 10.0 ** iso["logTe"].to_numpy(dtype=float),
        "M_G": iso["Gmag"].to_numpy(dtype=float),
    })
    result = fit_best_age(
        observed,
        age_grid=np.arange(7.0, 9.1, 0.1),
        metallicity=0.0,
        isochrones_dir=str(tmp_path),
    )
    assert np.isclose(result["best_log_age"], 8.0, atol=0.1)
    assert isinstance(result["best_isochrone"], pd.DataFrame)
    assert not result["best_isochrone"].empty


def test_solar_isochrone_passes_near_sun() -> None:
    df = parse_isochrone_file(str(FIXTURE_DIR / "iso_logage_9.6_mh_0.0.dat"))
    filtered = filter_evolutionary_phases(df)
    near_sun = filtered[
        filtered["Gmag"].between(4.63, 5.03)
        & filtered["logTe"].between(3.74, 3.78)
    ]
    assert not near_sun.empty

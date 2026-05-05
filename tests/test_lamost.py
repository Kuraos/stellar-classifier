"""Tests para src/lamost.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.io import fits

from src.lamost import (
    SPECTRAL_LINES,
    analyse_star_spectrum,
    batch_analyse_spectra,
    load_spectrum_from_cache,
    measure_equivalent_widths,
    spectral_type_from_ew,
    teff_from_ew_h_alpha,
)


# ---------------------------------------------------------------------
# Helpers para crear espectros sinteticos
# ---------------------------------------------------------------------

def _make_synthetic_spectrum(
    teff: float = 5800.0,
    n_points: int = 3000,
    wave_min: float = 3800.0,
    wave_max: float = 7000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crea un espectro sintetico con lineas de absorcion gaussianas."""
    wavelength = np.linspace(wave_min, wave_max, n_points)
    flux = np.ones(n_points)

    # Profundidad aproximada de H-alpha segun T_eff (maximo en A0)
    t_norm = (teff - 3500.0) / (10000.0 - 3500.0)
    ha_depth = 0.5 * np.exp(-((t_norm - 0.5) ** 2) / 0.1) + 0.05

    for line_name, center in SPECTRAL_LINES.items():
        if wave_min <= center <= wave_max:
            sigma = 3.0
            depth = ha_depth if "alpha" in line_name.lower() else ha_depth * 0.6
            flux -= depth * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)

    rng = np.random.default_rng(42)
    flux += rng.normal(0, 0.01, n_points)
    flux = np.clip(flux, 0.1, 1.5)
    return wavelength, flux


def _write_synthetic_fits(
    path: Path,
    wavelength: np.ndarray,
    flux: np.ndarray,
) -> None:
    """Escribe un FITS minimo en formato compatible con el parser."""
    log_wave = np.log10(wavelength)
    data = np.zeros((4, len(flux)), dtype=np.float32)
    data[0] = flux.astype(np.float32)
    data[2] = log_wave.astype(np.float32)
    hdu = fits.PrimaryHDU(data=data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=True)


# ---------------------------------------------------------------------
# Tests de spectral_type_from_ew
# ---------------------------------------------------------------------

def test_spectral_type_from_ew_A() -> None:
    assert spectral_type_from_ew(11.0) == "A"


def test_spectral_type_from_ew_F() -> None:
    assert spectral_type_from_ew(7.0) == "F"


def test_spectral_type_from_ew_G() -> None:
    assert spectral_type_from_ew(3.0) == "G"


def test_spectral_type_from_ew_K() -> None:
    assert spectral_type_from_ew(1.0) == "K"


def test_spectral_type_from_ew_M() -> None:
    assert spectral_type_from_ew(0.2) == "M"


def test_spectral_type_from_ew_nan_returns_unknown() -> None:
    assert spectral_type_from_ew(float("nan")) == "?"


# ---------------------------------------------------------------------
# Tests de teff_from_ew_h_alpha
# ---------------------------------------------------------------------

def test_teff_from_ew_h_alpha_solar_value() -> None:
    """W_Halpha ~ 3 A (G2V solar) debe dar T_eff cercano al solar."""
    teff = teff_from_ew_h_alpha(3.0)
    assert np.isfinite(teff)
    assert 5000 < teff < 6500


def test_teff_from_ew_h_alpha_increasing() -> None:
    """T_eff crece con W_Halpha en el lado frio (FGK)."""
    t_k = teff_from_ew_h_alpha(1.0)
    t_g = teff_from_ew_h_alpha(3.0)
    t_f = teff_from_ew_h_alpha(7.0)
    t_a = teff_from_ew_h_alpha(11.0)
    assert t_k < t_g < t_f < t_a
    assert 4000 < t_k < 5500
    assert 5000 < t_g < 6500
    assert 6500 < t_f < 8000
    assert 8500 < t_a < 10500


def test_teff_from_ew_h_alpha_out_of_range() -> None:
    """W fuera de rango debe devolver NaN."""
    assert np.isnan(teff_from_ew_h_alpha(-1.0))
    assert np.isnan(teff_from_ew_h_alpha(15.0))
    assert np.isnan(teff_from_ew_h_alpha(float("nan")))


# ---------------------------------------------------------------------
# Tests de measure_equivalent_widths
# ---------------------------------------------------------------------

def test_measure_equivalent_widths_detects_h_alpha() -> None:
    """Espectro solar sintetico debe tener H-alpha ajustada."""
    wavelength, flux = _make_synthetic_spectrum(teff=5800.0)
    result = measure_equivalent_widths(wavelength, flux)
    assert "H_alpha" in result
    ha = result["H_alpha"]
    assert ha["fitted"] is True
    assert 0.5 < ha["EW"] < 8.0


def test_measure_equivalent_widths_line_out_of_range() -> None:
    """Lineas fuera del rango deben devolver fitted=False y EW NaN."""
    wavelength, flux = _make_synthetic_spectrum(wave_min=5000.0, wave_max=7000.0)
    result = measure_equivalent_widths(wavelength, flux)
    assert result["Ca_II_K"]["fitted"] is False
    assert np.isnan(result["Ca_II_K"]["EW"])


def test_measure_equivalent_widths_returns_all_lines() -> None:
    """El resultado debe incluir todas las lineas definidas."""
    wavelength, flux = _make_synthetic_spectrum()
    result = measure_equivalent_widths(wavelength, flux)
    for line in SPECTRAL_LINES:
        assert line in result


# ---------------------------------------------------------------------
# Tests de load_spectrum_from_cache y download_spectrum
# ---------------------------------------------------------------------

def test_load_spectrum_from_cache_returns_none_when_missing(tmp_path) -> None:
    result = load_spectrum_from_cache(99999, cache_dir=tmp_path)
    assert result is None


def test_load_spectrum_from_cache_reads_existing_fits(tmp_path) -> None:
    wavelength, flux = _make_synthetic_spectrum()
    fits_path = tmp_path / "spec_12345.fits"
    _write_synthetic_fits(fits_path, wavelength, flux)

    result = load_spectrum_from_cache(12345, cache_dir=tmp_path)
    assert result is not None
    loaded_wave, loaded_flux = result
    assert len(loaded_wave) == len(wavelength)
    assert np.all(np.isfinite(loaded_wave))
    assert np.all(np.isfinite(loaded_flux))


def test_load_spectrum_from_cache_reads_lamost_bintable_format(tmp_path) -> None:
    """Valida el parser sobre un FITS con estructura BinTable real de LAMOST."""
    wave, flux = _make_synthetic_spectrum(teff=5800.0)
    log_wave = np.log10(wave).astype(np.float32)
    flux_f = flux.astype(np.float32)

    col_flux = fits.Column(
        name="FLUX", format=f"{len(flux_f)}E", array=np.array([flux_f]),
    )
    col_wave = fits.Column(
        name="WAVELENGTH", format=f"{len(log_wave)}E", array=np.array([log_wave]),
    )
    hdu_primary = fits.PrimaryHDU()
    hdu_table = fits.BinTableHDU.from_columns([col_flux, col_wave])
    hdul = fits.HDUList([hdu_primary, hdu_table])

    fits_path = tmp_path / "spec_77.fits"
    hdul.writeto(fits_path, overwrite=True)

    result = load_spectrum_from_cache(77, cache_dir=tmp_path)
    assert result is not None
    loaded_wave, loaded_flux = result
    assert len(loaded_wave) == len(wave)
    assert np.all(np.isfinite(loaded_wave))
    assert 3000 < float(np.nanmin(loaded_wave)) < 8000
    assert np.all(np.isfinite(loaded_flux))


def test_download_spectrum_uses_cache_when_available(tmp_path) -> None:
    """Si existe cache local, no debe invocar red."""
    from src.lamost import download_spectrum

    wavelength, flux = _make_synthetic_spectrum()
    fits_path = tmp_path / "spec_12345.fits"
    _write_synthetic_fits(fits_path, wavelength, flux)

    with patch("src.lamost.requests.get") as mock_get:
        result = download_spectrum(12345, cache_dir=tmp_path)
        mock_get.assert_not_called()

    assert result is not None


def test_download_spectrum_returns_none_on_network_error(tmp_path) -> None:
    """Error de red debe devolver None sin lanzar excepcion."""
    from src.lamost import download_spectrum
    import requests

    with patch("src.lamost.requests.get", side_effect=requests.ConnectionError("sin red")):
        result = download_spectrum(99999, cache_dir=tmp_path)
    assert result is None


# ---------------------------------------------------------------------
# Tests de analyse_star_spectrum
# ---------------------------------------------------------------------

def test_analyse_star_spectrum_success(tmp_path) -> None:
    """Pipeline completo usando un espectro sintetico en cache."""
    wavelength, flux = _make_synthetic_spectrum(teff=5800.0)
    fits_path = tmp_path / "spec_42.fits"
    _write_synthetic_fits(fits_path, wavelength, flux)

    result = analyse_star_spectrum(
        source_id=1001,
        obsid=42,
        cache_dir=tmp_path,
        teff_photometric=5800.0,
    )

    assert result["success"] is True
    assert result["error"] is None
    assert result["spectral_type_spec"] in {"A", "B", "F", "G", "K", "M", "?"}
    assert np.isfinite(result["teff_spectroscopic"])
    assert result["teff_diff_K"] is not None
    assert "H_alpha" in result["equivalent_widths"]


def test_analyse_star_spectrum_missing_spectrum(tmp_path) -> None:
    """Si no hay cache y falla red, success=False."""
    with patch("src.lamost.requests.get", side_effect=Exception("sin red")):
        result = analyse_star_spectrum(
            source_id=9999,
            obsid=99999,
            cache_dir=tmp_path,
        )
    assert result["success"] is False
    assert result["error"] is not None


# ---------------------------------------------------------------------
# Tests de batch_analyse_spectra
# ---------------------------------------------------------------------

def test_batch_analyse_spectra_returns_dataframe(tmp_path) -> None:
    """batch_analyse_spectra devuelve DataFrame con columnas esperadas."""
    for obsid in [1, 2, 3]:
        wave, flux = _make_synthetic_spectrum(teff=5500.0 + obsid * 200)
        _write_synthetic_fits(tmp_path / f"spec_{obsid}.fits", wave, flux)

    df_crossmatch = pd.DataFrame({
        "source_id": [101, 102, 103],
        "obsid": [1, 2, 3],
    })
    df_processed = pd.DataFrame({
        "source_id": [101, 102, 103],
        "teff": [5500.0, 5700.0, 5900.0],
    })

    result = batch_analyse_spectra(
        df_crossmatch,
        df_processed,
        cache_dir=tmp_path,
        max_spectra=3,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    for col in ["source_id", "obsid", "spectral_type_spec", "teff_spectroscopic", "success"]:
        assert col in result.columns


def test_batch_analyse_spectra_calls_progress(tmp_path) -> None:
    """El callback de progreso se llama al menos una vez por espectro."""
    wave, flux = _make_synthetic_spectrum()
    _write_synthetic_fits(tmp_path / "spec_1.fits", wave, flux)

    df_crossmatch = pd.DataFrame({"source_id": [1], "obsid": [1]})
    df_processed = pd.DataFrame({"source_id": [1], "teff": [5800.0]})

    calls = []

    def fake_progress(n_done, n_total):
        calls.append((n_done, n_total))

    batch_analyse_spectra(
        df_crossmatch,
        df_processed,
        cache_dir=tmp_path,
        max_spectra=1,
        progress_callback=fake_progress,
    )
    assert len(calls) >= 1

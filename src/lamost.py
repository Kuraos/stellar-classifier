"""Integracion de espectros LAMOST DR9 para clasificacion estelar.

Este modulo centraliza el acceso, cache y analisis espectral basico de
estrellas con observaciones en LAMOST, incluyendo medicion de lineas de
absorcion diagnosticas y estimaciones simples de tipo espectral y T_eff.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import gzip
import warnings

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

try:
    from astroquery.vizier import Vizier
except Exception:  # pragma: no cover - entorno sin astroquery
    Vizier = None

from src.line_fitting import fit_absorption_line


LAMOST_VIZIER_CATALOG = "V/164"  # LAMOST DR9 en Vizier

# Longitudes de onda de referencia en Angstroms (vacio)
SPECTRAL_LINES: dict[str, float] = {
    "H_alpha": 6562.8,
    "H_beta": 4861.3,
    "H_gamma": 4340.5,
    "Ca_II_K": 3933.7,
    "Ca_II_H": 3968.5,
    "Mg_I_b": 5183.6,
    "Na_I_D": 5895.9,
}

# Ventana de ajuste por linea (semiancho en A)
LINE_WINDOWS: dict[str, float] = {
    "H_alpha": 15.0,
    "H_beta": 12.0,
    "H_gamma": 12.0,
    "Ca_II_K": 10.0,
    "Ca_II_H": 10.0,
    "Mg_I_b": 8.0,
    "Na_I_D": 8.0,
}

SPECTRA_CACHE_DIR = Path("data/spectra")


def _normalize_obsid(obsid: int | str | float) -> str:
    """Normaliza obsid para URL/cache evitando sufijos '.0' en enteros."""
    try:
        as_float = float(obsid)
        if np.isfinite(as_float) and as_float.is_integer():
            return str(int(as_float))
    except Exception:
        pass
    return str(obsid).strip()


def _empty_crossmatch_df() -> pd.DataFrame:
    """Devuelve el DataFrame vacio con columnas contractuales de cross-match."""
    return pd.DataFrame(
        columns=[
            "source_id",
            "obsid",
            "ra_lamost",
            "dec_lamost",
            "snrg",
            "snrr",
            "class_lamost",
            "subclass_lamost",
        ]
    )


def _pick_column(row: pd.Series, candidates: list[str]) -> object:
    """Selecciona el primer campo existente y finito entre varios alias."""
    for name in candidates:
        if name in row.index:
            value = row.get(name)
            if pd.notna(value):
                return value
    return np.nan


def _table_to_dataframe(table: Table) -> pd.DataFrame:
    """Convierte tabla astropy a pandas minimizando efectos de masked arrays."""
    try:
        return table.filled(np.nan).to_pandas()
    except Exception:
        return table.to_pandas()


def crossmatch_lamost(
    df: pd.DataFrame,
    radius_arcsec: float = 2.0,
    max_stars: int = 500,
) -> pd.DataFrame:
    """Busca espectros LAMOST DR9 para las estrellas del DataFrame.

    Usa astroquery.vizier para buscar en el catalogo V/164 por coordenadas
    (ra, dec) con radio radius_arcsec.

    Solo procesa las primeras max_stars estrellas para evitar timeouts en
    Vizier con muestras grandes.

    Devuelve un DataFrame con columnas:
        source_id, obsid, ra_lamost, dec_lamost,
        snrg, snrr, class_lamost, subclass_lamost

    Si no hay coincidencias, devuelve DataFrame vacio con esas columnas.
    Maneja errores de red silenciosamente (devuelve vacio + warning).

    Referencia: Zhao et al. (2012), RAA, 12, 723 (catalogo LAMOST).
    """
    if df is None or df.empty or "ra" not in df.columns or "dec" not in df.columns:
        return _empty_crossmatch_df()

    radius_arcsec = float(radius_arcsec)
    if radius_arcsec <= 0:
        radius_arcsec = 2.0

    max_n = int(max_stars) if max_stars is not None else 500
    if max_n <= 0:
        max_n = 500

    if Vizier is None:
        warnings.warn("crossmatch_lamost: astroquery no disponible en este entorno")
        return _empty_crossmatch_df()

    subset = df.head(max_n).copy()
    vizier = Vizier(columns=["*"], row_limit=20)

    rows: list[dict[str, object]] = []
    n_errors = 0
    for _, star in subset.iterrows():
        source_id = star.get("source_id")
        ra = star.get("ra")
        dec = star.get("dec")
        try:
            ra_f = float(ra)
            dec_f = float(dec)
        except Exception:
            continue

        if not np.isfinite(ra_f) or not np.isfinite(dec_f):
            continue

        coord = SkyCoord(ra=ra_f * u.deg, dec=dec_f * u.deg, frame="icrs")
        result = None
        last_exc: Exception | None = None
        # Reintento corto para amortiguar cortes transitorios de Vizier.
        for _attempt in range(2):
            try:
                result = vizier.query_region(
                    coord,
                    radius=radius_arcsec * u.arcsec,
                    catalog=LAMOST_VIZIER_CATALOG,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            n_errors += 1
            continue

        if not result:
            continue

        table = result[0]
        match_df = _table_to_dataframe(table)
        if match_df.empty:
            continue

        # Elegir la coincidencia mas cercana si existe columna de separacion.
        if "_r" in match_df.columns:
            chosen = match_df.sort_values("_r", ascending=True).iloc[0]
        else:
            chosen = match_df.iloc[0]

        rows.append(
            {
                "source_id": source_id,
                "obsid": _pick_column(chosen, ["obsid", "ObsID", "OBSID", "obsid_1"]),
                "ra_lamost": _pick_column(chosen, ["RAJ2000", "RA_ICRS", "RAdeg", "RA"]),
                "dec_lamost": _pick_column(chosen, ["DEJ2000", "DE_ICRS", "DEdeg", "DEC"]),
                "snrg": _pick_column(chosen, ["snrg", "SNRG", "snr_g", "SNR_G"]),
                "snrr": _pick_column(chosen, ["snrr", "SNRR", "snr_r", "SNR_R"]),
                "class_lamost": _pick_column(chosen, ["class", "Class", "objType"]),
                "subclass_lamost": _pick_column(chosen, ["subclass", "SubClass", "spType"]),
            }
        )

    if n_errors > 0:
        warnings.warn(
            "crossmatch_lamost: "
            f"{n_errors} consultas fallaron por red/Vizier; se devolvieron coincidencias parciales"
        )

    if not rows:
        return _empty_crossmatch_df()

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["obsid"]).copy()
    if out.empty:
        return _empty_crossmatch_df()
    return out.reset_index(drop=True)


def _normalize_flux(flux: np.ndarray) -> np.ndarray:
    """Normaliza el flujo a continuo ~1 usando la region de alto percentil."""
    y = np.asarray(flux, dtype=float)
    mask = np.isfinite(y)
    if not mask.any():
        return y

    y_valid = y[mask]
    p90 = np.nanpercentile(y_valid, 90)
    high = y_valid[y_valid >= p90]
    if high.size == 0:
        scale = np.nanmedian(y_valid)
    else:
        scale = np.nanmedian(high)

    if not np.isfinite(scale) or scale == 0:
        scale = 1.0

    return y / scale


def _read_lamost_fits(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Lee FITS LAMOST y devuelve (wavelength_A, flux_norm) o None si falla."""
    try:
        with fits.open(path) as hdul:
            flux_data: np.ndarray | None = None
            wave_log: np.ndarray | None = None

            # Caso preferido: LAMOST con HDU[1] y filas [0]=flux, [2]=log10(lambda)
            if len(hdul) > 1 and hdul[1].data is not None:
                data1 = hdul[1].data
                names = [str(n).upper() for n in getattr(data1, "names", [])] if hasattr(data1, "names") else []

                # Formato BinTable comun en LAMOST DR9: columnas FLUX y WAVELENGTH.
                if "FLUX" in names and "WAVELENGTH" in names and len(data1) > 0:
                    flux_data = np.asarray(data1[0]["FLUX"], dtype=float)
                    wave_vals = np.asarray(data1[0]["WAVELENGTH"], dtype=float)
                    if np.nanmedian(wave_vals) < 100.0:
                        wave_log = wave_vals
                    else:
                        wave_log = None
                        # Marcamos con variable temporal usando wave_vals directas.
                        wave = wave_vals
                        flux = _normalize_flux(flux_data)
                        mask = np.isfinite(wave) & np.isfinite(flux)
                        wave = wave[mask]
                        flux = flux[mask]
                        if wave.size >= 10 and flux.size >= 10:
                            order = np.argsort(wave)
                            return wave[order], flux[order]

                # Formato alternativo por filas: fila 0=flux, fila 2=log10(lambda)
                if flux_data is None or wave_log is None:
                    arr = np.asarray(data1)
                    if arr.ndim >= 2 and arr.shape[0] > 2:
                        flux_data = np.asarray(arr[0], dtype=float)
                        wave_log = np.asarray(arr[2], dtype=float)

            # Fallback para fixtures sinteticos: PrimaryHDU con forma (4, n)
            if flux_data is None or wave_log is None:
                if hdul[0].data is None:
                    return None
                arr0 = np.asarray(hdul[0].data)
                if arr0.ndim >= 2 and arr0.shape[0] > 2:
                    flux_data = np.asarray(arr0[0], dtype=float)
                    wave_log = np.asarray(arr0[2], dtype=float)
                elif arr0.ndim == 1:
                    # Caso limite: sin eje de onda explicito.
                    return None

            if flux_data is None or wave_log is None:
                return None

            wave = np.power(10.0, wave_log, dtype=float)
            flux = _normalize_flux(flux_data)

            mask = np.isfinite(wave) & np.isfinite(flux)
            wave = wave[mask]
            flux = flux[mask]
            if wave.size < 10 or flux.size < 10:
                return None

            order = np.argsort(wave)
            return wave[order], flux[order]
    except Exception as exc:
        warnings.warn(f"load_spectrum_from_cache: FITS corrupto o invalido: {exc}")
        return None


def load_spectrum_from_cache(
    obsid: int | str,
    cache_dir: str | Path = SPECTRA_CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Carga un espectro desde el cache local sin acceder a la red.

    Devuelve (wavelength, flux) o None si no esta en cache.
    """
    cache_path = Path(cache_dir)
    obsid_norm = _normalize_obsid(obsid)
    fits_path = cache_path / f"spec_{obsid_norm}.fits"
    if not fits_path.exists():
        return None
    return _read_lamost_fits(fits_path)


def _looks_like_fits_bytes(payload: bytes) -> bool:
    """Valida si un payload parece FITS o FITS comprimido en gzip."""
    if not payload:
        return False

    # FITS sin comprimir inicia con cabecera SIMPLE.
    if payload[:6] == b"SIMPLE":
        return True

    # FITS comprimido: magic gzip 1f 8b
    if payload[:2] == b"\x1f\x8b":
        try:
            head = gzip.decompress(payload[:8192]) if len(payload) <= 8192 else gzip.decompress(payload)
            return head[:6] == b"SIMPLE"
        except Exception:
            return False

    return False


def _decode_fits_payload(payload: bytes) -> bytes:
    """Devuelve bytes FITS sin comprimir cuando el payload llega en gzip."""
    if payload[:2] == b"\x1f\x8b":
        return gzip.decompress(payload)
    return payload


def download_spectrum(
    obsid: int | str,
    cache_dir: str | Path = SPECTRA_CACHE_DIR,
    timeout: int = 60,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Descarga un espectro LAMOST por obsid desde el servidor publico.

    URL base: https://www.lamost.org/dr9/api/spectra/fits/{obsid}
    Guarda en cache_dir/spec_{obsid}.fits.
    Si ya existe en cache, lo carga directamente (no descarga de nuevo).

    Devuelve (wavelength_angstrom, flux_normalized) o None si falla.

    El FITS de LAMOST tiene:
    - HDU[0]: datos sin usar
    - HDU[1]: flux en multiples filas (fila 0 = flux, fila 2 = wavelength)
    - La longitud de onda esta en log10(A): wavelength = 10^(array)

    Normaliza el flujo dividiendo por la mediana de la region del continuo
    (percentil 90) para obtener un espectro relativo con continuo ~1.

    Maneja errores de red y FITS corruptos devolviendo None + warning.

    Referencia: LAMOST DR9 data model documentation.
    """
    obsid_norm = _normalize_obsid(obsid)
    cached = load_spectrum_from_cache(obsid_norm, cache_dir=cache_dir)
    if cached is not None:
        return cached

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fits_path = cache_path / f"spec_{obsid_norm}.fits"

    # Endpoint principal solicitado + fallback observado operativo en DR9.
    url_candidates = [
        f"https://www.lamost.org/dr9/api/spectra/fits/{obsid_norm}",
        f"https://www.lamost.org/dr9/spectrum/fits/{obsid_norm}",
    ]

    response_content: bytes | None = None
    last_exc: Exception | None = None
    for url in url_candidates:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            content = response.content
            if not _looks_like_fits_bytes(content):
                last_exc = ValueError("respuesta no FITS (posible HTML de error)")
                continue
            response_content = _decode_fits_payload(content)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc

    if response_content is None:
        warnings.warn(
            "download_spectrum: no se pudo descargar obsid="
            f"{obsid_norm}: {last_exc}"
        )
        return None

    try:
        fits_path.write_bytes(response_content)
    except Exception as exc:
        warnings.warn(f"download_spectrum: no se pudo escribir cache obsid={obsid_norm}: {exc}")
        return None

    parsed = _read_lamost_fits(fits_path)
    if parsed is None:
        warnings.warn(f"download_spectrum: FITS invalido para obsid={obsid_norm}")
    return parsed


def measure_equivalent_widths(
    wavelength: np.ndarray,
    flux: np.ndarray,
    lines: dict[str, float] | None = None,
    windows: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Mide anchos equivalentes de las lineas espectrales diagnosticas.

    Usa fit_absorption_line de src/line_fitting.py para cada linea.
    Solo intenta ajustar lineas dentro del rango del espectro.

    Devuelve dict con una entrada por linea:
    {
        "H_alpha": {
            "EW": float,
            "EW_err": float,
            "center": float,
            "depth": float,
            "fwhm": float,
            "fitted": bool,
        },
        ...
    }

    Si una linea no esta en el rango del espectro o el ajuste falla,
    devuelve fitted=False y EW=NaN para esa linea.

    Reutiliza src/line_fitting.fit_absorption_line.
    Referencia: Gray (2008), The Observation and Analysis of Stellar Photospheres.
    """
    line_map = lines or SPECTRAL_LINES
    window_map = windows or LINE_WINDOWS

    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(flux, dtype=float)
    results: dict[str, dict] = {}

    if x.size == 0 or y.size == 0:
        for name in line_map:
            results[name] = {
                "EW": np.nan,
                "EW_err": np.nan,
                "center": np.nan,
                "depth": np.nan,
                "fwhm": np.nan,
                "fitted": False,
            }
        return results

    wmin = float(np.nanmin(x))
    wmax = float(np.nanmax(x))

    for line_name, center in line_map.items():
        window = float(window_map.get(line_name, 10.0))
        base = {
            "EW": np.nan,
            "EW_err": np.nan,
            "center": np.nan,
            "depth": np.nan,
            "fwhm": np.nan,
            "fitted": False,
        }

        if center < wmin or center > wmax:
            results[line_name] = base
            continue

        try:
            fit = fit_absorption_line(x, y, center_guess=center, window=window)
            ew = float(fit.get("equivalent_width", np.nan))
            perr = fit.get("parameter_errors", {})
            sigma_err = float(perr.get("sigma", np.nan)) if isinstance(perr, dict) else np.nan
            depth = float(fit.get("depth", np.nan))
            continuum = float(fit.get("continuum", np.nan))
            if np.isfinite(depth) and np.isfinite(sigma_err) and np.isfinite(continuum):
                ew_err = abs(np.sqrt(2.0 * np.pi) * depth * sigma_err)
            else:
                ew_err = np.nan

            results[line_name] = {
                "EW": ew,
                "EW_err": ew_err,
                "center": float(fit.get("center", np.nan)),
                "depth": depth,
                "fwhm": float(fit.get("fwhm", np.nan)),
                "fitted": bool(np.isfinite(ew)),
            }
        except Exception:
            results[line_name] = base

    return results


def spectral_type_from_ew(
    ew_h_alpha: float,
    ew_ca_k: float | None = None,
) -> str:
    """Estima el tipo espectral de Harvard desde el ancho equivalente de H-alpha.

    Clasificacion basada en la curva de intensidad de Balmer:
        W_Halpha > 10 A  -> "A"
        5 < W_Halpha <= 10 -> "F"
        2 < W_Halpha <= 5  -> "G"
        0.5 < W_Halpha <= 2 -> "K"
        W_Halpha <= 0.5   -> "M"
        W_Halpha NaN o ajuste fallido -> "?"

    Si ew_ca_k esta disponible y es consistente con Halpha, refinar:
        Ca II K fuerte (EW > 5 A) con Halpha debil confirma K o M.

    Devuelve string de un caracter: "O","B","A","F","G","K","M","?".

    Referencia: Gray (2008), Cap. 8.
    """
    try:
        w = float(ew_h_alpha)
    except Exception:
        return "?"

    if not np.isfinite(w):
        return "?"

    if w > 10.0:
        spt = "A"
    elif w > 5.0:
        spt = "F"
    elif w > 2.0:
        spt = "G"
    elif w > 0.5:
        spt = "K"
    else:
        spt = "M"

    if ew_ca_k is not None:
        try:
            ca = float(ew_ca_k)
        except Exception:
            ca = np.nan
        if np.isfinite(ca) and ca > 5.0 and w <= 2.0:
            spt = "M" if w <= 0.5 else "K"

    return spt


def teff_from_ew_h_alpha(ew_h_alpha: float) -> float:
    """Estima T_eff desde el ancho equivalente de H-alpha.

    Relacion empirica simplificada (Gray 2008, valida para FGK):
        T_eff ~= 9000 * exp(-0.18 * W_Halpha) + 3500   [K]

    Devuelve NaN si W_Halpha es NaN, negativo o > 15 A (fuera de rango).

    Referencia: Gray (2008), The Observation and Analysis of Stellar Photospheres.
    """
    try:
        w = float(ew_h_alpha)
    except Exception:
        return float("nan")

    if not np.isfinite(w) or w < 0.0 or w > 15.0:
        return float("nan")

    return float(9000.0 * np.exp(-0.18 * w) + 3500.0)


def analyse_star_spectrum(
    source_id: int | str,
    obsid: int | str,
    cache_dir: str | Path = SPECTRA_CACHE_DIR,
    teff_photometric: float | None = None,
) -> dict:
    """Pipeline completo de analisis espectral para una estrella.

    Pasos:
    1. Cargar espectro (cache o descarga).
    2. Medir anchos equivalentes con measure_equivalent_widths.
    3. Estimar tipo espectral con spectral_type_from_ew.
    4. Estimar T_eff espectroscopica con teff_from_ew_h_alpha.
    5. Si se pasa teff_photometric, calcular la diferencia.

    Devuelve dict con resultados y banderas de exito/error.

    Si la descarga o el ajuste fallan, success=False y error describe el motivo.
    Nunca lanza excepcion; maneja todos los errores internamente.
    """
    base = {
        "source_id": source_id,
        "obsid": obsid,
        "wavelength": np.array([], dtype=float),
        "flux": np.array([], dtype=float),
        "equivalent_widths": {},
        "spectral_type_spec": "?",
        "teff_spectroscopic": float("nan"),
        "teff_photometric": teff_photometric,
        "teff_diff_K": None,
        "teff_diff_pct": None,
        "success": False,
        "error": None,
    }

    try:
        spectrum = download_spectrum(obsid=obsid, cache_dir=cache_dir)
        if spectrum is None:
            base["error"] = "no se pudo cargar/descargar el espectro"
            return base

        wavelength, flux = spectrum
        ew = measure_equivalent_widths(wavelength, flux)
        ew_h_alpha = float(ew.get("H_alpha", {}).get("EW", np.nan))
        ew_ca_k = float(ew.get("Ca_II_K", {}).get("EW", np.nan))

        spectral_type_spec = spectral_type_from_ew(ew_h_alpha, ew_ca_k=ew_ca_k)
        teff_spec = teff_from_ew_h_alpha(ew_h_alpha)

        teff_diff_k: float | None = None
        teff_diff_pct: float | None = None
        if teff_photometric is not None:
            try:
                tphot = float(teff_photometric)
                if np.isfinite(teff_spec) and np.isfinite(tphot) and tphot != 0.0:
                    teff_diff_k = float(teff_spec - tphot)
                    teff_diff_pct = float(100.0 * teff_diff_k / tphot)
            except Exception:
                teff_diff_k = None
                teff_diff_pct = None

        # TODO: aplicar transformacion G->V (Evans et al. 2018) antes de
        # comparaciones fotometricas/espectroscopicas de precision.

        return {
            "source_id": source_id,
            "obsid": obsid,
            "wavelength": wavelength,
            "flux": flux,
            "equivalent_widths": ew,
            "spectral_type_spec": spectral_type_spec,
            "teff_spectroscopic": teff_spec,
            "teff_photometric": teff_photometric,
            "teff_diff_K": teff_diff_k,
            "teff_diff_pct": teff_diff_pct,
            "success": True,
            "error": None,
        }
    except Exception as exc:
        base["error"] = str(exc)
        return base


def _pick_teff_photometric(df_processed: pd.DataFrame, source_id: object) -> float | None:
    """Recupera T_eff fotometrica para un source_id dado."""
    if df_processed is None or df_processed.empty or "source_id" not in df_processed.columns:
        return None

    row = df_processed.loc[df_processed["source_id"].astype(str) == str(source_id)]
    if row.empty:
        return None

    for col in ["teff", "teff_corr"]:
        if col in row.columns:
            value = row.iloc[0].get(col)
            try:
                val = float(value)
                if np.isfinite(val):
                    return val
            except Exception:
                continue
    return None


def batch_analyse_spectra(
    df_crossmatch: pd.DataFrame,
    df_processed: pd.DataFrame,
    cache_dir: str | Path = SPECTRA_CACHE_DIR,
    max_spectra: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Analiza multiples espectros en lote.

    Para cada fila de df_crossmatch (que tiene source_id y obsid):
    - Busca teff_photometric en df_processed por source_id.
    - Llama a analyse_star_spectrum.
    - Acumula los resultados.

    Llama a progress_callback(n_done, n_total) si se proporciona.
    Limita el analisis a max_spectra espectros para evitar timeouts.

    Devuelve DataFrame con una fila por estrella analizada y columnas:
        source_id, obsid, spectral_type_spec, teff_spectroscopic,
        teff_photometric, teff_diff_K, teff_diff_pct,
        EW_H_alpha, EW_H_beta, EW_Ca_II_K, EW_Mg_I_b, success
    """
    if df_crossmatch is None or df_crossmatch.empty:
        return pd.DataFrame(
            columns=[
                "source_id",
                "obsid",
                "spectral_type_spec",
                "teff_spectroscopic",
                "teff_photometric",
                "teff_diff_K",
                "teff_diff_pct",
                "EW_H_alpha",
                "EW_H_beta",
                "EW_Ca_II_K",
                "EW_Mg_I_b",
                "success",
            ]
        )

    n_total = min(int(max_spectra), len(df_crossmatch)) if max_spectra is not None else len(df_crossmatch)
    subset = df_crossmatch.head(n_total)

    rows: list[dict[str, object]] = []
    for idx, (_, item) in enumerate(subset.iterrows(), start=1):
        source_id = item.get("source_id")
        obsid = item.get("obsid")
        teff_phot = _pick_teff_photometric(df_processed, source_id)

        result = analyse_star_spectrum(
            source_id=source_id,
            obsid=obsid,
            cache_dir=cache_dir,
            teff_photometric=teff_phot,
        )

        ew = result.get("equivalent_widths", {}) if isinstance(result, dict) else {}
        rows.append(
            {
                "source_id": source_id,
                "obsid": obsid,
                "spectral_type_spec": result.get("spectral_type_spec"),
                "teff_spectroscopic": result.get("teff_spectroscopic"),
                "teff_photometric": result.get("teff_photometric"),
                "teff_diff_K": result.get("teff_diff_K"),
                "teff_diff_pct": result.get("teff_diff_pct"),
                "EW_H_alpha": (ew.get("H_alpha") or {}).get("EW", np.nan),
                "EW_H_beta": (ew.get("H_beta") or {}).get("EW", np.nan),
                "EW_Ca_II_K": (ew.get("Ca_II_K") or {}).get("EW", np.nan),
                "EW_Mg_I_b": (ew.get("Mg_I_b") or {}).get("EW", np.nan),
                "success": bool(result.get("success", False)),
            }
        )

        if progress_callback is not None:
            try:
                progress_callback(idx, n_total)
            except Exception:
                pass

    return pd.DataFrame(rows)

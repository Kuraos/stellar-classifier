"""Carga y ajuste de isócronas PARSEC.

Este módulo trabaja con archivos CMD 3.7 de Padova/PARSEC y provee
utilidades puras para:
- parsear archivos de texto con cabeceras comentadas,
- listar y cargar isócronas por edad y metalicidad,
- filtrar fases evolutivas,
- convertir la isócrona a observables,
- calcular un chi-cuadrado geométrico contra observaciones,
- y buscar la mejor edad en un grid.

Referencia principal:
Bressan et al. (2012), MNRAS, 427, 127.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_ISOCHRONES_DIR = "data/isochrones"
_COLUMN_ALIASES = {
    "log_age": "log_age",
    "MH": "MH",
    "Mini": "Mini",
    "Mass": "Mass",
    "logL": "logL",
    "logTe": "logTe",
    "logg": "logg",
    "label": "label",
    "Gmag": "Gmag",
    "G_BPmag": "G_BPmag",
    "G_RPmag": "G_RPmag",
}
_FILENAME_RE = re.compile(
    r"logage_(?P<log_age>-?\d+(?:\.\d+)?)_mh_(?P<mh>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IsochroneRecord:
    """Metadatos resumidos de una isócrona disponible."""

    log_age: float
    metallicity: float
    age_gyr: float
    filepath: str
    label_humano: str


def _as_path(isochrones_dir: str | Path) -> Path:
    """Convierte una ruta a `Path` sin imponer existencia."""
    path = Path(isochrones_dir)
    return path if path.is_absolute() else Path.cwd() / path


def _parse_filename_metadata(filepath: Path) -> tuple[float | None, float | None]:
    """Intenta extraer log_age y metalicidad desde el nombre del archivo."""
    match = _FILENAME_RE.search(filepath.stem)
    if not match:
        return None, None
    return float(match.group("log_age")), float(match.group("mh"))


def _human_age_label(log_age: float, metallicity: float) -> str:
    """Genera una etiqueta breve y legible para la GUI."""
    age_gyr = 10.0 ** log_age / 1e9
    if age_gyr >= 1.0:
        age_text = f"{age_gyr:.1f} Gyr"
    else:
        age_text = f"{age_gyr * 1e3:.0f} Myr"
    return f"{age_text}, [M/H]={metallicity:.1f}"


def _detect_header_columns(lines: Iterable[str]) -> list[str] | None:
    """Detecta la última línea comentada que contiene nombres de columnas."""
    candidates: list[list[str]] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped.startswith("#"):
            continue
        tokens = [token for token in re.split(r"\s+", stripped.lstrip("# ")) if token]
        if len(tokens) >= 5 and ("log_age" in tokens or "Gmag" in tokens or "logTe" in tokens):
            candidates.append(tokens)
    if not candidates:
        return None
    return candidates[-1]


def parse_isochrone_file(filepath: str) -> pd.DataFrame:
    """Parsea un archivo PARSEC formato CMD 3.7.

    El formato suele incluir varias líneas comentadas con `#`; la última
    cabecera útil contiene los nombres de columnas. Esta función extrae al
    menos: `log_age`, `MH`, `Mini`, `Mass`, `logL`, `logTe`, `logg`, `label`,
    `Gmag`, `G_BPmag` y `G_RPmag`.

    Referencia: Bressan et al. (2012), MNRAS, 427, 127.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    header_columns = _detect_header_columns(lines)
    if header_columns is None:
        header_columns = list(_COLUMN_ALIASES)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=header_columns,
        header=None,
        engine="python",
    )

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "label" in df.columns:
        df["label"] = df["label"].round().astype("Int64")

    for column in _COLUMN_ALIASES:
        if column not in df.columns:
            df[column] = np.nan

    ordered_columns = [column for column in _COLUMN_ALIASES if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    return df[ordered_columns + remaining_columns].reset_index(drop=True)


def list_available_isochrones(
    isochrones_dir: str = DEFAULT_ISOCHRONES_DIR,
) -> list[dict]:
    """Lista todas las isócronas disponibles en el directorio.

    Devuelve una lista de diccionarios con:
    `log_age`, `metallicity`, `age_gyr`, `filepath`, `label_humano`.
    La salida está ordenada por `log_age` ascendente.

    Referencia: Bressan et al. (2012), MNRAS, 427, 127.
    """
    base_path = _as_path(isochrones_dir)
    if not base_path.exists():
        return []

    records: list[IsochroneRecord] = []
    for filepath in sorted(base_path.rglob("*.dat")):
        try:
            isochrone = parse_isochrone_file(str(filepath))
        except Exception:
            continue

        if isochrone.empty:
            continue

        if {"log_age", "MH"}.issubset(isochrone.columns):
            combos = (
                isochrone[["log_age", "MH"]]
                .dropna()
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            for log_age, metallicity in combos:
                records.append(
                    IsochroneRecord(
                        log_age=float(log_age),
                        metallicity=float(metallicity),
                        age_gyr=10.0 ** float(log_age) / 1e9,
                        filepath=str(filepath),
                        label_humano=_human_age_label(float(log_age), float(metallicity)),
                    )
                )

    records.sort(key=lambda item: (item.log_age, item.metallicity, item.filepath))
    return [record.__dict__ for record in records]


def load_isochrone(
    log_age: float,
    metallicity: float = 0.0,
    isochrones_dir: str = DEFAULT_ISOCHRONES_DIR,
) -> pd.DataFrame:
    """Carga una isócrona específica buscando por `log_age` y metalicidad.

    Se admite una tolerancia de ±0.05 dex en `log_age`. Si no se encuentra,
    la función lanza `FileNotFoundError` con un mensaje que enumera las
    isócronas disponibles.

    Referencia: Bressan et al. (2012), MNRAS, 427, 127.
    """
    base_path = _as_path(isochrones_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"No existe el directorio de isócronas: {base_path}")

    matches: list[pd.DataFrame] = []
    tolerance = 0.05
    for filepath in sorted(base_path.rglob("*.dat")):
        try:
            isochrone = parse_isochrone_file(str(filepath))
        except Exception:
            continue

        if isochrone.empty:
            continue

        age_mask = np.isclose(isochrone["log_age"].to_numpy(dtype=float), float(log_age), atol=tolerance, rtol=0.0)
        mh_mask = np.isclose(isochrone["MH"].to_numpy(dtype=float), float(metallicity), atol=0.05, rtol=0.0)
        subset = isochrone.loc[age_mask & mh_mask].copy()
        if not subset.empty:
            subset["filepath"] = str(filepath)
            matches.append(subset)

    if matches:
        return pd.concat(matches, ignore_index=True).sort_values(
            by=["log_age", "Mini", "logTe"],
            na_position="last",
        ).reset_index(drop=True)

    available = list_available_isochrones(isochrones_dir)
    available_labels = ", ".join(item["label_humano"] for item in available) or "sin archivos disponibles"
    raise FileNotFoundError(
        f"No se encontró una isócrona para log_age={log_age:.2f}, [M/H]={metallicity:.1f}. "
        f"Disponibles: {available_labels}"
    )


def filter_evolutionary_phases(
    isochrone: pd.DataFrame,
    phases: tuple[int, ...] = (0, 1, 2, 3, 4, 7, 8),
) -> pd.DataFrame:
    """Filtra puntos de la isócrona por fase evolutiva.

    Códigos PARSEC habituales:
    0 = pre-main sequence
    1 = main sequence
    2 = subgiant branch
    3 = red giant branch
    4 = core helium burning
    5,6 = AGB
    7,8 = post-AGB / WD

    Por defecto se conservan MS+SGB+RGB+HB y se excluye AGB/WD.

    Referencia: Bressan et al. (2012), MNRAS, 427, 127.
    """
    if "label" not in isochrone.columns:
        raise KeyError("La isócrona no contiene la columna 'label'.")

    mask = isochrone["label"].astype("Int64").isin(phases)
    return isochrone.loc[mask].copy().reset_index(drop=True)


def isochrone_to_observables(
    isochrone: pd.DataFrame,
    distance_pc: float | None = None,
    extinction_A_G: float = 0.0,
    reddening_E_BR: float = 0.0,
) -> pd.DataFrame:
    """Convierte una isócrona a observables absolutos o aparentes.

    Si `distance_pc` es `None`, devuelve magnitudes absolutas y colores
    intrínsecos. Si se proporciona una distancia, calcula magnitudes
    aparentes aplicando módulo de distancia y extinción.

    Salida mínima:
    - `log_T_eff`
    - `M_G_iso`
    - `BP_RP_iso`
    - opcionalmente `m_G_iso` y `BP_RP_obs`

    Referencia: Bressan et al. (2012), MNRAS, 427, 127.
    """
    required = {"logTe", "Gmag", "G_BPmag", "G_RPmag"}
    missing = required.difference(isochrone.columns)
    if missing:
        raise KeyError(f"Faltan columnas para convertir la isócrona: {sorted(missing)}")

    output = pd.DataFrame(index=isochrone.index)
    output["log_T_eff"] = pd.to_numeric(isochrone["logTe"], errors="coerce")
    output["M_G_iso"] = pd.to_numeric(isochrone["Gmag"], errors="coerce")
    output["BP_RP_iso"] = pd.to_numeric(isochrone["G_BPmag"], errors="coerce") - pd.to_numeric(
        isochrone["G_RPmag"], errors="coerce"
    )

    if distance_pc is not None:
        distance = float(distance_pc)
        with np.errstate(divide="ignore", invalid="ignore"):
            modulus = 5.0 * np.log10(distance) - 5.0
        if not np.isfinite(modulus):
            modulus = np.nan
        output["m_G_iso"] = output["M_G_iso"] + modulus + float(extinction_A_G)
        output["BP_RP_obs"] = output["BP_RP_iso"] + float(reddening_E_BR)

    return output


def chi_squared_isochrone(
    df_observed: pd.DataFrame,
    isochrone: pd.DataFrame,
    use_corrected: bool = False,
    use_bayesian: bool = False,
) -> float:
    """Calcula un chi-cuadrado geométrico entre observaciones e isócrona.

    Para cada estrella observada, se busca el punto más cercano de la
    isócrona en el espacio normalizado `(log T_eff, M_G)`. Las dimensiones
    se normalizan por su desviación estándar para evitar que `M_G` domine el
    ajuste.

    Selección de columnas:
    - `use_corrected=True` -> usa `teff_corr` y `M_G_corr` si están disponibles.
    - `use_bayesian=True` -> usa `teff` y `M_G_bayesian` si están disponibles.
    - Ambas banderas activas -> prima `use_corrected`.

    Referencia: Bressan et al. (2012), MNRAS, 427, 127.
    """
    if isochrone.empty:
        return float("inf")

    teff_col = "teff"
    mg_col = "M_G"
    if use_corrected and {"teff_corr", "M_G_corr"}.issubset(df_observed.columns):
        teff_col = "teff_corr"
        mg_col = "M_G_corr"
    elif use_bayesian and "M_G_bayesian" in df_observed.columns:
        mg_col = "M_G_bayesian"

    obs_teff = pd.to_numeric(df_observed[teff_col], errors="coerce").to_numpy(dtype=float)
    obs_mg = pd.to_numeric(df_observed[mg_col], errors="coerce").to_numpy(dtype=float)
    obs_log_teff = np.log10(obs_teff)

    valid_obs = np.isfinite(obs_log_teff) & np.isfinite(obs_mg) & (obs_teff > 0)
    obs_log_teff = obs_log_teff[valid_obs]
    obs_mg = obs_mg[valid_obs]
    if obs_log_teff.size == 0:
        return float("inf")

    if {"logTe", "Gmag"}.issubset(isochrone.columns):
        iso_log_teff = pd.to_numeric(isochrone["logTe"], errors="coerce").to_numpy(dtype=float)
        iso_mg = pd.to_numeric(isochrone["Gmag"], errors="coerce").to_numpy(dtype=float)
    elif {"log_T_eff", "M_G_iso"}.issubset(isochrone.columns):
        iso_log_teff = pd.to_numeric(isochrone["log_T_eff"], errors="coerce").to_numpy(dtype=float)
        iso_mg = pd.to_numeric(isochrone["M_G_iso"], errors="coerce").to_numpy(dtype=float)
    else:
        raise KeyError("La isócrona no contiene columnas compatibles para chi-cuadrado.")

    valid_iso = np.isfinite(iso_log_teff) & np.isfinite(iso_mg)
    iso_log_teff = iso_log_teff[valid_iso]
    iso_mg = iso_mg[valid_iso]
    if iso_log_teff.size == 0:
        return float("inf")

    obs_stack = np.column_stack([obs_log_teff, obs_mg])
    iso_stack = np.column_stack([iso_log_teff, iso_mg])

    scale = np.array([
        np.nanstd(np.concatenate([obs_log_teff, iso_log_teff])),
        np.nanstd(np.concatenate([obs_mg, iso_mg])),
    ])
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0

    obs_scaled = obs_stack / scale
    iso_scaled = iso_stack / scale

    chi2 = 0.0
    for point in obs_scaled:
        distances = np.sum((iso_scaled - point) ** 2, axis=1)
        chi2 += float(np.nanmin(distances))
    return chi2


def fit_best_age(
    df_observed: pd.DataFrame,
    age_grid: np.ndarray,
    metallicity: float = 0.0,
    use_corrected: bool = False,
    use_bayesian: bool = False,
    isochrones_dir: str = DEFAULT_ISOCHRONES_DIR,
) -> dict:
    """Busca la isócrona que minimiza chi² sobre un grid de edades.

    Si una edad no tiene archivo disponible, se avisa con `warnings.warn` y
    se sigue con el resto del grid.

    Devuelve un diccionario con:
    - `best_log_age`
    - `best_age_yr`
    - `best_age_label`
    - `min_chi2`
    - `chi2_curve`
    - `ages_probed`
    - `best_isochrone`

    Referencia: Bressan et al. (2012), MNRAS, 427, 127.
    """
    ages = np.asarray(age_grid, dtype=float)
    chi2_curve = np.full(ages.shape, np.nan, dtype=float)
    best_idx: int | None = None
    best_isochrone: pd.DataFrame | None = None

    for index, log_age in enumerate(ages):
        try:
            loaded = load_isochrone(float(log_age), metallicity=metallicity, isochrones_dir=isochrones_dir)
        except FileNotFoundError as exc:
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            continue

        filtered = filter_evolutionary_phases(loaded)
        chi2 = chi_squared_isochrone(
            df_observed=df_observed,
            isochrone=filtered,
            use_corrected=use_corrected,
            use_bayesian=use_bayesian,
        )
        chi2_curve[index] = chi2
        if np.isfinite(chi2) and (best_idx is None or chi2 < chi2_curve[best_idx]):
            best_idx = index
            best_isochrone = filtered

    if best_idx is None or best_isochrone is None:
        raise FileNotFoundError("No se encontró ninguna isócrona disponible para el grid solicitado.")

    best_log_age = float(ages[best_idx])
    best_age_yr = float(10.0 ** best_log_age)
    best_age_label = _human_age_label(best_log_age, metallicity).split(",", maxsplit=1)[0]

    return {
        "best_log_age": best_log_age,
        "best_age_yr": best_age_yr,
        "best_age_label": best_age_label,
        "min_chi2": float(chi2_curve[best_idx]),
        "chi2_curve": chi2_curve,
        "ages_probed": ages,
        "best_isochrone": best_isochrone,
    }

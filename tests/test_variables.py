"""Tests para src/variables.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.variables import (
    PERIOD_LUMINOSITY_TYPES,
    VARIABLE_LABELS,
    VARIABLE_MARKERS,
    VARIABLE_PLOT_STYLE,
    add_variability_columns,
    cepheid_distance,
    classify_variable_type,
    compare_distances,
    rrlyrae_distance,
)


@pytest.mark.parametrize(
    ("class_name", "period_days", "classification_result", "expected"),
    [
        ("DCEP", 10.0, None, "DCEP"),
        ("T2CEP", 15.0, None, "T2CEP"),
        ("RRAB", 0.6, True, "RRAB"),
        ("ECL", None, True, "ECL"),
        ("ROT", None, True, "ROT"),
    ],
)
def test_classify_variable_type_known_cases(
    class_name: object,
    period_days: object | None,
    classification_result: object | None,
    expected: str,
) -> None:
    assert classify_variable_type(class_name, period_days, classification_result) == expected


@pytest.mark.parametrize(
    ("class_name", "period_days", "classification_result"),
    [
        (None, None, None),
        (float("nan"), None, None),
        ("ALGO_RARO", None, False),
    ],
)
def test_classify_variable_type_missing_evidence_returns_non_variable(
    class_name: object,
    period_days: object | None,
    classification_result: object | None,
) -> None:
    assert classify_variable_type(class_name, period_days, classification_result) == "non_variable"


def test_classify_variable_type_unknown_with_period_returns_other() -> None:
    assert classify_variable_type("XYZ_UNKNOWN", period_days=5.0) == "OTHER"


def test_classify_variable_type_unknown_without_period_returns_non_variable() -> None:
    assert classify_variable_type("XYZ_UNKNOWN", period_days=None) == "non_variable"


def test_cepheid_distance_known_classical() -> None:
    """La relación clásica debe producir una distancia finita y grande."""
    d = cepheid_distance(np.array([6.0]), np.array([10.0]))
    assert np.isfinite(d[0])
    assert d[0] > 100.0


def test_cepheid_distance_invalid_period_returns_nan() -> None:
    d = cepheid_distance(np.array([5.0, 5.0, 5.0]), np.array([0.0, -1.0, np.nan]))
    assert np.isnan(d).all()


def test_cepheid_distance_type2_differs_from_classical() -> None:
    """La Cepheid tipo II debe dar una distancia distinta a la clásica."""
    d_classical = cepheid_distance(np.array([8.0]), np.array([5.0]), is_type2=False)
    d_type2 = cepheid_distance(np.array([8.0]), np.array([5.0]), is_type2=True)
    assert not np.isclose(d_classical[0], d_type2[0], rtol=0.01)


def test_cepheid_distance_vectorized() -> None:
    periods = np.array([1.0, 5.0, 10.0, 50.0])
    g_mags = np.full(4, 8.0)
    d = cepheid_distance(g_mags, periods)
    assert d.shape == (4,)
    assert np.all(np.isfinite(d))
    assert d[3] > d[0]


def test_rrlyrae_distance_default_metallicity() -> None:
    """Con [Fe/H]=-1.5 la distancia debe quedar en un rango plausible."""
    d = rrlyrae_distance(np.array([11.0]), np.array([0.6]))
    assert np.isfinite(d[0])
    assert 800 < d[0] < 2000


def test_rrlyrae_distance_metallicity_array() -> None:
    """La metalicidad como array debe hacer broadcast correcto."""
    g = np.array([11.0, 11.0, 11.0])
    period = np.array([0.5, 0.5, 0.5])
    feh = np.array([-2.0, -1.5, -1.0])
    d = rrlyrae_distance(g, period, metallicity=feh)
    assert d.shape == (3,)
    assert np.all(np.isfinite(d))
    assert d[0] > d[2]


def test_rrlyrae_distance_invalid_period_returns_nan() -> None:
    d = rrlyrae_distance(np.array([10.0]), np.array([-1.0]))
    assert np.isnan(d[0])


def _make_variable_df() -> pd.DataFrame:
    """DataFrame sintético con varios tipos Gaia DR3."""
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4, 5],
            "phot_g_mean_mag": [6.5, 11.0, 9.0, 10.0, 8.0],
            "best_class_name": ["DCEP", "RRAB", "ECL", None, "ALGO_RARO"],
            "in_vari_classification_result": [True, True, True, False, True],
            "cepheid_period": [10.0, np.nan, np.nan, np.nan, np.nan],
            "rrlyrae_period": [np.nan, 0.6, np.nan, np.nan, np.nan],
        }
    )


def test_add_variability_columns_adds_columns() -> None:
    df = _make_variable_df()
    result = add_variability_columns(df)
    for col in ["variable_type", "is_variable", "pl_period_days", "distance_pc_PL"]:
        assert col in result.columns


def test_add_variability_columns_assigns_pl_distances() -> None:
    df = _make_variable_df()
    result = add_variability_columns(df)

    dcep_row = result[result["source_id"] == 1].iloc[0]
    rrab_row = result[result["source_id"] == 2].iloc[0]

    assert dcep_row["variable_type"] == "DCEP"
    assert bool(dcep_row["is_variable"])
    assert np.isfinite(dcep_row["distance_pc_PL"])

    assert rrab_row["variable_type"] == "RRAB"
    assert bool(rrab_row["is_variable"])
    assert np.isfinite(rrab_row["distance_pc_PL"])


def test_add_variability_columns_non_variable_and_other() -> None:
    df = _make_variable_df()
    result = add_variability_columns(df)

    ecl_row = result[result["source_id"] == 3].iloc[0]
    non_var_row = result[result["source_id"] == 4].iloc[0]
    other_row = result[result["source_id"] == 5].iloc[0]

    assert ecl_row["variable_type"] == "ECL"
    assert np.isnan(ecl_row["distance_pc_PL"])

    assert non_var_row["variable_type"] == "non_variable"
    assert not bool(non_var_row["is_variable"])

    assert other_row["variable_type"] == "OTHER"
    assert bool(other_row["is_variable"])


def test_add_variability_columns_no_variability_columns_safe() -> None:
    """Un DataFrame sin columnas Gaia de variabilidad no debe fallar."""
    df = pd.DataFrame(
        {
            "source_id": [1, 2],
            "phot_g_mean_mag": [10.0, 11.0],
            "teff": [5000.0, 4500.0],
        }
    )
    result = add_variability_columns(df)
    assert "variable_type" in result.columns
    assert (result["variable_type"] == "non_variable").all()
    assert not result["is_variable"].any()


def test_add_variability_columns_empty_df_safe() -> None:
    """Un DataFrame vacío no debe fallar."""
    df = pd.DataFrame(columns=["source_id", "phot_g_mean_mag"])
    result = add_variability_columns(df)
    assert "variable_type" in result.columns
    assert len(result) == 0


def test_compare_distances_empty_when_no_pl_column() -> None:
    df = pd.DataFrame({"distance_pc": [100.0, 200.0]})
    result = compare_distances(df)
    assert result["n_compared"] == 0


def test_compare_distances_consistency() -> None:
    """Distancias iguales deben producir diferencia absoluta nula."""
    df = pd.DataFrame(
        {
            "distance_pc": [100.0, 200.0, 300.0],
            "distance_pc_PL": [100.0, 200.0, 300.0],
        }
    )
    result = compare_distances(df)
    assert result["n_compared"] == 3
    assert np.isclose(result["median_abs_diff_pc"], 0.0)


def test_compare_distances_handles_nan() -> None:
    """NaN en distancia P-L no debe contar en la comparación."""
    df = pd.DataFrame(
        {
            "distance_pc": [100.0, 200.0, 300.0],
            "distance_pc_PL": [105.0, np.nan, 310.0],
        }
    )
    result = compare_distances(df)
    assert result["n_compared"] == 2


def test_public_constants_are_consistent() -> None:
    """Las constantes públicas deben estar alineadas con la nomenclatura corta."""
    valid_marker_values = {"DCEP", "T2CEP", "RRAB", "RRC", "ECL", "MIRA", "ROT"}
    for key, value in VARIABLE_MARKERS.items():
        assert value in valid_marker_values, f"{key} -> {value} no usa una clave corta"
        assert " " not in value

    for tipo in PERIOD_LUMINOSITY_TYPES:
        assert tipo in VARIABLE_LABELS

    required = {"marker", "color", "zorder", "size"}
    for tipo, style in VARIABLE_PLOT_STYLE.items():
        assert required.issubset(style.keys()), f"{tipo} no tiene el estilo completo"

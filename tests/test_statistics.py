from __future__ import annotations

from src.statistics import compute_statistics


def test_compute_statistics_has_expected_top_keys(sample_processed_df) -> None:
    stats = compute_statistics(sample_processed_df)
    expected = {
        "n_stars",
        "teff",
        "M_G",
        "distance_pc",
        "luminosity_solar",
        "spectral_distribution",
        "distance_comparison",
    }
    # "variability" es opcional: aparece solo cuando el DataFrame
    # contiene columna variable_type. Verificamos que las claves
    # obligatorias siempre estén presentes.
    assert expected.issubset(set(stats.keys()))


def test_compute_statistics_distribution_has_all_types(sample_processed_df) -> None:
    stats = compute_statistics(sample_processed_df)
    distribution = stats["spectral_distribution"]
    for key in ["O", "B", "A", "F", "G", "K", "M"]:
        assert key in distribution
        assert isinstance(distribution[key], int)


def test_compute_statistics_counts_match_dataframe(sample_processed_df) -> None:
    stats = compute_statistics(sample_processed_df)
    assert stats["n_stars"] == len(sample_processed_df)

"""Tests adicionales de robustez para crossmatch LAMOST."""
from __future__ import annotations

import pandas as pd

from src import lamost


class _FakeResultTable:
    """Tabla minima estilo astropy para pruebas de crossmatch."""

    def __init__(self) -> None:
        self._df = pd.DataFrame(
            {
                "obsid": [123456],
                "RAJ2000": [10.0],
                "DEJ2000": [20.0],
                "snrg": [25.0],
                "snrr": [30.0],
                "class": ["STAR"],
                "subclass": ["G2V"],
                "_r": [0.3],
            }
        )

    def filled(self, _value):
        return self

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


class _FakeVizier:
    """Vizier falso que falla la primera consulta y luego responde."""

    def __init__(self, columns=None, row_limit=None):
        self.calls = 0

    def query_region(self, coord, radius, catalog):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("fallo transitorio")
        return [_FakeResultTable()]


def test_crossmatch_lamost_survives_partial_network_failures(monkeypatch) -> None:
    """Un fallo transitorio no debe forzar DataFrame vacio global."""
    monkeypatch.setattr(lamost, "Vizier", _FakeVizier)

    df = pd.DataFrame(
        {
            "source_id": [1, 2],
            "ra": [10.0, 11.0],
            "dec": [20.0, 21.0],
        }
    )

    out = lamost.crossmatch_lamost(df, max_stars=2)

    assert not out.empty
    assert "obsid" in out.columns
    assert (out["source_id"].astype(float) == 2.0).any()


class _BulkResultTable:
    """Simula resultado bulk de Vizier con columna _q."""

    def __init__(self) -> None:
        self._df = pd.DataFrame(
            {
                "obsid": [111, 222, 333],
                "RAJ2000": [10.0, 11.0, 12.0],
                "DEJ2000": [20.0, 21.0, 22.0],
                "snrg": [25.0, 30.0, 18.0],
                "snrr": [27.0, 28.0, 19.0],
                "class": ["STAR", "STAR", "STAR"],
                "subclass": ["G2V", "K0V", "F8V"],
                "_q": [1, 2, 3],
                "_r": [0.4, 0.6, 0.5],
            }
        )

    def filled(self, _v):
        return self

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


class _BulkVizier:
    def __init__(self, columns=None, row_limit=None):
        pass

    def query_region(self, coords, radius, catalog):
        return [_BulkResultTable()]


def test_crossmatch_lamost_bulk_uses_q_column(monkeypatch) -> None:
    """Una sola consulta bulk debe asignar source_ids via _q."""
    monkeypatch.setattr(lamost, "Vizier", _BulkVizier)
    df = pd.DataFrame(
        {
            "source_id": [101, 102, 103],
            "ra": [10.0, 11.0, 12.0],
            "dec": [20.0, 21.0, 22.0],
        }
    )
    out = lamost.crossmatch_lamost(df, max_stars=3)
    assert len(out) == 3
    assert set(out["source_id"].astype(int)) == {101, 102, 103}
    assert set(out["obsid"].astype(int)) == {111, 222, 333}

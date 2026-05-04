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

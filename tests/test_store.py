"""SQLite store helpers used by incremental fetching: offline, no network.

Covers ``latest_date`` (max stored date per ticker) and ``read_ohlcv`` (reconstruct
the ``download_ohlcv`` shape from ``ohlcv_raw`` for merging with a fresh tail).
"""

from __future__ import annotations

import pandas as pd

from concinvest.data import store


def _store_raw(conn, raw: dict[str, pd.DataFrame]) -> None:
    for ticker, df in raw.items():
        out = df.reset_index()
        out.insert(1, "ticker", ticker)
        store.upsert(conn, "ohlcv_raw", out)


def test_latest_date_per_ticker(tmp_path, synth_raw):
    conn = store.connect(tmp_path / "s.sqlite")
    _store_raw(conn, synth_raw)
    latest = store.latest_date(conn)
    conn.close()

    assert set(latest) == set(synth_raw)
    for ticker, df in synth_raw.items():
        assert latest[ticker] == max(df.index).strftime("%Y-%m-%d")


def test_latest_date_empty_db(tmp_path):
    conn = store.connect(tmp_path / "empty.sqlite")
    assert store.latest_date(conn) == {}
    conn.close()


def test_read_ohlcv_roundtrip(tmp_path, synth_raw):
    conn = store.connect(tmp_path / "r.sqlite")
    _store_raw(conn, synth_raw)
    back = store.read_ohlcv(conn, list(synth_raw))
    conn.close()

    assert set(back) == set(synth_raw)
    for ticker, df in synth_raw.items():
        got = back[ticker]
        # Same dates, sorted; OHLCV columns preserved.
        assert list(got.index) == sorted(df.index)
        assert {"open", "high", "low", "close", "adj_close", "volume"} <= set(got.columns)
        assert got["close"].iloc[-1] == df.loc[max(df.index), "close"]


def test_read_ohlcv_skips_absent_ticker(tmp_path, synth_raw):
    conn = store.connect(tmp_path / "p.sqlite")
    _store_raw(conn, {"TSLA": synth_raw["TSLA"]})
    back = store.read_ohlcv(conn, ["TSLA", "SIE.DE"])  # SIE.DE never stored
    conn.close()
    assert set(back) == {"TSLA"}

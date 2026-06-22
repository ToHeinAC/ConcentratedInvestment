"""Tests for cross-asset features, sentiment scaling, and SQLite storage."""

import pandas as pd

from concinvest.data import store
from concinvest.features import cross_asset, sentiment


def test_cross_asset_ratios(synth_raw):
    closes = {t: df["close"] for t, df in synth_raw.items()}
    closes = {t: pd.Series(s.values, index=pd.to_datetime(list(s.index)))
              for t, s in closes.items()}
    cross = cross_asset.build_cross_asset_frame(closes)
    assert "gold_oil_ratio" in cross.columns
    assert "copper_gold_ratio" in cross.columns
    assert "vix_sma20_ratio" in cross.columns
    # gold/oil ratio = GC=F / CL=F
    expected = closes["GC=F"] / closes["CL=F"]
    assert abs(cross["gold_oil_ratio"].iloc[-1] - expected.iloc[-1]) < 1e-9


def test_sentiment_scale_and_neutral():
    assert sentiment.score_headlines([]) == 0.0


def test_sentiment_sign_for_obvious_text():
    pos = sentiment.score_headlines(["Company posts record profit, raises guidance"])
    neg = sentiment.score_headlines(["Company collapses amid fraud scandal and huge losses"])
    assert -3.0 <= neg <= 3.0 and -3.0 <= pos <= 3.0
    assert pos > neg


def test_store_upsert_roundtrip(tmp_path):
    db = tmp_path / "t.sqlite"
    conn = store.connect(db)
    df = pd.DataFrame(
        {"date": ["2024-01-01", "2024-01-02"], "ticker": ["X", "X"], "close": [1.0, 2.0]}
    )
    # daily_market has these columns; upsert a subset.
    n = store.upsert(conn, "daily_market", df)
    assert n == 2
    # Idempotent: re-upsert same keys does not duplicate.
    store.upsert(conn, "daily_market", df)
    back = store.read_table(conn, "daily_market", ticker="X")
    assert len(back) == 2
    conn.close()

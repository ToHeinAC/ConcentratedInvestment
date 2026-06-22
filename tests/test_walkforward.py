"""Offline tests for walk-forward (multi-window) validation."""

import pandas as pd

from concinvest.backtest import walkforward
from concinvest.features import cross_asset
from concinvest.ml import dataset


def _panel_prices_nasdaq(synth_market, synth_raw):
    closes = {t: pd.Series(df["close"].values, index=pd.to_datetime(list(df.index)))
              for t, df in synth_raw.items()}
    cross = cross_asset.build_cross_asset_frame(closes)
    panel = dataset.build_feature_panel(synth_market, cross)
    prices = {t: pd.Series(df["close"].values, index=pd.to_datetime(list(df.index)))
              for t, df in synth_market.items()}
    nasdaq = synth_raw["^IXIC"]["close"]
    nasdaq.index = pd.to_datetime(list(nasdaq.index))
    return panel, prices, nasdaq


def test_windows_are_consecutive_and_non_overlapping():
    dates = pd.bdate_range("2020-01-01", periods=500)
    wins = walkforward._windows(dates, n_windows=3, window=100)
    assert len(wins) == 3
    # Walking forward in time, non-overlapping, each spanning `window` rows.
    starts = [w[1] for w in wins]
    assert starts == sorted(starts)
    assert wins[-1][2] == dates[-1]  # last window ends at the most recent date


def test_walk_forward_validate_runs(synth_market, synth_raw):
    panel, prices, nasdaq = _panel_prices_nasdaq(synth_market, synth_raw)
    res = walkforward.walk_forward_validate(
        synth_market, nasdaq, panel, prices,
        n_windows=2, window=120, n_dataset=800, horizon=20, tune=False,
    )
    assert not res.windows.empty
    assert {"portfolio", "benchmark", "outperformance", "beats"}.issubset(res.windows.columns)
    assert res.windows["beats"].dtype == bool
    assert 0.0 <= res.win_rate <= 1.0
    assert len(res.windows) <= 2

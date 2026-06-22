"""Tests for dataset generation, model training, forecasting, and backtest."""

import pandas as pd

from concinvest.backtest import engine
from concinvest.features import cross_asset
from concinvest.ml import dataset, forecast, model


def _panel_and_prices(synth_market, synth_raw):
    closes = {t: pd.Series(df["close"].values, index=pd.to_datetime(list(df.index)))
              for t, df in synth_raw.items()}
    cross = cross_asset.build_cross_asset_frame(closes)
    panel = dataset.build_feature_panel(synth_market, cross)
    prices = {t: pd.Series(df["close"].values, index=pd.to_datetime(list(df.index)))
              for t, df in synth_market.items()}
    return panel, prices


def test_generate_dataset_shapes_and_balance(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=500, horizon=20, seed=1)
    assert list(X.columns) == dataset.FEATURE_COLS
    assert len(X) == len(y) == 500
    assert set(y.unique()).issubset({0, 1})
    # roughly balanced buys/sells
    assert 0.3 < X["is_sell"].mean() < 0.7


def test_train_and_forecast_five_fields(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=800, horizon=20, seed=2)
    trained = model.train(X, y, n_estimators=50, n_splits=3)
    assert len(trained.feature_importance) == len(dataset.FEATURE_COLS)

    snaps = {t: panel.xs(t, level="ticker").iloc[-1] for t in synth_market}
    fcs = forecast.forecast(trained, snaps, threshold=0.0)  # force trades
    assert fcs, "expected at least one forecast with threshold 0"
    f = fcs[0]
    assert f.action in {"buy", "sell"}
    assert f.leverage in {"stock", "2x", "3x"}
    assert 0.0 <= f.confidence <= 1.0
    assert f.amount_eur > 0


def test_backtest_produces_curve(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=600, horizon=20, seed=3)
    trained = model.train(X, y, n_estimators=50, n_splits=3)
    bench = synth_raw["^IXIC"]["close"]
    bench.index = pd.to_datetime(list(bench.index))
    res = engine.run_backtest(synth_market, bench, trained, panel)
    assert {"portfolio", "benchmark"}.issubset(res.curve.columns)
    assert len(res.curve) > 50
    assert isinstance(res.beats_benchmark, bool)

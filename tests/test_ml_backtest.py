"""Tests for dataset generation, model training, forecasting, and backtest."""

import pandas as pd
import pytest

from concinvest import config
from concinvest.backtest import engine
from concinvest.features import cross_asset
from concinvest.ml import dataset, forecast, model
from concinvest.portfolio import state as pstate


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


def test_generate_dataset_is_chronologically_ordered(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=500, horizon=20, seed=7)
    assert isinstance(X.index, pd.DatetimeIndex)
    assert X.index.is_monotonic_increasing  # honest input for TimeSeriesSplit
    assert (X.index == y.index).all()


def test_train_validate_split_by_date(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=600, horizon=20, seed=8)
    Xtr, ytr, Xva, yva = dataset.train_validate_split(X, y, validation_years=1)
    assert len(Xtr) + len(Xva) == len(X)
    assert len(Xtr) == len(ytr) and len(Xva) == len(yva)
    # Every train date precedes every validation date (no temporal overlap).
    if len(Xva):
        assert Xtr.index.max() < Xva.index.min()


def test_tune_returns_grid_params(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=900, horizon=20, seed=9)
    params, scores = model.tune(X, y, n_splits=3)
    assert params in [dict(p) for p in model.PARAM_GRID]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_prune_keeps_action_features_and_forecasts(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=900, horizon=20, seed=11)
    m = model.tune_and_train(X, y, n_splits=3, prune=True)
    assert set(m.features).issubset(set(dataset.FEATURE_COLS))
    assert set(dataset.ACTION_FEATURES).issubset(set(m.features))
    # Forecast/predict still work on the pruned feature set.
    snaps = {t: panel.xs(t, level="ticker").iloc[-1] for t in synth_market}
    assert forecast.forecast(m, snaps, threshold=0.0)


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


def test_apply_book_limits_caps_buys_at_cash_and_sells_at_holdings():
    fcs = [
        forecast.Forecast("A", "buy", 10_000.0, "stock", 0.7),
        forecast.Forecast("B", "buy", 10_000.0, "2x", 0.6),    # cash partly exhausted
        forecast.Forecast("C", "sell", 10_000.0, "3x", 0.8),   # only 2k held -> capped
        forecast.Forecast("D", "sell", 5_000.0, "stock", 0.6),  # nothing held -> dropped
    ]
    out = forecast.apply_book_limits(fcs, cash=12_000.0, held={("C", "3x"): 2_000.0})
    by = {f.ticker: f.amount_eur for f in out}
    assert by["A"] == 10_000.0          # funded in full
    assert by["B"] == 2_000.0           # only remaining cash after A
    assert by["C"] == 2_000.0           # capped to the held 3x value
    assert "D" not in by                # no position to sell


def test_rules_backtest_produces_curve(synth_market, synth_raw):
    bench = synth_raw["^IXIC"]["close"]
    bench.index = pd.to_datetime(list(bench.index))
    res = engine.run_rules_backtest(synth_market, bench)
    assert {"portfolio", "benchmark"}.issubset(res.curve.columns)
    assert len(res.curve) > 50
    assert (res.curve["portfolio"] > 0).all()  # book never goes negative
    assert isinstance(res.beats_benchmark, bool)


def test_target_exposure_base_case_faithful():
    base = config.BASE_STOCK_ALLOCATION
    assert engine._target_exposure(0.5) == base   # neutral -> full base case
    assert engine._target_exposure(0.9) == base   # bullish -> capped at base
    assert abs(engine._target_exposure(0.25) - base * 0.5) < 1e-9  # bearish -> de-risk
    assert engine._target_exposure(0.0) == 0.0


def test_target_name_fraction_base_case_faithful():
    base = engine._PER_NAME_BASE
    assert engine._target_name_fraction(0.5) == base          # neutral -> per-name base
    assert engine._target_name_fraction(0.9) == base          # bullish -> capped
    # Mildly bearish de-risks proportionally, staying above the 6% floor (base*0.5 = 9%).
    assert abs(engine._target_name_fraction(0.25) - base * 0.5) < 1e-9
    # A deeply bearish read is floored at the 6% per-name minimum.
    assert engine._target_name_fraction(0.0) == config.MIN_NAME_WEIGHT


def test_rebalance_names_handles_each_name_independently():
    st = pstate.build_base_case(100_000.0, stocks=["A", "B"])
    # A bearish (target halved), B neutral (stays at base) -> only A is trimmed.
    targets = {"A": engine._target_name_fraction(0.25),
               "B": engine._target_name_fraction(0.5)}
    trades = engine._rebalance_names_to_target(st, targets, ["A", "B"])
    assert trades and {t.ticker for t in trades} == {"A"}  # only the bearish name
    assert all(t.action == "sell" for t in trades)
    # Pro-rata sell is logged per tier with an actual € amount each (not one aggregate).
    assert {t.tier for t in trades} == {1, 2, 3}
    assert all(t.amount_eur > 0 for t in trades)


def test_is_crisis_detects_sharp_drop():
    crash = pd.Series([-0.03] * 12)  # ~26% cumulative over 10 days
    assert engine._is_crisis(crash, 11)
    calm = pd.Series([0.001] * 12)
    assert not engine._is_crisis(calm, 11)
    assert not engine._is_crisis(crash, 3)  # too early: lookback not yet filled


def test_benchmark_curve_handles_leading_holiday_gap():
    # Window opens on a date the benchmark didn't trade (leading NaN after reindex).
    dates = pd.date_range("2025-07-04", periods=5, freq="D")
    bench = pd.Series([100.0, 110.0], index=pd.to_datetime(["2025-07-07", "2025-07-08"]))
    curve = engine._benchmark_curve(bench, dates)
    assert not curve.isna().any()  # leading NaN back-filled, not propagated
    assert curve.iloc[0] == config.INITIAL_CAPITAL_EUR  # rebased to first known price


def test_dividend_yields_recover_total_minus_price_return(synth_market):
    import numpy as np

    # Give one stock a dividend-bearing adj_close compounding 0.1%/day above price.
    t = next(iter(synth_market))
    df = synth_market[t].copy()
    df["adj_close"] = df["close"].values * np.cumprod(np.full(len(df), 1.001))
    synth_market[t] = df
    dates = pd.to_datetime(list(df.index))
    divs = engine._dividend_yields(synth_market, dates)
    # The doctored name yields ~0.1%/day; others (adj_close == close) yield 0.
    assert abs(divs[t].iloc[1:].mean() - 0.001) < 1e-4
    assert (divs.drop(columns=[t]).to_numpy() == 0.0).all()


def test_deploy_records_buy_trades():
    st = pstate.PortfolioState(cash=1000.0, high_water=1000.0)
    trades = engine._deploy(st, 1000.0, ["A", "B"])
    assert {t.ticker for t in trades} == {"A", "B"}
    assert all(t.action == "buy" and t.amount_eur > 0 for t in trades)
    assert st.cash < 1.0  # cash deployed into lots


def test_forecast_backtest_produces_curve(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=600, horizon=20, seed=4)
    trained = model.train(X, y, n_estimators=50, n_splits=3)
    bench = synth_raw["^IXIC"]["close"]
    bench.index = pd.to_datetime(list(bench.index))
    res = engine.run_forecast_backtest(synth_market, bench, trained, panel)
    assert {"portfolio", "benchmark"}.issubset(res.curve.columns)
    assert len(res.curve) > 50
    assert (res.curve["portfolio"] > 0).all()
    assert isinstance(res.beats_benchmark, bool)
    # Trade log: well-formed and dated (recording is unit-tested in _deploy / de-risk).
    assert isinstance(res.trades, list)
    assert all(t.date is not None and t.action in {"buy", "sell"} for t in res.trades)
    # Final book is exposed for the Current-portfolio view; matches the curve's end.
    assert res.final_state is not None
    assert res.final_state.total_value() == pytest.approx(
        float(res.curve["portfolio"].iloc[-1])
    )
    # Per-tier balance history for the Strategy tab: (ticker, tier) columns, dated.
    tc = res.tier_curve
    assert tc is not None and not tc.empty
    assert tc.columns.names == ["ticker", "tier"]
    a_stock = synth_market and list(synth_market)[0]
    assert a_stock in tc.columns.get_level_values("ticker")
    assert set(tc[a_stock].columns) <= {"stock", "2x", "3x"}


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

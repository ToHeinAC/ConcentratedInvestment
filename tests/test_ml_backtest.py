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


def test_select_features_relaxes_cutoff_for_wide_feature_set():
    # Uniform importances over the full (wide, lag-augmented) contract: each ~1/69 ≈
    # 0.0145, below the absolute 0.02 floor (which would collapse the model to the
    # action encoding) but above the scaled cutoff 0.5/69 ≈ 0.0072 — so they survive.
    feats = dataset.FEATURE_COLS
    imp = {f: 1.0 / len(feats) for f in feats}
    tm = model.TrainedModel(clf=None, feature_importance=imp)
    kept = model.select_features(tm)
    assert set(dataset.ACTION_FEATURES).issubset(kept)
    assert len(kept) == len(feats)  # market features not pruned away


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


def test_apply_book_limits_drops_orders_below_min_trade():
    fcs = [
        forecast.Forecast("A", "buy", 9_800.0, "stock", 0.7),  # leaves 200 € cash
        forecast.Forecast("B", "buy", 1_000.0, "2x", 0.6),     # capped to 200 € -> dropped
        forecast.Forecast("C", "sell", 1_000.0, "3x", 0.8),    # only 300 € held -> dropped
    ]
    out = forecast.apply_book_limits(fcs, cash=10_000.0, held={("C", "3x"): 300.0})
    # B and C fall below MIN_TRADE_EUR after capping; only A (9.8k) survives.
    assert {f.ticker for f in out} == {"A"}


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
    # Daily cash history for the Strategy tab's Cash view: aligned, non-negative, and
    # cash + invested == total portfolio value each day (the evolution of cash).
    cash = res.cash_curve
    assert cash is not None
    assert (cash.index == res.curve.index).all()
    assert (cash >= -1e-6).all()
    invested = tc.sum(axis=1).reindex(res.curve.index)
    assert (cash + invested).to_numpy() == pytest.approx(
        res.curve["portfolio"].to_numpy()
    )


# --- aggressive strategy -------------------------------------------------
def test_agg_stop_loss_exits_only_lots_below_threshold():
    st = pstate.PortfolioState(cash=0.0, high_water=100_000.0)
    st.lots = [pstate.Lot("A", 3, 10_000.0, 3_900.0),   # -61% -> stop out
               pstate.Lot("B", 3, 10_000.0, 5_000.0),   # -50% -> hold
               pstate.Lot("A", 1, 10_000.0, 1_000.0)]   # underlying base -> never stopped
    trades = engine._agg_stop_loss(st)
    assert [t.ticker for t in trades] == ["A"] and trades[0].tier == 3
    assert {lot.ticker for lot in st.lots} == {"A", "B"}  # only the A 3x lot left
    assert all(not (lot.ticker == "A" and lot.tier == 3) for lot in st.lots)


def test_agg_take_profit_skims_rebases_and_seeds_underlying():
    st = pstate.PortfolioState(cash=0.0, high_water=100_000.0)
    st.lots = [pstate.Lot("A", 3, 10_000.0, 16_000.0, tp_basis=10_000.0)]  # +60%
    sell_row = pd.Series({"A": 0.0})  # no extra ML conviction -> minimum 30% skim
    trades = engine._agg_take_profit(st, sell_row)
    # Skim 30% of 16k = 4.8k gross (no gain tax-free portion: gain 6k -> partial tax).
    sell = next(t for t in trades if t.action == "sell")
    assert sell.tier == 3 and abs(sell.amount_eur - 4_800.0) < 1e-6
    a3 = next(lot for lot in st.lots if lot.tier == 3)
    assert abs(a3.value - 11_200.0) < 1e-6        # 70% of the position remains
    assert abs(a3.tp_basis - 11_200.0) < 1e-6     # reference re-based to the remainder
    # ~half of net proceeds seeded a permanent tier-1 underlying lot; rest is cash.
    assert any(lot.tier == 1 for lot in st.lots)
    buy = next(t for t in trades if t.action == "buy")
    assert buy.tier == 1 and st.cash > 0.0


def test_agg_take_profit_skips_below_threshold():
    st = pstate.PortfolioState(cash=0.0, high_water=100_000.0)
    st.lots = [pstate.Lot("A", 3, 10_000.0, 15_000.0, tp_basis=10_000.0)]  # +50% only
    assert engine._agg_take_profit(st, pd.Series({"A": 0.9})) == []


def test_agg_entries_deploys_fixed_chunk_on_bullish_names():
    st = pstate.PortfolioState(cash=20_000.0, high_water=100_000.0)
    buy_row = pd.Series({"A": 0.9, "B": 0.5})  # A bullish, B below entry threshold
    trades = engine._agg_entries(st, buy_row, total=100_000.0)
    assert [t.ticker for t in trades] == ["A"]
    assert trades[0].tier == 3
    assert abs(trades[0].amount_eur - 10_000.0) < 1e-6  # 10% of 100k portfolio value
    assert abs(st.cash - 10_000.0) < 1e-6


def test_agg_entries_skip_when_cash_below_min_trade():
    st = pstate.PortfolioState(cash=300.0, high_water=100_000.0)  # < MIN_TRADE_EUR
    assert engine._agg_entries(st, pd.Series({"A": 0.9}), total=100_000.0) == []


def test_agg_entries_skip_names_already_at_cap():
    st = pstate.PortfolioState(cash=50_000.0, high_water=100_000.0)
    st.lots = [pstate.Lot("A", 3, 40_000.0, 40_000.0),   # 40% -> over the 33% cap
               pstate.Lot("B", 3, 10_000.0, 10_000.0)]    # 10% -> still has room
    trades = engine._agg_entries(st, pd.Series({"A": 0.9, "B": 0.9}), total=100_000.0)
    assert [t.ticker for t in trades] == ["B"]  # no new entry into the capped name A


def test_agg_cap_overweight_trims_back_to_cap_3x_first():
    st = pstate.PortfolioState(cash=0.0, high_water=60_000.0)
    # Book = 60k: A is 50k (stock 10k + 3x 40k, > the 33% cap of 19.8k), B is 10k (under).
    st.lots = [pstate.Lot("A", 1, 10_000.0, 10_000.0),
               pstate.Lot("A", 3, 40_000.0, 40_000.0),
               pstate.Lot("B", 3, 10_000.0, 10_000.0)]
    trades = engine._agg_cap_overweight(st)
    assert {t.ticker for t in trades} == {"A"}  # only the over-cap name
    assert trades[0].tier == 3  # riskiest tier shed first
    # A trimmed back to exactly the 33% cap; the 30.2k excess comes entirely from the 3x
    # tier (it had 40k), leaving the underlying base (10k) untouched.
    cap = config.PER_NAME_CAP * 60_000.0
    assert abs(st.name_value("A") - cap) < 1e-6
    a3 = sum(lot.value for lot in st.lots if lot.ticker == "A" and lot.tier == 3)
    assert abs(a3 - (40_000.0 - (50_000.0 - cap))) < 1e-6  # 40k - 30.2k excess


def test_aggressive_backtest_produces_curve(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=600, horizon=20, seed=5)
    trained = model.train(X, y, n_estimators=50, n_splits=3)
    bench = synth_raw["^IXIC"]["close"]
    bench.index = pd.to_datetime(list(bench.index))
    res = engine.run_aggressive_backtest(synth_market, bench, trained, panel)
    assert {"portfolio", "benchmark"}.issubset(res.curve.columns)
    assert len(res.curve) > 50
    assert (res.curve["portfolio"] > 0).all()
    assert all(t.date is not None and t.action in {"buy", "sell"} for t in res.trades)
    # Final book matches the curve end and only ever holds 3x + underlying (stock) tiers.
    assert res.final_state.total_value() == pytest.approx(
        float(res.curve["portfolio"].iloc[-1])
    )
    assert all(lot.tier in (1, 3) for lot in res.final_state.lots)  # no 2x in this book
    # Per-name concentration cap holds at end of window (within one min-trade slack).
    fs = res.final_state
    cap = config.PER_NAME_CAP * fs.total_value() + config.MIN_TRADE_EUR
    assert all(fs.name_value(t) <= cap for t in synth_market)
    tc = res.tier_curve
    assert tc is not None and not tc.empty
    assert set(tc.columns.get_level_values("tier")) <= {"stock", "3x"}
    # cash + invested == portfolio value each day.
    invested = tc.sum(axis=1).reindex(res.curve.index)
    assert (res.cash_curve + invested).to_numpy() == pytest.approx(
        res.curve["portfolio"].to_numpy()
    )


def test_forecast_restricts_to_3x_when_requested(synth_market, synth_raw):
    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=600, horizon=20, seed=6)
    trained = model.train(X, y, n_estimators=50, n_splits=3)
    snaps = {t: panel.xs(t, level="ticker").iloc[-1] for t in synth_market}
    fcs = forecast.forecast(trained, snaps, threshold=0.0, leverages=(3,))
    assert fcs and all(f.leverage == "3x" for f in fcs)


def test_build_dated_book_per_tier_dates(synth_market):
    from concinvest import pipeline

    t = next(iter(synth_market))
    closes = pd.Series(synth_market[t]["close"].values,
                       index=pd.to_datetime(list(synth_market[t].index))).sort_index()
    early, late = closes.index[len(closes) // 3], closes.index[2 * len(closes) // 3]
    # Same name, two tiers, each with its **own** buy date — evaluated separately.
    positions = [
        {"ticker": t, "tier": 1, "invested_eur": 1000.0, "buy_date": early},
        {"ticker": t, "tier": 3, "invested_eur": 1000.0, "buy_date": late},
    ]
    st = pipeline.build_dated_book(positions, synth_market, cash=500.0)

    assert st.cash == 500.0
    assert all(abs(l.cost_basis - 1000.0) < 1e-9 for l in st.lots)  # invested = cost basis
    h1 = closes[closes.index >= early]
    lot1 = next(l for l in st.lots if l.tier == 1)
    assert abs(lot1.value - 1000.0 * h1.iloc[-1] / h1.iloc[0]) < 1e-6  # 1x from early date
    # 3x marked from its own (later) buy date: daily-rebalanced 3x leverage (a real
    # leveraged ETF / the backtest's state.mark) — cumprod of (1 + 3*daily_return),
    # not the total return scaled once.
    h3 = closes[closes.index >= late]
    factor3 = (1.0 + 3.0 * h3.pct_change().fillna(0.0)).clip(lower=0.0).cumprod()
    lot3 = next(l for l in st.lots if l.tier == 3)
    assert abs(lot3.value - 1000.0 * factor3.iloc[-1]) < 1e-6
    assert lot3.tp_basis == 1000.0  # take-profit reference starts at cost
    assert st.high_water >= st.total_value() - 1e-6  # peak of the marked book path


def test_dated_book_value_path(synth_market):
    # The Live tab's performance chart: combined €-value path = cash + each lot's
    # daily-rebalanced leverage value, starting at the earliest buy date. Its first equals
    # cash + invested (no drift yet) and its last equals the marked book value.
    from concinvest import pipeline

    t = next(iter(synth_market))
    closes = pd.Series(synth_market[t]["close"].values,
                       index=pd.to_datetime(list(synth_market[t].index))).sort_index()
    buy = closes.index[len(closes) // 3]
    positions = [{"ticker": t, "tier": 2, "invested_eur": 1000.0, "buy_date": buy}]
    path = pipeline.dated_book_value_path(positions, synth_market, cash=500.0)

    assert not path.empty
    assert path.index[0] == buy
    assert abs(path.iloc[0] - 1500.0) < 1e-6  # cash + invested at the buy date
    st = pipeline.build_dated_book(positions, synth_market, cash=500.0)
    assert abs(path.iloc[-1] - st.total_value()) < 1e-6  # last point = marked book value


def test_build_dated_book_high_water_never_below_current(synth_market):
    # Regression: a lot bought *after* the last close has an empty price path; it must
    # still count toward the high-water, else drawdown goes spuriously negative.
    from concinvest import pipeline

    t = next(iter(synth_market))
    last = pd.Timestamp(synth_market[t].index[-1])
    future = last + pd.Timedelta(days=5)  # beyond available data -> empty path, fallback
    positions = [
        {"ticker": t, "tier": 1, "invested_eur": 1000.0,
         "buy_date": synth_market[t].index[len(synth_market[t]) // 2]},  # has a path
        {"ticker": t, "tier": 3, "invested_eur": 1000.0, "buy_date": future},  # no path
    ]
    st = pipeline.build_dated_book(positions, synth_market, cash=0.0)
    drawdown = (st.high_water - st.total_value()) / st.high_water
    assert drawdown >= -1e-9  # never negative


def test_build_dated_book_ignores_trailing_nan_close(synth_market):
    # Regression: yfinance sometimes returns a NaN close for the latest bar. Such a
    # trailing NaN must not poison the lot's value (-> book current value NaN and the
    # position vanishing from the pie); valuation uses the last *valid* close.
    import math

    from concinvest import pipeline

    t = next(iter(synth_market))
    market = {k: v.copy() for k, v in synth_market.items()}
    last_valid = float(market[t]["close"].iloc[-2])
    market[t].iloc[-1, market[t].columns.get_loc("close")] = float("nan")  # NaN latest bar

    buy = market[t].index[len(market[t]) // 3]
    positions = [{"ticker": t, "tier": 3, "invested_eur": 1000.0, "buy_date": buy}]
    st = pipeline.build_dated_book(positions, market, cash=500.0)

    lot = next(l for l in st.lots if l.ticker == t)
    assert math.isfinite(lot.value) and lot.value > 0  # not NaN -> still on the pie
    assert math.isfinite(st.total_value())
    # Value derives from the last valid close, not the NaN bar (daily-rebalanced 3x).
    held = pd.Series(market[t]["close"].values,
                     index=pd.to_datetime(list(market[t].index))).dropna()
    held = held[held.index >= pd.Timestamp(buy)]
    factor = (1.0 + 3.0 * held.pct_change().fillna(0.0)).clip(lower=0.0).cumprod()
    assert abs(lot.value - 1000.0 * factor.iloc[-1]) < 1e-6
    assert abs(held.iloc[-1] - last_valid) < 1e-9  # last valid close, not the NaN bar
    assert st.high_water >= st.total_value() - 1e-6


def test_portfolio_store_roundtrip(tmp_path):
    from concinvest.data import portfolio_store as ps

    positions = pd.DataFrame(
        [{"ticker": "TSLA", "tier": 1, "invested_eur": 9000.0,
          "buy_date": pd.Timestamp("2024-01-15")},
         {"ticker": "TSLA", "tier": 3, "invested_eur": 4500.0,
          "buy_date": pd.Timestamp("2024-06-01")}],
        columns=ps.POSITION_COLS,
    )
    ps.save_portfolio("mybook", positions, cash=7000.0, base=tmp_path)
    assert ps.list_portfolios(base=tmp_path) == ["mybook"]
    loaded, cash = ps.load_portfolio("mybook", base=tmp_path)
    assert cash == 7000.0
    assert list(loaded.columns) == ps.POSITION_COLS
    assert len(loaded) == 2 and "CASH" not in set(loaded["ticker"])  # cash row split out
    # Per-tier buy dates survive the round-trip.
    assert loaded.set_index("tier").loc[3, "buy_date"] == pd.Timestamp("2024-06-01")


def test_recommend_for_portfolio_user_book(synth_market, synth_raw):
    from concinvest import pipeline

    panel, prices = _panel_and_prices(synth_market, synth_raw)
    X, y = dataset.generate_dataset(panel, prices, n=600, horizon=20, seed=12)
    trained = model.train(X, y, n_estimators=50, n_splits=3)
    # A concentrated user book: one name way over the 33% cap -> a guardrail trim fires.
    names = list(synth_market)
    st = pstate.PortfolioState(cash=5_000.0)
    st.lots = [pstate.Lot(names[0], 1, 60_000.0, 60_000.0),
               pstate.Lot(names[0], 3, 20_000.0, 20_000.0),
               pstate.Lot(names[1], 1, 15_000.0, 15_000.0)]
    before = st.total_value()
    fcs, sent, guard = pipeline.recommend_for_portfolio(
        st, trained, panel, synth_market, with_sentiment=False
    )
    assert sent.empty  # sentiment disabled -> no live fetch
    assert isinstance(fcs, list)
    # The over-cap name is trimmed, riskiest tier first; side-effect-free on the input.
    assert any(t.ticker == names[0] and t.action == "sell" for t in guard)
    assert st.total_value() == before  # guardrails ran on a copy, not the caller's state


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

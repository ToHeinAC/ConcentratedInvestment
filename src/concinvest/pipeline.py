"""End-to-end Phase 1 orchestration: fetch -> features -> store -> ML -> forecast.

``run_phase1`` runs the full thin slice and returns everything the UI / CLI needs.
``fetch_and_store`` is the daily-ETL building block reused by the cron job later.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import pandas as pd

from . import config
from .backtest.engine import (
    BacktestResult,
    run_aggressive_backtest,
    run_forecast_backtest,
)
from .backtest.walkforward import WalkForwardResult, walk_forward_validate
from .data import fetch, store, tickers
from .features import analyst, cross_asset, regime, technical
from .features.regime import Regime
from .ml import dataset, forecast, model, overlay
from .ml.forecast import Forecast


@dataclass
class Phase1Result:
    market: dict[str, pd.DataFrame]
    cross: pd.DataFrame
    model: model.TrainedModel
    forecasts: list[Forecast]
    backtest: BacktestResult
    correlation: pd.DataFrame
    sentiment: pd.DataFrame  # live analyst/sentiment rows (empty if disabled)
    nasdaq: pd.Series  # raw NASDAQ close (for the Strategy tab)
    panel: pd.DataFrame  # (date, ticker) feature panel (reused by the Live tab)
    regime: Regime | None = None  # rising-market badge (None if ^GSPC/^VIX absent)


# Re-pull this many days of already-stored bars on an incremental fetch: yfinance
# revises the most recent bars and dividends post adj_close retroactively.
_REFETCH_OVERLAP_DAYS = 7


def fetch_and_store(
    universe: list[str],
    start: _dt.date | str = config.START_DATE,
    end: _dt.date | str | None = None,
    db_path=None,
    full: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Download OHLCV, compute features, and persist all tables.

    Incremental: only bars newer than what is already stored (minus a small overlap)
    are fetched from the network, then merged with the full stored history so feature
    windows (SMA-200 etc.) keep their depth and the model trains on the whole series.
    An empty/partial database falls back to a full fetch from ``start`` (self-healing);
    ``full=True`` forces that path (e.g. after a stock split rescales deep history).

    Returns ``(market_with_features, cross_asset_frame, raw)`` where ``market`` is
    restricted to the portfolio stocks (the ML/forecast targets) and ``raw`` is the
    full per-ticker OHLCV history (reused for benchmark + correlation).
    """
    conn = store.connect(db_path)

    # Decide the fetch window: incremental tail when every ticker is already stored,
    # else a full history pull from ``start``.
    stored = {} if full else store.latest_date(conn)
    if stored and all(t in stored for t in universe):
        seam = min(pd.to_datetime(d) for d in stored.values())
        fetch_start: _dt.date | str = (seam - pd.Timedelta(days=_REFETCH_OVERLAP_DAYS)).date()
    else:
        fetch_start = start

    tail = fetch.download_ohlcv(universe, start=fetch_start, end=end)

    # Persist the freshly-fetched bars (overlap rows refreshed idempotently).
    for ticker, df in tail.items():
        out = df.reset_index()
        out.insert(1, "ticker", ticker)
        store.upsert(conn, "ohlcv_raw", out)

    # Merge stored history + fresh tail: read the full series back so features and
    # training see full depth (the tail alone is too short for the long MAs).
    raw = store.read_ohlcv(conn, universe) or tail

    # Per-stock technical features -> daily_market.
    market: dict[str, pd.DataFrame] = {}
    market_cols = [
        "open", "high", "low", "close", "volume",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_12", "ema_26", "ema_50", "rsi_14", "macd", "macd_signal",
        "bollinger_upper", "bollinger_lower",
        "price_sma50_ratio", "price_sma200_ratio", "sma50_sma200_ratio",
        "volume_sma20_ratio",
    ]
    for ticker in tickers.STOCKS:
        if ticker not in raw:
            continue
        feat = technical.add_technical_features(raw[ticker])
        market[ticker] = feat
        out = feat.reset_index()
        out.insert(1, "ticker", ticker)
        keep = ["date", "ticker"] + [c for c in market_cols if c in out.columns]
        store.upsert(conn, "daily_market", out[keep])

    # Cross-asset features -> cross_asset.
    closes = {t: df["close"] for t, df in raw.items()}
    cross = cross_asset.build_cross_asset_frame(closes)
    store.upsert(conn, "cross_asset", cross.reset_index())

    conn.close()
    return market, cross, raw


def _fetch_sentiment(
    stocks: list[str], as_of: _dt.date | None = None, db_path=None
) -> pd.DataFrame:
    """Fetch live analyst/sentiment rows for ``stocks`` (dated ``as_of``) and persist."""
    rows = [analyst.build_sentiment_row(t, as_of=as_of) for t in stocks]
    frame = pd.concat(rows, ignore_index=True)
    conn = store.connect(db_path)
    store.upsert(conn, "sentiment_analyst", frame)
    conn.close()
    return frame


def daily_etl(
    start: _dt.date | str = config.START_DATE,
    end: _dt.date | str | None = None,
    with_sentiment: bool = True,
    as_of: _dt.date | None = None,
    db_path=None,
    full: bool = False,
) -> dict:
    """Daily cron ETL (Phase 5): persist OHLCV + features + cross-asset, plus a dated
    sentiment snapshot so the live analyst signals accumulate history (the prerequisite
    to making them trainable). Incremental by default; ``full=True`` forces a full
    re-fetch. Returns a summary dict for the cron log."""
    market, cross, raw = fetch_and_store(
        tickers.ALL_TICKERS, start=start, end=end, db_path=db_path, full=full
    )
    sentiment_rows = 0
    if with_sentiment:
        sent = _fetch_sentiment(list(market), as_of=as_of, db_path=db_path)
        sentiment_rows = len(sent)
    return {
        "as_of": str(as_of or _dt.date.today()),
        "tickers": len(raw),
        "stocks": len(market),
        "cross_rows": len(cross),
        "sentiment_rows": sentiment_rows,
    }


def _live_snapshots(
    panel: pd.DataFrame, sentiment_df: pd.DataFrame
) -> dict[str, pd.Series]:
    """Latest feature row per stock, enriched with live sentiment when available."""
    by_ticker = sentiment_df.set_index("ticker") if not sentiment_df.empty else None
    snapshots: dict[str, pd.Series] = {}
    for ticker in panel.index.get_level_values("ticker").unique():
        sub = panel.xs(ticker, level="ticker")
        snap = sub.iloc[-1].copy()
        if by_ticker is not None and ticker in by_ticker.index:
            row = by_ticker.loc[ticker]
            snap["news_sentiment_score"] = row["news_sentiment_score"] or 0.0
            snap["put_call_ratio"] = row["put_call_ratio"] or 0.0
        snapshots[ticker] = snap
    return snapshots


def _book_cash_held(state) -> tuple[float, dict[tuple[str, str], float]]:
    """(cash, {(ticker, leverage_label): held value}) from the backtest's final book,
    for sizing the live forecast. Falls back to the base €100k cash / empty book."""
    if state is None:
        return config.INITIAL_CAPITAL_EUR, {}
    label = {1: "stock", 2: "2x", 3: "3x"}
    held: dict[tuple[str, str], float] = {}
    for lot in state.lots:
        key = (lot.ticker, label[lot.tier])
        held[key] = held.get(key, 0.0) + lot.value
    return state.cash, held


def _correlation_matrix(raw_closes: dict[str, pd.Series], window: int = 60) -> pd.DataFrame:
    df = pd.DataFrame({t: s for t, s in raw_closes.items()})
    df.index = pd.to_datetime(df.index)
    rets = df.pct_change().tail(window)
    return rets.corr()


def run_phase1(
    start: _dt.date | str = config.START_DATE,
    end: _dt.date | str | None = None,
    n_dataset: int = 4_000,
    horizon: int = 20,
    with_sentiment: bool = True,
    tune: bool = True,
    strategy: str = "default",
    db_path=None,
    progress=None,
) -> Phase1Result:
    """Run the full slice over the full ticker universe; return artifacts for UI/CLI.

    ``strategy`` selects the backtest/forecast book: ``"default"`` (the guardrailed
    90/10 base case) or ``"aggressive"`` (the all-3x book; see ``run_aggressive_backtest``).
    ``progress(fraction, label)`` is an optional callback the UI uses to drive a progress
    bar through the main steps; it is a no-op when ``None`` (CLI)."""
    def _tick(fraction: float, label: str) -> None:
        if progress is not None:
            progress(fraction, label)

    universe = tickers.ALL_TICKERS
    _tick(0.05, "Fetching market data…")
    market, cross, raw = fetch_and_store(universe, start=start, end=end, db_path=db_path)

    # Feature panel + synthetic dataset + model. Train only on the pre-validation
    # split so the validation-window backtest is honest out-of-sample.
    _tick(0.30, "Building dataset & training model…")
    panel = dataset.build_feature_panel(market, cross)
    prices = {t: pd.Series(df["close"].values, index=pd.to_datetime(df.index))
              for t, df in market.items()}
    X, y = dataset.generate_dataset(panel, prices, n=n_dataset, horizon=horizon)
    X_tr, y_tr, _, _ = dataset.train_validate_split(X, y)
    trained = (model.tune_and_train(X_tr, y_tr) if tune else model.train(X_tr, y_tr))

    # Rules + forecast backtest over the validation window (last VALIDATION_YEARS).
    # Run it first so the live forecast can be sized against the evolved book (cash on
    # hand + open positions), not a notional €100k.
    _tick(0.70, "Running backtest…")
    val_start = (pd.Timestamp(market[next(iter(market))].index[-1])
                 - pd.DateOffset(years=config.VALIDATION_YEARS)).strftime("%Y-%m-%d")
    nasdaq = (raw[config.BENCHMARK_TICKER]["close"]
              if config.BENCHMARK_TICKER in raw
              else pd.DataFrame({t: df["close"] for t, df in market.items()}).mean(axis=1))
    run_bt = run_aggressive_backtest if strategy == "aggressive" else run_forecast_backtest
    bt = run_bt(market, nasdaq, trained, panel, start=val_start)

    # Live analyst/sentiment, then forecast from the latest snapshots. The sentiment
    # overlay tilts the live forecast only (these signals have no history to backtest);
    # apply_book_limits then caps buys at cash and sells at the held tier (Story.md).
    _tick(0.85, "Live sentiment & forecast…")
    sentiment_df = (_fetch_sentiment(list(market), db_path=db_path)
                    if with_sentiment else pd.DataFrame())
    snaps = _live_snapshots(panel, sentiment_df)
    book_value = bt.final_state.total_value() if bt.final_state else config.INITIAL_CAPITAL_EUR
    leverages = (3,) if strategy == "aggressive" else config.LEVERAGE_TIERS
    forecasts = forecast.forecast(trained, snaps, portfolio_value=book_value,
                                  leverages=leverages)
    latest_close = {t: float(df["close"].iloc[-1]) for t, df in market.items()}
    forecasts = overlay.apply_overlay(forecasts, sentiment_df, latest_close)
    forecasts = forecast.apply_book_limits(forecasts, *_book_cash_held(bt.final_state))

    _tick(0.95, "Correlation & regime…")
    corr = _correlation_matrix({t: df["close"] for t, df in raw.items()})
    reg = (regime.detect_regime(
               raw["^GSPC"]["close"], raw["^VIX"]["close"],
               {t: df["close"] for t, df in market.items()},
               gold=raw["GC=F"]["close"] if "GC=F" in raw else None,
               oil=raw["CL=F"]["close"] if "CL=F" in raw else None)
           if "^GSPC" in raw and "^VIX" in raw else None)
    _tick(1.0, "Done")
    return Phase1Result(
        market=market, cross=cross, model=trained,
        forecasts=forecasts, backtest=bt, correlation=corr,
        sentiment=sentiment_df, nasdaq=nasdaq, panel=panel, regime=reg,
    )


def _lot_value_path(closes: pd.Series, tier: int, invested: float, entry) -> pd.Series:
    """Value path of an ``invested``-EUR tier-``tier`` lot opened at ``entry``.

    Simple-leverage model (Live tab): ``value = invested * (1 + tier * perf)`` where
    ``perf`` is the underlying's **total** % return since the buy date — i.e. the
    underlying performance scaled by the leverage, *not* daily-rebalanced compounding.
    Floored at 0 (a long position can't be worth less than nothing). Empty if ``entry``
    is past the last close."""
    held = closes[closes.index >= pd.Timestamp(entry)]
    if held.empty:
        return pd.Series(dtype=float)
    perf = held / held.iloc[0] - 1.0  # total underlying performance since entry
    return (invested * (1.0 + tier * perf)).clip(lower=0.0)


def _dated_lots_and_paths(positions, market: dict[str, pd.DataFrame]):
    """``(lots, value-paths)`` for the funded, in-universe positions — shared by
    `build_dated_book` and `dated_book_value_path`. Each lot's cost basis is the invested
    amount; its value path is `_lot_value_path` (empty when the buy date is past the last
    close, in which case the lot's current value falls back to its cost)."""
    from .portfolio.state import Lot

    if isinstance(positions, pd.DataFrame):
        positions = positions.to_dict("records")
    lots: list[Lot] = []
    paths: list[pd.Series] = []
    for p in positions:
        ticker = str(p["ticker"])
        invested = float(p.get("invested_eur") or 0.0)
        if invested <= 0 or ticker not in market:
            continue
        tier = int(p["tier"])
        df = market[ticker]
        closes = pd.Series(df["close"].values, index=pd.to_datetime(list(df.index))).sort_index()
        # Drop NaN closes so a single stale/missing bar (yfinance occasionally returns a
        # NaN close for the latest day) can't poison the lot's value -> the whole book's
        # current value going NaN and the position vanishing from the pie.
        closes = closes.dropna()
        path = _lot_value_path(closes, tier, invested, p["buy_date"])
        current = float(path.iloc[-1]) if not path.empty else invested
        lots.append(Lot(ticker, tier, invested, current, tp_basis=invested))
        if not path.empty:
            paths.append(path)
    return lots, paths


def _combine_paths(paths: list[pd.Series], cash: float) -> pd.Series:
    """Daily combined book value (cash + each lot; ffill holds across gaps, leading
    pre-entry NaN -> 0). Empty when no lot has an in-history path."""
    if not paths:
        return pd.Series(dtype=float)
    idx = paths[0].index
    for p in paths[1:]:
        idx = idx.union(p.index)
    total = pd.Series(float(cash), index=idx)
    for p in paths:
        total = total.add(p.reindex(idx).ffill().fillna(0.0), fill_value=0.0)
    return total


def build_dated_book(positions, market: dict[str, pd.DataFrame], cash: float):
    """Build a ``PortfolioState`` from **per-position dated invested amounts**.

    ``positions`` is a DataFrame or iterable of mappings with ``ticker, tier,
    invested_eur, buy_date`` — the EUR invested in one ``(stock, tier)`` on **its own buy
    date** (each tier's date is evaluated separately). A lot's **cost basis** is the
    invested amount (the real tax basis), its **current value** is ``invested * (1 + tier *
    underlying-%-return-since-buy-date)`` (simple leverage, `_lot_value_path`), and its
    take-profit reference starts at cost. The book's **high-water** is the peak of the
    combined daily value (cash + lots, 0 before each lot's buy date, held flat across
    gaps), so the drawdown guardrail is meaningful. Pure (no network) — derives everything
    from the already-fetched prices."""
    from .portfolio.state import PortfolioState

    lots, paths = _dated_lots_and_paths(positions, market)
    state = PortfolioState(cash=float(cash), lots=lots)
    # High-water always includes "now" (the current book value): lots whose buy date is
    # beyond their ticker's last close have an empty path and aren't in `total`, so the
    # historical max alone could sit below current and yield a spurious negative drawdown.
    total = _combine_paths(paths, cash)
    peak = state.total_value()
    if not total.empty:
        peak = max(peak, float(total.max()))
    state.high_water = peak
    return state


def dated_book_value_path(positions, market: dict[str, pd.DataFrame], cash: float) -> pd.Series:
    """Daily combined €-value of a dated user book since its earliest buy date — the same
    series whose peak sets `build_dated_book`'s high-water. Empty when no position has an
    in-history path. Pure (no network) — for the Live tab's performance-vs-NASDAQ chart."""
    _, paths = _dated_lots_and_paths(positions, market)
    return _combine_paths(paths, cash)


def _strategy_actions(state, trained, panel, strategy):
    """Deterministic, value/cost-aware actions for ``state`` under ``strategy`` (mutates
    the passed copy). Default: the full guardrails (drawdown de-risk now fires off the
    derived high-water, plus underlying-dominance and the 33% trim). Aggressive: the
    real lot-level stop-loss (−60% vs cost) + take-profit (+60% skim) + 33% cap."""
    from .backtest import engine
    from .portfolio import rules

    if strategy != "aggressive":
        return rules.apply_guardrails(state)
    last = panel.index.get_level_values("date").unique().sort_values()[-5:]
    sell_conf = engine._name_confidence(trained, panel, pd.DatetimeIndex(last),
                                        is_sell=1, leverage=3)
    sell_row = sell_conf.iloc[-1] if not sell_conf.empty else None
    return (engine._agg_stop_loss(state)
            + engine._agg_take_profit(state, sell_row)
            + rules.trim_overweight(state))


def recommend_for_portfolio(
    state,
    trained: model.TrainedModel,
    panel: pd.DataFrame,
    market: dict[str, pd.DataFrame],
    *,
    strategy: str = "default",
    with_sentiment: bool = True,
    db_path=None,
) -> tuple[list[Forecast], pd.DataFrame, list]:
    """Live action recommendations for a **user-supplied** book under ``strategy``.

    Reuses an already-trained model (no retraining): fetches live analyst/sentiment,
    scores the latest market snapshot per stock, sizes the 5-field forecast to the user's
    book (cash on hand / open positions), tilts it by the sentiment overlay, and adds the
    strategy's deterministic value/cost-aware actions (`_strategy_actions`). Side-effect-
    free — the actions run on a copy of ``state``. Returns
    ``(forecasts, sentiment_df, action_trades)``."""
    import copy

    sentiment_df = (_fetch_sentiment(list(market), db_path=db_path)
                    if with_sentiment else pd.DataFrame())
    snaps = _live_snapshots(panel, sentiment_df)
    latest_close = {t: float(df["close"].iloc[-1]) for t, df in market.items()}
    leverages = (3,) if strategy == "aggressive" else config.LEVERAGE_TIERS
    fcs = forecast.forecast(trained, snaps, portfolio_value=state.total_value(),
                            leverages=leverages)
    fcs = overlay.apply_overlay(fcs, sentiment_df, latest_close)
    cash, held = _book_cash_held(state)
    fcs = forecast.apply_book_limits(fcs, cash, held)
    actions = _strategy_actions(copy.deepcopy(state), trained, panel, strategy)
    return fcs, sentiment_df, actions


def _nasdaq_series(raw: dict, market: dict) -> pd.Series:
    """NASDAQ close, falling back to the stock-basket mean if the benchmark is absent."""
    if config.BENCHMARK_TICKER in raw:
        return raw[config.BENCHMARK_TICKER]["close"]
    return pd.DataFrame({t: df["close"] for t, df in market.items()}).mean(axis=1)


def run_walkforward(
    start: _dt.date | str = config.START_DATE,
    end: _dt.date | str | None = None,
    n_dataset: int = 10_000,
    horizon: int = 20,
    n_windows: int = 4,
    window: int = 252,
    tune: bool = True,
    db_path=None,
) -> WalkForwardResult:
    """Fetch the universe and run walk-forward validation vs NASDAQ."""
    market, cross, raw = fetch_and_store(tickers.ALL_TICKERS, start=start, end=end, db_path=db_path)
    panel = dataset.build_feature_panel(market, cross)
    prices = {t: pd.Series(df["close"].values, index=pd.to_datetime(df.index))
              for t, df in market.items()}
    return walk_forward_validate(
        market, _nasdaq_series(raw, market), panel, prices,
        n_windows=n_windows, window=window, n_dataset=n_dataset, horizon=horizon, tune=tune,
    )

"""End-to-end Phase 1 orchestration: fetch -> features -> store -> ML -> forecast.

``run_phase1`` runs the full thin slice and returns everything the UI / CLI needs.
``fetch_and_store`` is the daily-ETL building block reused by the cron job later.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import pandas as pd

from . import config
from .backtest.engine import BacktestResult, run_forecast_backtest
from .backtest.walkforward import WalkForwardResult, walk_forward_validate
from .data import fetch, store, tickers
from .features import analyst, cross_asset, technical
from .ml import dataset, forecast, model
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


def fetch_and_store(
    universe: list[str],
    start: _dt.date | str = config.START_DATE,
    end: _dt.date | str | None = None,
    db_path=None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Download OHLCV, compute features, and persist all tables.

    Returns ``(market_with_features, cross_asset_frame, raw)`` where ``market`` is
    restricted to the portfolio stocks (the ML/forecast targets) and ``raw`` is the
    full per-ticker OHLCV download (reused for benchmark + correlation).
    """
    raw = fetch.download_ohlcv(universe, start=start, end=end)
    conn = store.connect(db_path)

    # Persist raw OHLCV.
    for ticker, df in raw.items():
        out = df.reset_index()
        out.insert(1, "ticker", ticker)
        store.upsert(conn, "ohlcv_raw", out)

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


def _fetch_sentiment(stocks: list[str], db_path=None) -> pd.DataFrame:
    """Fetch live analyst/sentiment rows for ``stocks`` and persist them."""
    rows = [analyst.build_sentiment_row(t) for t in stocks]
    frame = pd.concat(rows, ignore_index=True)
    conn = store.connect(db_path)
    store.upsert(conn, "sentiment_analyst", frame)
    conn.close()
    return frame


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
    db_path=None,
) -> Phase1Result:
    """Run the full slice over the full ticker universe; return artifacts for UI/CLI."""
    universe = tickers.ALL_TICKERS
    market, cross, raw = fetch_and_store(universe, start=start, end=end, db_path=db_path)

    # Feature panel + synthetic dataset + model. Train only on the pre-validation
    # split so the validation-window backtest is honest out-of-sample.
    panel = dataset.build_feature_panel(market, cross)
    prices = {t: pd.Series(df["close"].values, index=pd.to_datetime(df.index))
              for t, df in market.items()}
    X, y = dataset.generate_dataset(panel, prices, n=n_dataset, horizon=horizon)
    X_tr, y_tr, _, _ = dataset.train_validate_split(X, y)
    trained = (model.tune_and_train(X_tr, y_tr) if tune else model.train(X_tr, y_tr))

    # Live analyst/sentiment, then forecast from the latest snapshots.
    sentiment_df = (_fetch_sentiment(list(market), db_path=db_path)
                    if with_sentiment else pd.DataFrame())
    snaps = _live_snapshots(panel, sentiment_df)
    forecasts = forecast.forecast(trained, snaps)

    # Rules + forecast backtest over the validation window (last VALIDATION_YEARS).
    val_start = (pd.Timestamp(market[next(iter(market))].index[-1])
                 - pd.DateOffset(years=config.VALIDATION_YEARS)).strftime("%Y-%m-%d")
    nasdaq = (raw[config.BENCHMARK_TICKER]["close"]
              if config.BENCHMARK_TICKER in raw
              else pd.DataFrame({t: df["close"] for t, df in market.items()}).mean(axis=1))
    bt = run_forecast_backtest(market, nasdaq, trained, panel, start=val_start)

    corr = _correlation_matrix({t: df["close"] for t, df in raw.items()})
    return Phase1Result(
        market=market, cross=cross, model=trained,
        forecasts=forecasts, backtest=bt, correlation=corr,
        sentiment=sentiment_df,
    )


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

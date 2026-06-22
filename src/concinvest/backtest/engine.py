"""Phase 1 backtest: model-timed concentrated portfolio vs. NASDAQ.

A deliberately simple, point-in-time strategy to validate the end-to-end loop:
hold an equal-weight basket of the 5 stocks, but scale daily equity *exposure* by
the model's average buy-confidence (the rest sits in cash). Exposure is lagged one
day to avoid look-ahead. Returns cumulative value vs. a NASDAQ buy-and-hold.

Full allocation/risk/leverage/tax logic (Story.md) lands in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import config
from ..ml.dataset import FEATURE_COLS
from ..ml.model import TrainedModel
from ..portfolio import rules
from ..portfolio import state as pstate


@dataclass
class BacktestResult:
    curve: pd.DataFrame  # index=date, columns: portfolio, benchmark
    portfolio_return: float
    benchmark_return: float

    @property
    def outperformance(self) -> float:
        return self.portfolio_return - self.benchmark_return

    @property
    def beats_benchmark(self) -> bool:
        return self.portfolio_return > self.benchmark_return


def _daily_exposure(model: TrainedModel, panel: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Mean P(buy profitable, leverage=1) across stocks per date, lagged 1 day."""
    rows = panel.loc[panel.index.get_level_values("date").isin(dates)].copy()
    if rows.empty:
        return pd.Series(1.0, index=dates)
    feats = rows.reindex(columns=FEATURE_COLS).copy()
    feats["is_sell"] = 0
    feats["leverage"] = 1
    feats = feats.fillna(0.0)
    conf = model.predict_confidence(feats)
    rows = rows.assign(_conf=conf)
    expo = rows.groupby(level="date")["_conf"].mean()
    expo = expo.reindex(dates).ffill().fillna(1.0)
    return expo.shift(1).bfill().clip(0.0, 1.0)


def run_backtest(
    market: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    model: TrainedModel,
    panel: pd.DataFrame,
    start: str | None = None,
) -> BacktestResult:
    """Run the Phase 1 backtest over the window starting at ``start``."""
    # Equal-weight daily returns across available stocks.
    closes = pd.DataFrame({t: df["close"] for t, df in market.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    if start:
        closes = closes.loc[start:]
    rets = closes.pct_change().mean(axis=1)  # equal-weight basket return
    dates = pd.DatetimeIndex(rets.index)

    exposure = _daily_exposure(model, panel, dates)
    strat_ret = (rets * exposure).fillna(0.0)
    portfolio = config.INITIAL_CAPITAL_EUR * (1.0 + strat_ret).cumprod()

    bench = benchmark_close.copy()
    bench.index = pd.to_datetime(bench.index)
    bench = bench.reindex(dates).ffill()
    benchmark = config.INITIAL_CAPITAL_EUR * (bench / bench.iloc[0])

    curve = pd.DataFrame({"portfolio": portfolio, "benchmark": benchmark}).dropna()
    p_ret = float(curve["portfolio"].iloc[-1] / curve["portfolio"].iloc[0] - 1.0)
    b_ret = float(curve["benchmark"].iloc[-1] / curve["benchmark"].iloc[0] - 1.0)
    return BacktestResult(curve=curve, portfolio_return=p_ret, benchmark_return=b_ret)


def _benchmark_curve(benchmark_close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """NASDAQ buy-and-hold rebased to the initial capital over ``dates``."""
    bench = benchmark_close.copy()
    bench.index = pd.to_datetime(bench.index)
    bench = bench.reindex(dates).ffill()
    return config.INITIAL_CAPITAL_EUR * (bench / bench.iloc[0])


def run_rules_backtest(
    market: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    start: str | None = None,
    capital: float = config.INITIAL_CAPITAL_EUR,
) -> BacktestResult:
    """Replay the Story.md base-case leveraged book under the daily risk guardrails.

    Starts from the 90/10 base case (per-name 12%/3%/3% stock/2x/3x), marks every
    lot to market each day, then applies the sell-side guardrails (drawdown de-risk,
    per-name trim, 10%/day cap) with German tax on realized gains. No re-entry yet —
    forecast-driven buys/sells are the next Phase 4 increment.
    """
    closes = pd.DataFrame({t: df["close"] for t, df in market.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    if start:
        closes = closes.loc[start:]
    rets = closes.pct_change().fillna(0.0)
    dates = pd.DatetimeIndex(rets.index)

    state = pstate.build_base_case(capital, stocks=list(market))
    values: list[float] = []
    for _, row in rets.iterrows():
        state.mark(row.to_dict())
        rules.apply_guardrails(state)
        values.append(state.total_value())

    portfolio = pd.Series(values, index=dates)
    benchmark = _benchmark_curve(benchmark_close, dates)
    curve = pd.DataFrame({"portfolio": portfolio, "benchmark": benchmark}).dropna()
    p_ret = float(curve["portfolio"].iloc[-1] / curve["portfolio"].iloc[0] - 1.0)
    b_ret = float(curve["benchmark"].iloc[-1] / curve["benchmark"].iloc[0] - 1.0)
    return BacktestResult(curve=curve, portfolio_return=p_ret, benchmark_return=b_ret)

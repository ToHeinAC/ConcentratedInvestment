"""Phase 1 backtest: model-timed concentrated portfolio vs. NASDAQ.

A deliberately simple, point-in-time strategy to validate the end-to-end loop:
hold an equal-weight basket of the 5 stocks, but scale daily equity *exposure* by
the model's average buy-confidence (the rest sits in cash). Exposure is lagged one
day to avoid look-ahead. Returns cumulative value vs. a NASDAQ buy-and-hold.

Full allocation/risk/leverage/tax logic (Story.md) lands in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    trades: list[rules.Trade] = field(default_factory=list)  # forecast backtest only
    final_state: pstate.PortfolioState | None = None  # end-of-window book (forecast bt)

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


def _name_confidence(
    model: TrainedModel, panel: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per-stock P(buy profitable, leverage=1) as a date×ticker frame, lagged 1 day.

    Unlike ``_daily_exposure`` (which averages across stocks into one series), this
    keeps each name's own confidence so the forecast backtest can trade names
    independently (Story.md: the forecast is per ticker)."""
    rows = panel.loc[panel.index.get_level_values("date").isin(dates)].copy()
    if rows.empty:
        return pd.DataFrame(index=dates)
    feats = rows.reindex(columns=FEATURE_COLS).copy()
    feats["is_sell"] = 0
    feats["leverage"] = 1
    feats = feats.fillna(0.0)
    conf = pd.Series(model.predict_confidence(feats), index=rows.index)
    wide = conf.unstack("ticker").reindex(dates).ffill().fillna(1.0)
    return wide.shift(1).bfill().clip(0.0, 1.0)


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


def _dividend_yields(
    market: dict[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per-day dividend yield per stock = total-return minus price return.

    With ``auto_adjust=False``, ``adj_close`` is the dividend/split-adjusted total
    return while ``close`` is price-only, so their daily return difference recovers
    the dividend yield (Story.md: dividends accrue to the underlying only). Stocks
    without an ``adj_close`` column contribute zero.
    """
    cols = {}
    for ticker, df in market.items():
        if "adj_close" not in df.columns:
            continue
        idx = pd.to_datetime(df.index)
        adj = pd.Series(df["adj_close"].values, index=idx).pct_change()
        price = pd.Series(df["close"].values, index=idx).pct_change()
        cols[ticker] = (adj - price).clip(lower=0.0)
    if not cols:
        return pd.DataFrame(index=dates)
    return pd.DataFrame(cols).reindex(dates).fillna(0.0)


def _benchmark_curve(benchmark_close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """NASDAQ buy-and-hold rebased to the initial capital over ``dates``."""
    bench = benchmark_close.copy()
    bench.index = pd.to_datetime(bench.index)
    # ffill interior gaps; bfill a leading NaN when the window opens on a date the
    # benchmark didn't trade (e.g. a US holiday while EU/JP stocks traded).
    bench = bench.reindex(dates).ffill().bfill()
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

    divs = _dividend_yields(market, dates)
    state = pstate.build_base_case(capital, stocks=list(market))
    values: list[float] = []
    for date, row in rets.iterrows():
        state.mark(row.to_dict())
        state.pay_dividends(divs.loc[date].to_dict() if date in divs.index else {})
        rules.apply_guardrails(state)
        values.append(state.total_value())

    portfolio = pd.Series(values, index=dates)
    benchmark = _benchmark_curve(benchmark_close, dates)
    curve = pd.DataFrame({"portfolio": portfolio, "benchmark": benchmark}).dropna()
    p_ret = float(curve["portfolio"].iloc[-1] / curve["portfolio"].iloc[0] - 1.0)
    b_ret = float(curve["benchmark"].iloc[-1] / curve["benchmark"].iloc[0] - 1.0)
    return BacktestResult(curve=curve, portfolio_return=p_ret, benchmark_return=b_ret)


# Classifier-neutral confidence: at/above this the model is not bearish, so the
# book stays fully at the base-case allocation; below it we de-risk toward cash.
_NEUTRAL_CONF: float = 0.5


def _target_exposure(confidence: float) -> float:
    """Base-case-faithful target equity fraction from mean buy-confidence.

    Holds the 90% base case while the model is neutral-to-bullish
    (``confidence >= 0.5``); only a bearish read (< 0.5) scales exposure down
    proportionally toward cash. The drawdown guardrail de-risks crashes separately.
    """
    return config.BASE_STOCK_ALLOCATION * min(1.0, confidence / _NEUTRAL_CONF)


# Per-name base weight (12%+3%+3%) and a dead-band scaled from the book-level
# REBALANCE_BAND by the per-name share of the base allocation, so a single name's
# rebalance fires at the same relative sensitivity as the old aggregate dial.
_PER_NAME_BASE: float = sum(config.BASE_PER_NAME_SPLIT.values())
_PER_NAME_BAND: float = config.REBALANCE_BAND * _PER_NAME_BASE / config.BASE_STOCK_ALLOCATION


def _target_name_fraction(confidence: float) -> float:
    """Base-case-faithful target portfolio fraction for one name from its own
    buy-confidence (holds the per-name base while neutral-to-bullish, de-risks below)."""
    return _PER_NAME_BASE * min(1.0, confidence / _NEUTRAL_CONF)


def _rebalance_names_to_target(
    state: pstate.PortfolioState, targets: dict[str, float], stocks: list[str]
) -> list[rules.Trade]:
    """Nudge **each name** toward its own target portfolio fraction, within the daily
    move cap and a per-name dead-band. Names are handled independently (Story.md), so a
    bearish read on one stock trims only that stock. Returns the trades performed."""
    total = state.total_value()
    if total <= 0:
        return []
    trades: list[rules.Trade] = []
    for ticker in stocks:
        dev = targets.get(ticker, _PER_NAME_BASE) - state.name_value(ticker) / total
        if abs(dev) < _PER_NAME_BAND:
            continue
        move = min(abs(dev), config.MAX_DAILY_SELL) * total  # cap daily turnover
        if dev > 0:
            trades += _deploy_name(state, ticker, move)
        elif state.sell_name(ticker, move) > 0:
            trades.append(rules.Trade(ticker, "sell", move))
    return trades


def _is_crisis(basket_ret: pd.Series, i: int) -> bool:
    """True if the basket fell more than ``CRISIS_DROP`` over the trailing lookback."""
    if i + 1 < config.CRISIS_LOOKBACK:
        return False
    window = basket_ret.iloc[i + 1 - config.CRISIS_LOOKBACK : i + 1]
    return float((1.0 + window).prod() - 1.0) <= -config.CRISIS_DROP


def _deploy_name(
    state: pstate.PortfolioState, ticker: str, amount: float
) -> list[rules.Trade]:
    """Deploy ``amount`` of cash into one name by the base-case tier split (12/3/3).
    Returns one aggregate buy ``Trade`` if any cash was actually invested."""
    split = config.BASE_PER_NAME_SPLIT
    weight_sum = sum(split.values())
    invested = 0.0
    for tier_name, weight in split.items():
        invested += state.buy(ticker, rules._TIER_OF[tier_name], amount * weight / weight_sum)
    return [rules.Trade(ticker, "buy", invested)] if invested > 0 else []


def _deploy(state: pstate.PortfolioState, amount: float, stocks: list[str]) -> list[rules.Trade]:
    """Deploy ``amount`` of cash equally across stocks by the base-case tier split
    (crisis buy-the-dip). One aggregate buy ``Trade`` per stock actually funded."""
    per_stock = amount / len(stocks)
    trades: list[rules.Trade] = []
    for ticker in stocks:
        trades += _deploy_name(state, ticker, per_stock)
    return trades


def run_forecast_backtest(
    market: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    model: TrainedModel,
    panel: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    capital: float = config.INITIAL_CAPITAL_EUR,
) -> BacktestResult:
    """Rules + forecast backtest: the base-case leveraged book where **each name's**
    target portfolio fraction tracks that name's own buy-confidence (lagged, scaled by
    its per-name base weight), with cash re-entry, daily guardrails, and German tax.
    Names are rebalanced independently (Story.md: the forecast is per ticker), so a
    bearish read on one stock trims only that stock. ``start``/``end`` bound the window
    (inclusive)."""
    closes = pd.DataFrame({t: df["close"] for t, df in market.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    if start is not None or end is not None:
        closes = closes.loc[start:end]
    rets = closes.pct_change().fillna(0.0)
    dates = pd.DatetimeIndex(rets.index)

    name_conf = _name_confidence(model, panel, dates)  # per-stock buy-confidence, lagged
    basket_ret = rets.mean(axis=1)  # equal-weight basket return, for crisis detection
    divs = _dividend_yields(market, dates)
    stocks = list(market)
    state = pstate.build_base_case(capital, stocks=stocks)
    crisis_day: int | None = None
    values: list[float] = []
    trades: list[rules.Trade] = []
    for i, (date, row) in enumerate(rets.iterrows()):
        state.mark(row.to_dict())
        state.pay_dividends(divs.loc[date].to_dict() if date in divs.index else {})
        day = rules.trim_overweight(state)  # per-name cap applies even in crisis
        in_crisis = crisis_day is not None and (i - crisis_day) < config.CRISIS_REVERT_DAYS
        if not in_crisis:
            crisis_day = None
            day += rules.drawdown_derisk(state)  # riskiest-tier-first de-risk
            if _is_crisis(basket_ret, i):
                crisis_day = i
                day += _deploy(state, state.cash, stocks)  # buy the dip
            else:
                conf_row = name_conf.loc[date] if date in name_conf.index else None
                targets = {
                    t: _target_name_fraction(
                        float(conf_row[t])
                        if conf_row is not None and t in conf_row else 1.0
                    )
                    for t in stocks
                }
                day += _rebalance_names_to_target(state, targets, stocks)
        # During crisis: stay fully invested (no de-risk, no rebalance toward cash).
        for t in day:
            t.date = date
        trades.extend(day)
        values.append(state.total_value())

    portfolio = pd.Series(values, index=dates)
    benchmark = _benchmark_curve(benchmark_close, dates)
    curve = pd.DataFrame({"portfolio": portfolio, "benchmark": benchmark}).dropna()
    p_ret = float(curve["portfolio"].iloc[-1] / curve["portfolio"].iloc[0] - 1.0)
    b_ret = float(curve["benchmark"].iloc[-1] / curve["benchmark"].iloc[0] - 1.0)
    return BacktestResult(curve=curve, portfolio_return=p_ret, benchmark_return=b_ret,
                          trades=trades, final_state=state)

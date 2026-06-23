"""Deterministic risk guardrails (Story.md).

Sell-side rules that protect the book regardless of the forecast:

- **Per-name trim** — when a name (stock + 2x + 3x) exceeds 33% of the portfolio,
  trim 3% of portfolio value from it.
- **Drawdown de-risk** — beyond a 20% drawdown from the high-water mark, sell down
  toward cash, drawing each name's daily sell from the **riskiest tier first**
  (3x → 2x → stock) so the most damaging leverage is cut first.
- **Daily sell cap** — every individual sell is limited to <10% of portfolio/day
  (Story.md), so a full de-risk may take several days.

The crisis 100%/2-month-revert path and forecast-driven "optimal weighting" are layered
on by the backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import config
from .state import PortfolioState

_TIER_OF = {"stock": 1, "2x": 2, "3x": 3}


@dataclass
class Trade:
    ticker: str
    action: str  # "buy" | "sell"
    amount_eur: float  # gross market value transacted
    tier: int | None = None  # 1/2/3 when tier-specific (throttle, de-risk, deploy)
    date: pd.Timestamp | None = None  # stamped by the backtest loop


def _names(state: PortfolioState) -> list[str]:
    return sorted({lot.ticker for lot in state.lots})


def _tier_value(state: PortfolioState, ticker: str, tier: int) -> float:
    return sum(lot.value for lot in state.lots if lot.ticker == ticker and lot.tier == tier)


def _capped_sell(state: PortfolioState, ticker: str, target: float) -> Trade | None:
    """Sell ``target`` EUR of ``ticker``, clamped to the 10%/day cap and holdings."""
    cap = config.MAX_DAILY_SELL * state.total_value()
    gross = min(target, cap, state.name_value(ticker))
    if gross <= 0:
        return None
    state.sell_name(ticker, gross)
    return Trade(ticker, "sell", gross)


def trim_overweight(state: PortfolioState) -> list[Trade]:
    """Trim 3% of portfolio from any name exceeding the 33% per-name cap."""
    total = state.total_value()
    trades: list[Trade] = []
    for ticker in _names(state):
        if state.name_value(ticker) > config.PER_NAME_CAP * total:
            trade = _capped_sell(state, ticker, config.TRIM_FRACTION * total)
            if trade:
                trades.append(trade)
    return trades


def drawdown_derisk(state: PortfolioState) -> list[Trade]:
    """Lever 2: beyond ``MAX_DRAWDOWN`` from the high-water mark, sell each name down by
    up to ``MAX_DAILY_SELL`` of the portfolio/day, drawn from the riskiest tier first
    (3x -> 2x -> stock)."""
    total = state.total_value()
    if state.high_water <= 0:
        return []
    drawdown = (state.high_water - total) / state.high_water
    if drawdown <= config.MAX_DRAWDOWN:
        return []
    trades: list[Trade] = []
    for ticker in _names(state):
        budget = config.MAX_DAILY_SELL * state.total_value()  # 10%/name/day (Story.md)
        for tier in (3, 2, 1):
            if budget <= 0:
                break
            gross = min(_tier_value(state, ticker, tier), budget)
            if gross <= 0:
                continue
            if state.sell_tier(ticker, tier, gross) > 0:
                trades.append(Trade(ticker, "sell", gross, tier=tier))
                budget -= gross
    return trades


def apply_guardrails(state: PortfolioState) -> list[Trade]:
    """Run the daily risk rules in priority order (de-risk, then trim)."""
    return drawdown_derisk(state) + trim_overweight(state)

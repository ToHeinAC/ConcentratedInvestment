"""Deterministic risk guardrails (Story.md).

Sell-side rules that protect the book regardless of the forecast:

- **Per-name trim** — when a name (stock + 2x + 3x) exceeds 33% of the portfolio,
  trim 3% of portfolio value from it.
- **Underlying dominance** — a name's underlying (tier 1) must stay ≥ its leveraged
  tiers (2x + 3x); when a rally lets the leverage outgrow the underlying, sell the
  excess from the **riskiest tier first** (Story.md).
- **Drawdown de-risk** — beyond a 20% drawdown from the high-water mark, sell down
  toward cash, drawing each name's daily sell from the **riskiest tier first**
  (3x → 2x → stock) so the most damaging leverage is cut first, but **never below the
  6% per-name floor** (the retained floor is underlying-only).
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


def sell_riskiest_first(
    state: PortfolioState, ticker: str, budget: float
) -> list[Trade]:
    """Sell up to ``budget`` EUR of ``ticker`` from the **riskiest tier first**
    (3x → 2x → stock), each clamped to holdings. The most damaging leverage is shed
    before the underlying — so a crash sheds 3x first, and an upstreak trim also comes
    out of the leveraged tiers first. Returns one dated-less ``Trade`` per tier sold."""
    trades: list[Trade] = []
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


def trim_overweight(state: PortfolioState) -> list[Trade]:
    """Trim 3% of portfolio from any name exceeding the 33% per-name cap, shedding the
    riskiest tier first and respecting the 10%/day sell cap."""
    total = state.total_value()
    trades: list[Trade] = []
    for ticker in _names(state):
        if state.name_value(ticker) > config.PER_NAME_CAP * total:
            budget = min(config.TRIM_FRACTION * total, config.MAX_DAILY_SELL * total)
            trades += sell_riskiest_first(state, ticker, budget)
    return trades


def enforce_underlying_dominance(state: PortfolioState) -> list[Trade]:
    """Keep each name's underlying (tier 1) ≥ its leveraged tiers (2x + 3x), per Story.md.
    A strong rally compounds the leveraged tiers faster than the underlying; when their
    sum exceeds the underlying, sell the excess from the riskiest tier first (3x → 2x),
    within the 10%/day sell cap (so a large divergence restores over several days)."""
    total = state.total_value()
    trades: list[Trade] = []
    for ticker in _names(state):
        underlying = _tier_value(state, ticker, 1)
        leveraged = _tier_value(state, ticker, 2) + _tier_value(state, ticker, 3)
        excess = leveraged - underlying
        if excess <= 0:
            continue
        budget = min(excess, config.MAX_DAILY_SELL * total)
        trades += sell_riskiest_first(state, ticker, budget)
    return trades


def drawdown_derisk(state: PortfolioState) -> list[Trade]:
    """Lever 2: beyond ``MAX_DRAWDOWN`` from the high-water mark, sell each name down by
    up to ``MAX_DAILY_SELL`` of the portfolio/day, drawn from the riskiest tier first
    (3x -> 2x -> stock), but **never below the ``MIN_NAME_WEIGHT`` per-name floor**
    (Story.md: each stock stays ≥ 6%, held as the underlying)."""
    total = state.total_value()
    if state.high_water <= 0:
        return []
    drawdown = (state.high_water - total) / state.high_water
    if drawdown <= config.MAX_DRAWDOWN:
        return []
    floor = config.MIN_NAME_WEIGHT * total
    trades: list[Trade] = []
    for ticker in _names(state):
        room = state.name_value(ticker) - floor  # don't sell below the 6% floor
        if room <= 0:
            continue
        budget = min(config.MAX_DAILY_SELL * total, room)
        trades += sell_riskiest_first(state, ticker, budget)
    return trades


def apply_guardrails(state: PortfolioState) -> list[Trade]:
    """Run the daily risk rules in priority order (de-risk, dominance, then trim)."""
    return (
        drawdown_derisk(state)
        + enforce_underlying_dominance(state)
        + trim_overweight(state)
    )

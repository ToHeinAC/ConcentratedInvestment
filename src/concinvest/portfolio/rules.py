"""Deterministic risk guardrails (Story.md).

Sell-side rules that protect the book regardless of the forecast:

- **Per-name trim** — when a name (stock + 2x + 3x) exceeds 33% of the portfolio,
  trim 3% of portfolio value from it.
- **Drawdown de-risk** — beyond a 20% drawdown from the high-water mark, sell down
  toward cash.
- **Daily sell cap** — every individual sell is limited to <10% of portfolio/day, so
  a full de-risk may take several days.

Discretionary buys, the crisis 100%/2-month-revert path, and forecast-driven
"optimal weighting" are layered on by the backtest / a later increment.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from .state import PortfolioState


@dataclass
class Trade:
    ticker: str
    action: str  # currently "sell" (guardrails are sell-side)
    amount_eur: float  # gross market value sold


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
    for ticker in sorted({lot.ticker for lot in state.lots}):
        if state.name_value(ticker) > config.PER_NAME_CAP * total:
            trade = _capped_sell(state, ticker, config.TRIM_FRACTION * total)
            if trade:
                trades.append(trade)
    return trades


def drawdown_derisk(state: PortfolioState) -> list[Trade]:
    """If drawdown from the high-water mark exceeds 20%, sell down toward cash."""
    total = state.total_value()
    if state.high_water <= 0:
        return []
    drawdown = (state.high_water - total) / state.high_water
    if drawdown <= config.MAX_DRAWDOWN:
        return []
    trades: list[Trade] = []
    for ticker in sorted({lot.ticker for lot in state.lots}):
        trade = _capped_sell(state, ticker, state.name_value(ticker))
        if trade:
            trades.append(trade)
    return trades


def apply_guardrails(state: PortfolioState) -> list[Trade]:
    """Run the daily risk rules in priority order (de-risk, then trim)."""
    return drawdown_derisk(state) + trim_overweight(state)

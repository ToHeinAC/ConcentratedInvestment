"""Portfolio state: leveraged lots + cash, with daily mark-to-market.

Leverage tiers (1/2/3) are modelled as daily-rebalanced constant-leverage return
multipliers (the documented project assumption): a tier-k lot's value evolves by
``(1 + k * underlying_daily_return)`` each day. Each lot keeps its cost basis (the
EUR originally invested) so partial/full sales can realize gains for German tax.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .. import config
from . import tax as tax_mod

_TIER_OF = {"stock": 1, "2x": 2, "3x": 3}


@dataclass
class Lot:
    ticker: str
    tier: int  # 1 (stock), 2 (2x), 3 (3x)
    cost_basis: float  # EUR originally invested (tax basis)
    value: float  # current EUR market value


@dataclass
class PortfolioState:
    cash: float
    lots: list[Lot] = field(default_factory=list)
    loss_carry: float = 0.0  # carried realized losses for tax offset
    high_water: float = 0.0  # peak total value, for drawdown checks

    def total_value(self) -> float:
        return self.cash + sum(lot.value for lot in self.lots)

    def name_value(self, ticker: str) -> float:
        """Market value of a name across all its tiers (Story.md trim basis)."""
        return sum(lot.value for lot in self.lots if lot.ticker == ticker)

    def mark(self, returns: dict[str, float]) -> None:
        """Apply one day's underlying returns to every lot; update high-water mark."""
        for lot in self.lots:
            lot.value *= 1.0 + lot.tier * returns.get(lot.ticker, 0.0)
        self.high_water = max(self.high_water, self.total_value())

    def pay_dividends(
        self, yields: dict[str, float], rate: float = config.CAPITAL_GAINS_TAX_RATE
    ) -> float:
        """Credit cash with dividends on the underlying (tier-1) lots only, net of the
        flat tax. Leveraged lots (tier > 1) receive no dividend (Story.md)."""
        gross = sum(
            lot.value * yields.get(lot.ticker, 0.0) for lot in self.lots if lot.tier == 1
        )
        net = gross * (1.0 - rate)
        self.cash += net
        return net

    def buy(self, ticker: str, tier: int, amount_eur: float) -> float:
        """Open a lot funded by cash (clamped to available cash). Returns invested EUR."""
        amount = min(amount_eur, self.cash)
        if amount <= 0:
            return 0.0
        self.cash -= amount
        self.lots.append(Lot(ticker, tier, amount, amount))
        return amount

    def sell_name(
        self,
        ticker: str,
        amount_eur: float,
        tax_fn: Callable[[float, float], tuple[float, float]] = tax_mod.tax_on_sale,
    ) -> float:
        """Sell ``amount_eur`` of market value of ``ticker`` proportionally across its
        lots, realize the gain/loss, pay tax, and credit net proceeds to cash."""
        return self._sell_lots(
            [lot for lot in self.lots if lot.ticker == ticker], amount_eur, tax_fn
        )

    def sell_tier(
        self,
        ticker: str,
        tier: int,
        amount_eur: float,
        tax_fn: Callable[[float, float], tuple[float, float]] = tax_mod.tax_on_sale,
    ) -> float:
        """Sell ``amount_eur`` from one ``tier`` of ``ticker`` only (tier-targeted
        de-leveraging / riskiest-first de-risk). Same tax + cash semantics as
        ``sell_name``."""
        return self._sell_lots(
            [lot for lot in self.lots if lot.ticker == ticker and lot.tier == tier],
            amount_eur,
            tax_fn,
        )

    def _sell_lots(
        self,
        lots: list[Lot],
        amount_eur: float,
        tax_fn: Callable[[float, float], tuple[float, float]],
    ) -> float:
        """Sell ``amount_eur`` of value proportionally across ``lots``; realize tax,
        credit net proceeds to cash, and drop emptied lots. Returns net proceeds."""
        name_val = sum(lot.value for lot in lots)
        amount = min(amount_eur, name_val)
        if amount <= 0:
            return 0.0
        frac = amount / name_val
        gross, realized = 0.0, 0.0
        for lot in lots:
            sell_val, basis = lot.value * frac, lot.cost_basis * frac
            realized += sell_val - basis
            lot.value -= sell_val
            lot.cost_basis -= basis
            gross += sell_val
        tax, self.loss_carry = tax_fn(realized, self.loss_carry)
        self.lots = [lot for lot in self.lots if lot.value > 1e-9]
        net = gross - tax
        self.cash += net
        return net


def build_base_case(
    capital: float = config.INITIAL_CAPITAL_EUR,
    stocks: list[str] | None = None,
    split: dict[str, float] | None = None,
) -> PortfolioState:
    """Construct the Story.md base case: per-name 12%/3%/3% stock/2x/3x, 10% cash."""
    from ..data import tickers

    stocks = stocks or tickers.STOCKS
    split = split or config.BASE_PER_NAME_SPLIT
    state = PortfolioState(cash=capital, high_water=capital)
    for ticker in stocks:
        for tier_name, weight in split.items():
            state.buy(ticker, _TIER_OF[tier_name], weight * capital)
    return state

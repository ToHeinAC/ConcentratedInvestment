"""German capital-gains tax (Story.md).

Flat 25% Abgeltungsteuer on net realized gains, with realized losses offsetting
gains before tax. Losses accumulate in a carry pool that future gains draw down.
"""

from __future__ import annotations

from .. import config


def tax_on_sale(
    realized_gain: float,
    loss_carry: float,
    rate: float = config.CAPITAL_GAINS_TAX_RATE,
) -> tuple[float, float]:
    """Tax due and the updated loss carry for a sale realizing ``realized_gain``.

    A loss (``realized_gain < 0``) is added to the carry and incurs no tax; a gain
    is first offset by any carried loss, and only the remainder is taxed at ``rate``.
    Returns ``(tax_due, new_loss_carry)``.
    """
    if realized_gain <= 0:
        return 0.0, loss_carry + (-realized_gain)
    offset = min(loss_carry, realized_gain)
    taxable = realized_gain - offset
    return taxable * rate, loss_carry - offset

"""Options-derived sentiment features.

Put/call open-interest ratio (a classic fear gauge) and the IV skew (OTM-put minus
ATM-call implied vol). The fetches live in ``data.fetch`` to keep all network access
in one place; this module is the feature-facing entrypoint.
"""

from __future__ import annotations

from ..data import fetch


def put_call_ratio(ticker: str) -> float | None:
    """Put/call open-interest ratio for the nearest expiry, or None."""
    return fetch.fetch_put_call_ratio(ticker)


def iv_skew(ticker: str) -> float | None:
    """OTM-put minus ATM-call implied volatility for the nearest expiry, or None."""
    return fetch.fetch_iv_skew(ticker)

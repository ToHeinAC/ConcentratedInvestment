"""Options-derived sentiment features.

Phase 1 exposes the put/call open-interest ratio (a classic fear gauge). IV skew
and max-pain are added in Phase 2. The fetch itself lives in ``data.fetch`` to keep
all network access in one place; this module is the feature-facing entrypoint.
"""

from __future__ import annotations

from ..data import fetch


def put_call_ratio(ticker: str) -> float | None:
    """Put/call open-interest ratio for the nearest expiry, or None."""
    return fetch.fetch_put_call_ratio(ticker)

"""Yahoo Finance ticker universe (from Story.md).

Grouped by role. ``ALL_TICKERS`` is the de-duplicated download list used by the
ETL pipeline. The 5 portfolio stocks are fixed for v1 (later: user-configurable,
still max 5).
"""

from __future__ import annotations

# --- Portfolio stocks (fixed v1, max 5) ----------------------------------
STOCKS: dict[str, str] = {
    "SIE.DE": "Siemens AG",
    "MUV2.DE": "Münchener Rückversicherungs-Gesellschaft AG",
    "FCX": "Freeport-McMoRan Inc.",
    "TSLA": "Tesla, Inc.",
    "8001.T": "ITOCHU Corporation",
}

# --- Global indices (5 picks) --------------------------------------------
INDICES: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^GDAXI": "DAX",
    "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225",
}

# --- Core commodities & macro assets -------------------------------------
COMMODITIES: dict[str, str] = {
    "GC=F": "Gold Futures",
    "CL=F": "WTI Crude Oil",
    "BZ=F": "Brent Crude Oil",
    "ZW=F": "Wheat Futures",
    "HG=F": "Copper Futures",
    "SI=F": "Silver Futures",
    "^SPGSCI": "S&P GSCI Commodity Index",
}

# --- Bonds / rates -------------------------------------------------------
BONDS: dict[str, str] = {
    "^TNX": "10-Year Treasury Yield",
    "^TYX": "30-Year Treasury Yield",
    "^FVX": "5-Year Treasury Yield",
    "^IRX": "13-Week T-Bill",
}

# --- Currency & volatility -----------------------------------------------
MACRO: dict[str, str] = {
    "DX-Y.NYB": "US Dollar Index",
    "^VIX": "CBOE VIX",
    "^VVIX": "VIX of VIX",
}

# --- Crypto / risk sentiment ---------------------------------------------
CRYPTO: dict[str, str] = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
}

# --- Phase 1 core slice --------------------------------------------------
# Thin vertical slice: 5 stocks + key indices/commodities/VIX only.
CORE_TICKERS: list[str] = [
    *STOCKS,
    "^GSPC",
    "^IXIC",
    "GC=F",
    "CL=F",
    "HG=F",
    "^VIX",
    "DX-Y.NYB",
    "^TNX",
]

# --- Full universe (de-duplicated, order-preserving) ---------------------
_GROUPS = (STOCKS, INDICES, COMMODITIES, BONDS, MACRO, CRYPTO)


def _dedup() -> list[str]:
    seen: dict[str, None] = {}
    for group in _GROUPS:
        for ticker in group:
            seen.setdefault(ticker, None)
    return list(seen)


ALL_TICKERS: list[str] = _dedup()

# Name lookup across all groups.
NAMES: dict[str, str] = {k: v for group in _GROUPS for k, v in group.items()}

# Search terms for German-language news scraping (finanzen.net /
# finanznachrichten.de) per portfolio stock. Used by the Phase 2 sentiment layer.
GERMAN_QUERY: dict[str, str] = {
    "SIE.DE": "Siemens",
    "MUV2.DE": "Munich Re",
    "FCX": "Freeport-McMoRan",
    "TSLA": "Tesla",
    "8001.T": "Itochu",
}

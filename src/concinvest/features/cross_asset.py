"""Cross-asset derived features (Table 3).

Built from a mapping of ticker -> close-price Series (each indexed by date). The
output is a single DataFrame indexed by date, aligned on the union of dates.
"""

from __future__ import annotations

import pandas as pd

from .technical import sma


def _close(closes: dict[str, pd.Series], ticker: str) -> pd.Series | None:
    s = closes.get(ticker)
    return s if s is not None and not s.empty else None


def build_cross_asset_frame(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute cross-asset ratios from a dict of close-price Series."""
    frame = pd.DataFrame(closes)
    out = pd.DataFrame(index=frame.index)

    gold, oil = _close(closes, "GC=F"), _close(closes, "CL=F")
    if gold is not None and oil is not None:
        out["gold_oil_ratio"] = gold / oil

    copper, gold2 = _close(closes, "HG=F"), _close(closes, "GC=F")
    if copper is not None and gold2 is not None:
        out["copper_gold_ratio"] = copper / gold2

    vix = _close(closes, "^VIX")
    if vix is not None:
        out["vix_level"] = vix
        out["vix_sma20_ratio"] = vix / sma(vix, 20)

    tnx = _close(closes, "^TNX")
    if tnx is not None:
        out["yield_10y"] = tnx

    dxy = _close(closes, "DX-Y.NYB")
    if dxy is not None:
        out["dollar_index"] = dxy

    btc = _close(closes, "BTC-USD")
    if btc is not None:
        out["btc_sma20_ratio"] = btc / sma(btc, 20)

    out.index.name = "date"
    return out

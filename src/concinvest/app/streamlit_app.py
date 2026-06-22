"""Streamlit entrypoint (Phase 0 placeholder).

Run on port 8505 (project convention)::

    streamlit run src/concinvest/app/streamlit_app.py --server.port 8505

Real views (performance vs. NASDAQ, allocations, forecast table, 10x10 correlation
matrix, regime detection) arrive in later phases.
"""

from __future__ import annotations

import streamlit as st

from concinvest import __version__, config
from concinvest.app import exit_button
from concinvest.data import tickers


def main() -> None:
    st.set_page_config(page_title="ConcentratedInvestment", page_icon="📈", layout="wide")
    st.title("ConcentratedInvestment")
    st.caption(f"v{__version__} · Phase 0 scaffold")

    with st.sidebar:
        st.header("Controls")
        exit_button.render(st)

    st.subheader("Portfolio stocks")
    st.table({"Ticker": list(tickers.STOCKS), "Name": list(tickers.STOCKS.values())})

    st.subheader("Config")
    st.write(
        {
            "start_date": str(config.START_DATE),
            "initial_capital_eur": config.INITIAL_CAPITAL_EUR,
            "benchmark": config.BENCHMARK_TICKER,
            "core_tickers": len(tickers.CORE_TICKERS),
            "full_universe": len(tickers.ALL_TICKERS),
        }
    )

    st.info("Scaffold only — data, forecast, and backtest views land in later phases.")


if __name__ == "__main__":
    main()

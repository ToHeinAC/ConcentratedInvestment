"""Streamlit entrypoint.

Run on port 8505 (project convention)::

    streamlit run src/concinvest/app/streamlit_app.py --server.port 8505

Two tabs: **Current market** (cross-asset correlation + live analyst/sentiment) and
**Forecast & Backtest** (5-field forecast, portfolio-vs-NASDAQ curve, feature importance).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from concinvest import __version__, config
from concinvest.app import exit_button
from concinvest.data import tickers
from concinvest.ml.forecast import forecasts_to_frame


def _base_case_positions(total_value: float) -> pd.DataFrame:
    """Base-case target portfolio (Story.md) valued at ``total_value`` EUR.

    No live position tracking yet (Phase 4); this is the base-case allocation:
    each of the 5 names split 12% / 3% / 3% across stock / 2x / 3x, plus 10% cash.
    """
    rows = []
    for ticker in tickers.STOCKS:
        for tier, weight in config.BASE_PER_NAME_SPLIT.items():
            rows.append({
                "ticker": ticker,
                "name": tickers.NAMES.get(ticker, ticker),
                "type": tier,
                "weight": weight,
                "value_eur": weight * total_value,
            })
    rows.append({"ticker": "CASH", "name": "Cash", "type": "cash",
                 "weight": config.BASE_CASH_ALLOCATION,
                 "value_eur": config.BASE_CASH_ALLOCATION * total_value})
    return pd.DataFrame(rows)


def _positions_pie(positions: pd.DataFrame):
    """Matplotlib pie of relative sizing per name (tiers summed) + cash."""
    import matplotlib.pyplot as plt

    by_name = positions.groupby("ticker", sort=False)["weight"].sum()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(by_name.values, labels=list(by_name.index), autopct="%1.0f%%",
           startangle=90, counterclock=False)
    ax.set_aspect("equal")
    return fig


@st.cache_data(show_spinner="Fetching data, training model, running backtest…", ttl=3600)
def _load(n_dataset: int, with_sentiment: bool):
    # Import here so the module loads fast even if heavy deps lag.
    from concinvest.pipeline import run_phase1

    res = run_phase1(n_dataset=n_dataset, with_sentiment=with_sentiment)
    return {
        "forecasts": forecasts_to_frame(res.forecasts),
        "curve": res.backtest.curve,
        "portfolio_return": res.backtest.portfolio_return,
        "benchmark_return": res.backtest.benchmark_return,
        "beats": res.backtest.beats_benchmark,
        "importance": res.model.feature_importance,
        "mean_cv": res.model.mean_cv,
        "correlation": res.correlation,
        "sentiment": res.sentiment,
    }


def main() -> None:
    st.set_page_config(page_title="ConcentratedInvestment", page_icon="📈", layout="wide")
    st.title("ConcentratedInvestment")
    st.caption(f"v{__version__} · Phase 2 — full universe, richer sentiment & analyst signals")

    with st.sidebar:
        st.header("Controls")
        n_dataset = st.select_slider(
            "Training datapoints", options=[1000, 2000, 4000, 8000], value=4000
        )
        with_sentiment = st.checkbox("Live sentiment (slower)", value=False)
        run = st.button("Run / refresh", type="primary")
        st.divider()
        exit_button.render(st)

    if not run and "loaded" not in st.session_state:
        st.info("Set options and press **Run / refresh** to fetch data and generate a forecast.")
        return

    if run:
        st.session_state["loaded"] = True
        _load.clear()

    data = _load(n_dataset, with_sentiment)

    tab_current, tab_forecast = st.tabs(["Current market", "Forecast & Backtest"])
    with tab_current:
        _render_current(data)
    with tab_forecast:
        _render_forecast(data)

    st.caption(f"Benchmark: {config.BENCHMARK_TICKER} · start {config.START_DATE}")


def _render_current(data: dict) -> None:
    """First tab: current portfolio, market state — correlation, live signals."""
    total_value = float(data["curve"]["portfolio"].iloc[-1])
    positions = _base_case_positions(total_value)

    st.subheader("Current portfolio (base case)")
    st.metric("Portfolio value", f"€{total_value:,.0f}")
    col_tbl, col_pie = st.columns([3, 2])
    with col_tbl:
        st.dataframe(
            positions.style.format({"weight": "{:.0%}", "value_eur": "€{:,.0f}"}),
            width="stretch", hide_index=True,
        )
    with col_pie:
        st.pyplot(_positions_pie(positions))
    st.caption(
        "Base-case target allocation (Story.md); live position tracking with "
        "realized 2x/3x lots arrives in Phase 4."
    )

    st.subheader("Cross-asset correlation (recent 60 days)")
    st.dataframe(
        data["correlation"].style.format("{:.2f}").background_gradient(
            cmap="RdYlGn", vmin=-1, vmax=1
        ),
        width="stretch",
    )

    sent = data["sentiment"]
    if sent is not None and not sent.empty:
        st.subheader("Live analyst & sentiment signals")
        st.dataframe(sent, width="stretch", hide_index=True)
        st.caption("Stored in `sentiment_analyst`; not yet model features (no history).")


def _render_forecast(data: dict) -> None:
    """Second tab: forecast, backtest, and model feature importance."""
    st.subheader("Forecast")
    fc = data["forecasts"]
    if fc.empty:
        st.success("Base case: **hold** — no trade triggered by current conditions.")
    else:
        st.dataframe(fc, width="stretch", hide_index=True)

    st.subheader("Backtest vs. NASDAQ (validation window)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio return", f"{data['portfolio_return']:.1%}")
    c2.metric("NASDAQ return", f"{data['benchmark_return']:.1%}")
    c3.metric(
        "Outperformance",
        f"{data['portfolio_return'] - data['benchmark_return']:+.1%}",
        delta="beats benchmark" if data["beats"] else "below benchmark",
    )
    st.line_chart(data["curve"])
    st.caption(f"Model CV ROC-AUC (mean): {data['mean_cv']:.3f}")

    st.subheader("Feature importance")
    imp = pd.Series(data["importance"]).sort_values(ascending=True)
    st.bar_chart(imp)


if __name__ == "__main__":
    main()

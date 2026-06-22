"""Streamlit entrypoint (Phase 1).

Run on port 8505 (project convention)::

    streamlit run src/concinvest/app/streamlit_app.py --server.port 8505

Shows the live Phase 1 slice: forecast table, portfolio-vs-NASDAQ backtest curve,
feature importances, and a cross-asset correlation matrix.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from concinvest import __version__, config
from concinvest.app import exit_button
from concinvest.ml.forecast import forecasts_to_frame


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
    }


def main() -> None:
    st.set_page_config(page_title="ConcentratedInvestment", page_icon="📈", layout="wide")
    st.title("ConcentratedInvestment")
    st.caption(f"v{__version__} · Phase 1 — live end-to-end slice")

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

    # --- Forecast -------------------------------------------------------
    st.subheader("Forecast")
    fc = data["forecasts"]
    if fc.empty:
        st.success("Base case: **hold** — no trade triggered by current conditions.")
    else:
        st.dataframe(fc, use_container_width=True, hide_index=True)

    # --- Backtest -------------------------------------------------------
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

    # --- Feature importance --------------------------------------------
    st.subheader("Feature importance")
    imp = pd.Series(data["importance"]).sort_values(ascending=True)
    st.bar_chart(imp)

    # --- Correlation matrix --------------------------------------------
    st.subheader("Cross-asset correlation (recent 60 days)")
    st.dataframe(
        data["correlation"].style.format("{:.2f}").background_gradient(
            cmap="RdYlGn", vmin=-1, vmax=1
        ),
        use_container_width=True,
    )

    st.caption(f"Benchmark: {config.BENCHMARK_TICKER} · start {config.START_DATE}")


if __name__ == "__main__":
    main()

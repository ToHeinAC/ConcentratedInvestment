# ConcentratedInvestment

ML-based recommendation system for a **concentrated 5-stock portfolio**, fed by daily Yahoo
Finance data. It builds a feature database, trains a Random Forest forecaster on synthetic trade
data, and surfaces trade recommendations through a Streamlit UI.

**Goal:** beat NASDAQ (`^IXIC`) over `2020-01-01 → present`, after German capital-gains tax,
validated on the held-out final year.

See [`Story.md`](./Story.md) for the domain spec and [`IMPLEMENTATION.md`](./IMPLEMENTATION.md)
for the phased build plan.

> **Status: Phase 1 (thin end-to-end slice).** Live pipeline: fetch core tickers → SQLite →
> technical + cross-asset + sentiment features → synthetic dataset → RandomForest → 5-field
> forecast → backtest vs. NASDAQ → Streamlit UI. Beating the benchmark is the Phase 3 tuning goal.

---

## Portfolio (fixed v1, max 5 stocks)

| Name | Ticker |
|------|--------|
| Siemens AG | `SIE.DE` |
| Münchener Rück | `MUV2.DE` |
| Freeport-McMoRan | `FCX` |
| Tesla | `TSLA` |
| ITOCHU | `8001.T` |

Base case: **90% stocks / 10% cash**, mostly **no trades** unless the data triggers one. Long-only
2x/3x leverage allowed, with strict per-name (33%), daily-sell (<10%), and drawdown (20%) limits.
Forecasts emit five fields: `ticker, buy|sell, amount_eur, stock|2x|3x, confidence`.

---

## Tech stack

Python 3.11+ · [`uv`](https://docs.astral.sh/uv/) · `pandas` · `yfinance` · `scikit-learn` ·
`streamlit` · `pytest` · SQLite · Docker.

---

## Quick start

```bash
# Install (creates .venv and resolves deps)
uv sync

# Show config + ticker universe
uv run concinvest info

# Daily ETL: fetch core tickers + compute/store features
uv run concinvest update

# Full Phase 1 slice: fetch -> model -> forecast -> backtest (prints results)
uv run concinvest run --n 4000          # add --sentiment for live news scoring

# Run tests
uv run pytest

# Launch the Streamlit app (port 8505 by project convention)
uv run streamlit run src/concinvest/app/streamlit_app.py --server.port 8505
```

Without `uv` you can also run directly:

```bash
PYTHONPATH=src python -m concinvest.cli info
PYTHONPATH=src python -m pytest
```

### Stopping the app

The Streamlit sidebar has a **safe-exit button** that finds the running port and kills only that
process (never your SSH session). Manual equivalent:

```bash
lsof -ti:8505 | xargs -r kill -9
```

---

## Project layout

```
src/concinvest/
├── config.py          # paths, dates, portfolio/risk constants, port 8505
├── cli.py             # `concinvest {info,update,run}`
├── pipeline.py        # run_phase1 / fetch_and_store orchestration
├── data/              # tickers.py (universe) · fetch.py (yfinance) · store.py (SQLite)
├── features/          # technical · cross_asset · sentiment (VADER) · analyst · options
├── ml/                # dataset (panel + synthetic) · model (RandomForest+TSCV) · forecast
├── backtest/          # engine: model-timed portfolio vs. NASDAQ
├── portfolio/         # state · rules · tax (German)                     [Phase 4]
└── app/               # streamlit_app.py · exit_button.py
tests/                 # pytest (20 tests, offline synthetic fixtures)
docs/                  # architecture.md · SCHEMA.md
```

Ticker universe: **5 portfolio stocks**, 5 global indices, commodities, bonds/rates, currency &
volatility, and crypto — 27 tickers total, with a 13-ticker **core slice** used by the Phase 1
vertical slice.

### Documentation

- [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) — current state, phase status, run/verify (start here)
- [`docs/architecture.md`](./docs/architecture.md) — modules, data flow, design decisions
- [`docs/SCHEMA.md`](./docs/SCHEMA.md) — database tables + model feature contract
- [`Story.md`](./Story.md) — original product spec / PRD

---

## Build phases

| Phase | Scope |
|-------|-------|
| **0** | Scaffold — package, config, tickers, CLI, tests, Docker, exit button *(done)* |
| **1** | Thin end-to-end slice: fetch → SQLite → features (incl. baseline sentiment) → RandomForest → forecast → backtest → UI *(done)* |
| **2** | Deepen data & features: full universe, FinBERT + German-news scraping, options IV skew, analyst revision momentum |
| **3** | Full 100k-datapoint synthetic dataset, TimeSeriesSplit CV, feature importance, tuning |
| **4** | Full rules engine + German tax, integrated into the backtest |
| **5** | UI polish (correlation matrix, regime detection), daily cron, Docker deploy |

See [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) for details.

---

## Docker

```bash
docker build -t concinvest .
docker run -p 8505:8505 concinvest
```

---

## Disclaimer

For research and educational use only. Not financial advice.

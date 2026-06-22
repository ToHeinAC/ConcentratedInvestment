# IMPLEMENTATION.md

Implementation plan for **ConcentratedInvestment** — a Yahoo-Finance-fed, ML-based portfolio
recommendation system. This document is the build blueprint derived from [`Story.md`](./Story.md).
Read `Story.md` for the full domain rationale; this file is the executable plan.

---

## 1. Overview & Goals

Build a daily-updated database and an sklearn Random Forest forecaster that recommends trades for a
**fixed concentrated 5-stock portfolio**, then surface it through a simple, modern Streamlit UI.

**Fixed stock universe (v1):**

| Name | Ticker |
|------|--------|
| Siemens AG | `SIE.DE` |
| Münchener Rück | `MUV2.DE` |
| Freeport-McMoRan | `FCX` |
| Tesla | `TSLA` |
| ITOCHU | `8001.T` |

(Later: user-configurable, still max 5.)

**Single success metric:** backtested portfolio **total return after** German 25% Abgeltungsteuer
(with realized-loss offsetting) must **beat NASDAQ (`^IXIC`)** over `2020-01-01 → present`,
validated on the **held-out final year** (first 4 years train, last 1 year validate).

**Operating principles (from `Story.md`):** KISS, modern/user-friendly UI, daily data + daily
updates, base case = mostly **no trades** (a trade requires a strong data-driven trigger).

**Forecast output — exactly 5 fields:** `ticker`, `buy|sell`, `amount_eur`, `stock|2x|3x`,
`confidence`.

---

## 2. Tech Stack (fixed by spec)

- Python 3.11+
- `uv` for packaging/deps
- `pandas` for data
- `yfinance` for market data
- `scikit-learn` (RandomForest) for ML
- `streamlit` for the UI
- `pytest` for tests
- Docker for deployment
- SQLite for storage (prototype; Postgres deferred)

Sentiment libs: `nltk` (VADER), `transformers`+`torch` (FinBERT), `requests`+`beautifulsoup4`
(German news scraping). Optional: `fredapi` (2Y yield / 10Y-2Y spread).

---

## 3. Architecture

```
ConcentratedInvestment/
├── pyproject.toml              # uv-managed
├── Dockerfile
├── IMPLEMENTATION.md
├── Story.md
├── .streamlit/config.toml      # exists (theme)
├── data/                       # SQLite db file(s), gitignored
├── src/concinvest/
│   ├── __init__.py
│   ├── config.py               # paths, dates (START=2020-01-01), constants
│   ├── data/
│   │   ├── tickers.py          # universe constants (stocks, indices, commodities, bonds, crypto, vol)
│   │   ├── fetch.py            # yf.download(threads=True) + retries + rate-limit delay
│   │   └── store.py            # SQLite schema + upsert; raw OHLCV separate from features
│   ├── features/
│   │   ├── technical.py        # SMA/EMA/RSI/MACD/Bollinger/ATR/OBV/ROC (pandas rolling/ewm)
│   │   ├── cross_asset.py      # gold/oil, copper/gold, vix ratios, yield spreads, rel. strength
│   │   ├── analyst.py          # recommendationMean, eps_revisions, analyst_price_targets
│   │   ├── sentiment.py        # news → VADER (baseline) / FinBERT; German-source scraping; -3..+3
│   │   └── options.py          # put/call ratio, IV skew from option_chain()
│   ├── portfolio/
│   │   ├── state.py            # positions (stock + 2x/3x leverage) + cash
│   │   ├── rules.py            # allocation/risk constraint engine (see §5)
│   │   └── tax.py              # German 25% flat tax + realized-loss offsetting
│   ├── ml/
│   │   ├── dataset.py          # synthetic 100k datapoint generator (50k buy / 50k sell)
│   │   ├── model.py            # RandomForest + TimeSeriesSplit CV + feature importance
│   │   └── forecast.py         # emit the 5-field forecast
│   ├── backtest/
│   │   └── engine.py           # replay 2020→now under rules; compare vs NASDAQ
│   ├── app/
│   │   ├── streamlit_app.py    # UI (perf, allocations, forecast table, 10x10 corr, regime)
│   │   └── exit_button.py      # safe-exit helper (find port 8505, lsof|kill, spare SSH)
│   ├── pipeline.py             # fetch → features → store (daily ETL)
│   └── cli.py                  # entrypoints: update, train, forecast, backtest
└── tests/                      # pytest, mirrors src layout
```

---

## 4. Database Schema (SQLite)

Raw OHLCV is stored **separately** from computed features so parameters can be re-tuned and
features recomputed without re-downloading (per `Story.md` Implementation Notes).

- **`ohlcv_raw`** — `date, ticker, open, high, low, close, adj_close, volume, dividends, splits`
  (PK `(date, ticker)`).
- **Table 1 `daily_market`** — `date, ticker`, OHLCV, `sma_{5,10,20,50,100,200}`,
  `ema_{12,26,50}`, `rsi_14, macd, macd_signal, bollinger_upper, bollinger_lower`,
  `price_sma50_ratio, price_sma200_ratio, sma50_sma200_ratio, volume_sma20_ratio`.
- **Table 2 `sentiment_analyst`** — `date, ticker, recommendation_mean, strong_buy, buy, hold,
  sell, strong_sell, eps_revision_up_7d, eps_revision_down_7d, analyst_target_{mean,high,low},
  news_sentiment_score, put_call_ratio`.
- **Table 3 `cross_asset`** — `date, gold_oil_ratio, copper_gold_ratio, vix_level,
  vix_sma20_ratio, yield_10y, yield_spread_10y_5y, dollar_index, btc_sma20_ratio`.

---

## 5. Portfolio Rules Engine (`portfolio/rules.py`)

Encodes every constraint from `Story.md` as pure, individually testable functions:

- **Base case:** 90% stocks / 10% cash; default action = **hold** (no trade unless triggered).
- **Leverage:** long-only 2x and 3x positions allowed.
- **Per-name cap:** if any single stock (underlying + its 2x/3x combined) exceeds **33%** of
  portfolio value → **trim 3%** of portfolio value from that name.
- **Daily sell cap:** each individual sell < **10%** of portfolio/day.
- **Drawdown guard:** max **20%** portfolio drawdown; on breach, shift allocations toward cash per
  the forecast.
- **Crisis mode:** may temporarily go 100% invested / 0% cash after a major pullback/black-swan,
  but must revert to base case within **2 months**.
- **Dividends:** credited on underlying stock positions only (not leveraged).
- **Tax (`tax.py`):** realized gains taxed at flat **25%** Abgeltungsteuer; realized losses offset
  gains so only the **net** is taxed.

Leverage modeling decision (v1): treat 2x/3x as **daily-rebalanced constant-leverage** return
multipliers on the underlying (documented assumption; revisit if modeling real LETF instruments).

---

## 6. ML Dataset & Model

- **`dataset.py`** — generate **100,000 stochastic datapoints** (50k buys / 50k sells). Each point
  is a **market snapshot** (full feature vector at a random date):
  - *Buys:* random valid date/stock incl. 2x/3x, **fixed 10%** portfolio position size, only when
    cash is available.
  - *Sells:* random valid liquidation of **currently-held** allocations that satisfy the rules;
    each sell's **label = forward performance** measured against the matching prior buy(s).
  - Guard against **look-ahead/label leakage**: features use only point-in-time data; labels use
    strictly forward windows.
- **`model.py`** — sklearn `RandomForest`, **TimeSeriesSplit** cross-validation, **feature
  importance** analysis. Train on first 4 years, validate on last year.
- **`forecast.py`** — produce the 5-field forecast (`ticker, buy|sell, amount_eur, stock|2x|3x,
  confidence`); `confidence` from model probability/score.

**Backtest start state:** 100k EUR at 2020-01-01, 80% stocks / 20% cash; per-stock dollar
allocation 12% stock + 3% (2x) + 3% (3x). Engine replays day-by-day applying forecasts under the
rules engine and reports return vs. `^IXIC`.

---

## 7. Streamlit UI (`app/streamlit_app.py`)

KISS + modern. Runs on port **8505**. Displays:

- Portfolio performance over time vs. NASDAQ.
- Current allocations (incl. leverage breakdown) and cash.
- **Forecast table** (the 5 fields) with confidence.
- **10×10 correlation matrix** highlighting the most significant correlations / anticorrelations of
  performance for the recent market condition; flag detected **rising-market / regime** conditions.
- **Safe-exit button** (`app/exit_button.py`): dynamically finds the running port and runs
  `lsof -ti:PORT | xargs -r kill -9` **without** killing the SSH connection.

---

## 8. Phased Delivery Plan

Strategy: **thin end-to-end vertical slice first** (every layer present, including a sentiment
signal), then deepen each layer.

### Phase 0 — Scaffold
`uv` project + `pyproject.toml`, package skeleton, `pytest` wired, `tickers.py` constants,
`config.py`, Dockerfile stub, `exit_button.py` helper. `.gitignore` already excludes `data/` &
secrets.

### Phase 1 — Thin vertical slice (end-to-end, sentiment-aware)
Fetch 5 stocks + core indices/commodities/VIX → SQLite. Compute a **minimal but representative**
feature set spanning **all groups**: technical (a few MAs + RSI/MACD), cross-asset (key ratios),
analyst numeric (recommendation mean), **baseline VADER news-sentiment score**, and **put/call
ratio**. Generate a small synthetic dataset, train a baseline RandomForest, emit a forecast, run a
simple backtest vs. NASDAQ, render a minimal Streamlit page. **Goal: a working, validatable
forecast with every architectural layer present.**

### Phase 2 — Deepen data & features
Full ticker universe; complete technical + cross-asset feature sets; **FinBERT** sentiment +
**finanzen.net / finanznachrichten.de** scraping; options **IV skew**; analyst **revision
momentum** (`eps_revisions`, `eps_trend`).

### Phase 3 — Full ML dataset & model
Complete **100k-datapoint** generator, TimeSeriesSplit CV, feature-importance analysis, model
tuning aimed at beating the benchmark.

### Phase 4 — Rules engine & tax (full)
Full allocation/risk/drawdown/trim/leverage logic + German tax with loss offsetting, integrated
into the backtest engine.

### Phase 5 — UI polish & ops
Correlation matrix, regime detection, richer performance views; **daily cron** (after US close,
~22:00 CET / Hamburg) running `pipeline.py`; Docker deployment.

---

## 9. Key Reuse / Library Notes

- `yfinance`: `yf.download(threads=True)` for batch OHLCV; `Ticker` attrs (`recommendations`,
  `eps_revisions`, `analyst_price_targets`, `get_news`, `option_chain`) per `Story.md` §"yfinance
  Data Types".
- `pandas`: `.rolling()` for SMA, `.ewm()` for EMA — computed in ETL, not at train time.
- Sentiment: `nltk` VADER (baseline, fast) → `transformers` FinBERT (accurate) on headlines;
  `requests` + `beautifulsoup4` for the two German news sources; aggregate to daily **-3..+3**.
- sklearn: `RandomForestClassifier`/`Regressor` + `TimeSeriesSplit`.
- `fredapi` (optional): series `T10Y2Y` for the 2-year yield / 10Y-2Y spread not on Yahoo.

---

## 10. Risks & Open Questions

- **yfinance throttling** — use retries + ~0.5s delay between non-OHLCV requests; cache raw OHLCV.
- **Sparse/low-relevance news** — Yahoo returns only ~8–10 articles; mitigate with the German
  sources.
- **Label leakage** in synthetic sell evaluation — enforce strict point-in-time features vs.
  forward-only labels.
- **Leverage modeling** — 2x/3x as daily-rebalanced multipliers vs. real LETF instruments
  (documented v1 assumption).
- **Look-ahead in features** — backtest must be strictly point-in-time.

---

## 11. Verification

- **Unit tests (`pytest`)** per module: rules-engine constraints (33% trim, <10%/day sell, 20%
  drawdown, 2-month crisis revert), tax calc (net loss offsetting), feature formulas (SMA/RSI/MACD
  vs. known values), dataset label correctness.
- **Phase-1 integration test**: run fetch → features → model → forecast on a tiny date range and a
  couple of tickers; assert a valid 5-field forecast is produced.
- **Backtest check**: report renders and asserts the NASDAQ comparison is computed over the
  validation year.
- **Manual UI check**: `streamlit run` on port 8505; confirm UI loads and the safe-exit button
  kills only the app port (not SSH).
```

> Status: planning document. No application code exists yet — implementation starts at Phase 0.

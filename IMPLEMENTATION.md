# IMPLEMENTATION.md

Compact, current-state reference for **ConcentratedInvestment**. Domain spec:
[`Story.md`](./Story.md). Deep docs: [`docs/architecture.md`](docs/architecture.md)
(modules + data flow) and [`docs/SCHEMA.md`](docs/SCHEMA.md) (database).

---

## 1. Goal

ML recommendation system for a **fixed concentrated 5-stock portfolio** fed by daily
Yahoo Finance data, surfaced through a Streamlit UI.

**Stocks (v1, max 5):** `SIE.DE`, `MUV2.DE`, `FCX`, `TSLA`, `8001.T` (later
user-configurable).

**Success metric:** backtested total return after German 25% Abgeltungsteuer (with
loss offsetting) must beat **NASDAQ (`^IXIC`)** over `2020-01-01 → present`, validated
on the held-out final year (first 4 yrs train / last 1 yr validate).

**Forecast — 5 fields:** `ticker`, `buy|sell`, `amount_eur`, `stock|2x|3x`,
`confidence`.

## 2. Tech stack

Python 3.11+ · `uv` · `pandas` · `yfinance` · `scikit-learn` (RandomForest) ·
`streamlit` (port 8505) · `pytest` · SQLite · Docker. Sentiment: `nltk` VADER
(default) or `transformers` FinBERT (opt-in via the `sentiment` extra), with
`requests`/`beautifulsoup4` German-news scraping. Backend set by `config.SENTIMENT_MODEL`.

## 3. Status

| Phase | Scope | State |
|-------|-------|-------|
| **0** | Scaffold: package, config, tickers, CLI, tests, Docker, exit button | ✅ done |
| **1** | Thin end-to-end slice (all layers, sentiment-aware) | ✅ done |
| **2** | Deepen data & features: full universe, FinBERT + German-news scraping, options IV skew, analyst revision momentum | ✅ done |
| **3** | Full 100k synthetic dataset, TimeSeriesSplit tuning, feature-importance selection — **tune to beat NASDAQ** | 🔄 in progress |
| **4** | Full rules engine (allocation/risk/leverage/drawdown/trim) + German tax, in backtest | 🔄 in progress |
| **5** | UI polish (regime detection), daily cron (~22:00 CET), Docker deploy | ⏳ planned |

## 4. Architecture (implemented)

Package `src/concinvest/`. Full detail in [`docs/architecture.md`](docs/architecture.md).

```
data/      tickers.py · fetch.py (yfinance, all network) · store.py (SQLite)
features/  technical.py · cross_asset.py · sentiment.py (VADER/FinBERT) · analyst.py · options.py
ml/        dataset.py (panel + synthetic gen) · model.py (RF+TSCV) · forecast.py (5 fields)
portfolio/ state.py (leveraged lots+cash) · tax.py (Abgeltungsteuer) · rules.py (guardrails)
backtest/  engine.py (forecast-driven + rules-based leveraged portfolio vs NASDAQ)
app/       streamlit_app.py · exit_button.py
config.py · pipeline.py (run_phase1 / fetch_and_store) · cli.py
```

**Data flow:** `fetch → features → store (SQLite) → ml.dataset panel → ml.model →
{forecast, backtest} → app`. Orchestrated by `pipeline.run_phase1`;
`pipeline.fetch_and_store` is the reusable daily-ETL block.

**Database:** 4 tables — `ohlcv_raw`, `daily_market` (Table 1), `sentiment_analyst`
(Table 2), `cross_asset` (Table 3). Raw OHLCV kept separate from derived features.
See [`docs/SCHEMA.md`](docs/SCHEMA.md).

**Model contract:** `ml.dataset.FEATURE_COLS` = technical + cross-asset + sentiment
placeholders + action encoding (`is_sell`, `leverage`). Sentiment is neutral over
history (no historical news feed) and filled live at forecast time.

## 5. Phase 1 design notes

- **Synthetic dataset** — half buys / half sells; each a point-in-time market
  snapshot + action; label = profitable over a forward horizon (buy good if price
  rose, sell good if it fell). Strict point-in-time features vs. forward labels avoid
  leakage. Phase 1 uses a modest `n`; the full 100k generator is Phase 3.
- **Backtest** — equal-weight 5-stock basket with daily equity exposure scaled by the
  model's mean buy-confidence (rest cash), lagged one day. Full allocation/risk/tax is
  Phase 4, so beating NASDAQ is **not** yet expected here.
- **Leverage** — 2x/3x as daily-rebalanced constant-leverage multipliers (documented
  assumption).

## 5b. Phase 2 design notes

- **Full universe** — `run_phase1` / `concinvest update` now fetch `ALL_TICKERS` (27).
  New cross-asset series (`^FVX`, `^VVIX`, `^SPGSCI`) feed `yield_spread_10y_5y`,
  `vvix_level`, `gsci_sma20_ratio`; the first two also join `FEATURE_COLS`.
- **Sentiment backends** — `score_headlines(model=…)` selects VADER (default) or
  FinBERT (`P(pos) − P(neg)` scaled to ±3), both lazy-loaded. German headlines from
  `finanznachrichten.de` (per-stock `tickers.GERMAN_QUERY`) are appended to the
  yfinance feed before scoring.
- **New live signals** — `sentiment_analyst` gains `eps_revision_up_7d/down_7d`,
  `analyst_target_mean`, `iv_skew` (OTM-put − ATM-call IV). These have no usable
  history, so they are **stored/displayed only**, not model features.
- **Additive migrations** — `store._migrate` `ALTER TABLE`s the new columns onto
  pre-Phase-2 databases idempotently; no rebuild required.

## 5c. Phase 3 design notes (in progress)

- **Time-honest dataset** — `generate_dataset` returns rows **sorted by snapshot
  date** (DatetimeIndex), so `TimeSeriesSplit` CV is valid. `n` is now arbitrary;
  the Story.md 100k run is `concinvest run --n 100000`.
- **Honest validation** — `train_validate_split` carves the last
  `VALIDATION_YEARS` off by calendar date. `run_phase1` trains **only on the
  pre-validation split**, so the validation-window backtest is true out-of-sample
  (previously the model saw the backtest window — leakage, now fixed).
- **Tuning** — `model.tune` / `tune_and_train` pick the best `PARAM_GRID` entry by
  mean TSCV ROC-AUC; `concinvest run` tunes by default (`--no-tune` to skip), and
  prints the chosen params. Selected params live on `TrainedModel.params`.
- **Open** — feature-importance-driven pruning of `FEATURE_COLS`, and the headline
  "beat NASDAQ" target, which is **gated on the Phase 4 allocation/leverage/tax
  engine** (the current confidence-scaled basket is not expected to clear NASDAQ).

## 5d. Phase 4 design notes (in progress)

- **`portfolio/` package** — `state.PortfolioState` holds leveraged lots (tier 1/2/3,
  each with a cost basis) + cash; `mark()` applies daily constant-leverage returns;
  `sell_name()` realizes gains proportionally and pays tax. `build_base_case()` is the
  90/10 book (per-name 12%/3%/3%).
- **`tax.tax_on_sale`** — 25% flat Abgeltungsteuer; realized losses accumulate in a
  carry pool that offsets future gains before tax.
- **`rules`** — deterministic sell-side guardrails: 33% per-name → trim 3%; 20%
  drawdown → de-risk toward cash; every sell capped at 10%/day. `apply_guardrails`
  runs them per day (de-risk, then trim).
- **`backtest.run_rules_backtest`** — replays the base-case leveraged book under the
  guardrails vs NASDAQ (sell-side only).
- **`backtest.run_forecast_backtest`** — the book's target equity exposure tracks the
  model's mean buy-confidence (lagged, scaled by the 90% base allocation) with a
  `REBALANCE_BAND` dead-band, cash re-entry, daily guardrails, and tax. **This is now
  the pipeline/UI backtest.** "Optimal weighting" stays delegated to ML confidence.
- **Open** — crisis 100%/2-month-revert path and dividends on the underlying.

## 6. Run & verify

```bash
uv sync --extra dev
uv run pytest                                   # 40 tests, offline (synthetic fixtures)
uv run concinvest run --n 4000                  # live: fetch→model→forecast→backtest
uv run streamlit run src/concinvest/app/streamlit_app.py --server.port 8505
```

- **Unit tests** (`tests/`, offline via `conftest.py` synthetic market): technical
  indicators vs known values, cross-asset ratios (incl. VVIX/GSCI/10y-5y spread),
  sentiment scaling, SQLite upsert/read roundtrip, additive schema migration, pure
  fetch helpers (IV nearest-strike, finanznachrichten headline parse), dataset
  shape/balance/no-leakage + chronological order + date split, TSCV tuning, model
  train + 5-field forecast, backtest curve, portfolio state/tax/guardrails +
  rules-based backtest.
- **Live integration**: `concinvest run` prints model CV ROC-AUC, portfolio vs NASDAQ
  return, and the 5-field forecast for all stocks.
- **UI**: app boots on 8505; **Run / refresh** fetches live data; safe-exit button
  kills only the app port, never SSH.

## 7. Remaining phases — detail

- **Phase 3** (🔄) — done: time-ordered generator (100k-capable), honest
  date-based train/validate split, TSCV hyperparameter tuning. Remaining:
  feature-importance-driven pruning of `FEATURE_COLS`; reach validation return >
  NASDAQ (depends on Phase 4 allocation/leverage/tax).
- **Phase 4** (🔄) — done: `portfolio/` `state.py` (leveraged lots + cash),
  `tax.py` (25% flat + loss offset), `rules.py` (90/10 base, 33%→trim 3%, <10%/day
  sell, 20% drawdown→cash); `backtest.run_forecast_backtest` (confidence-driven
  exposure + re-entry + guardrails + tax), now wired into the pipeline. Remaining:
  crisis 100% / 2-month revert, dividends on the underlying.
- **Phase 5** — correlation/regime UI, `pipeline.fetch_and_store` daily cron, Docker.

## 8. Conventions

Apache-2.0 / MIT-compatible licensing. Tests-first for behaviour changes; functions
≤ ~40 lines; small commits. Streamlit on port 8505. Secrets never committed; runtime
`data/` gitignored.

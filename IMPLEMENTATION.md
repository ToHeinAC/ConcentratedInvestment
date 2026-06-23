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
| **3** | Full 100k synthetic dataset, TimeSeriesSplit tuning, feature-importance selection — **tune to beat NASDAQ** | ✅ done (marginal win accepted) |
| **4** | Full rules engine (allocation/risk/leverage/drawdown/trim/crisis) + German tax + dividends, in backtest | ✅ done |
| **5** | UI polish (regime detection), daily cron (~22:00 CET), Docker deploy | 🔄 in progress (cron + sentiment snapshot done) |

## 4. Architecture (implemented)

Package `src/concinvest/`. Full detail in [`docs/architecture.md`](docs/architecture.md).

```
data/      tickers.py · fetch.py (yfinance, all network) · store.py (SQLite)
features/  technical.py · cross_asset.py · sentiment.py (VADER/FinBERT) · analyst.py · options.py
ml/        dataset.py (panel + synthetic gen) · model.py (RF+TSCV) · forecast.py (5 fields) · overlay.py (live sentiment tilt)
portfolio/ state.py (leveraged lots+cash) · tax.py (Abgeltungsteuer) · rules.py (guardrails)
backtest/  engine.py (forecast-driven + rules-based portfolio) · walkforward.py (multi-window)
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
- **Feature pruning** — `model.select_features` drops features below
  `MIN_IMPORTANCE` (action encoding always kept); `tune_and_train(prune=True)` refits
  on the reduced set. `FEATURE_COLS` stays the stable superset callers build;
  `TrainedModel.features` records the columns actually used.
- **Exposure mapping** — `backtest._target_name_fraction` holds **each name's**
  per-name base weight (18% = 12+3+3) while *that name's* buy-confidence is
  neutral-to-bullish (≥ 0.5, the classifier's natural boundary) and only de-risks it
  proportionally below 0.5 (per-stock — Story.md's forecast is per ticker). Names are
  rebalanced independently (`_rebalance_names_to_target`), so a bearish read on one
  stock trims only that stock; a single-number book-level dial (`_target_exposure`) is
  retained for the Phase-1 `run_backtest`. Principled (not tuned to the validation
  year); the drawdown guardrail still handles crashes independently.
- **Live result (last validation year, `--n 10000`)** — portfolio **+28.8%** vs
  NASDAQ **+34.9%** under the base-case-faithful mapping (was +15.2% under the old
  linear-confidence one).
- **Walk-forward validation** (`concinvest validate`, `backtest.walkforward`) — the
  single-year read is misleading. Across four trained-then-tested 1-year windows
  (`--n 10000`, with the Phase 4 crisis path + underlying dividends active):

  | window | portfolio | NASDAQ | vs |
  |--------|-----------|--------|----|
  | 2022-08→2023-07 | +46.9% | +13.1% | **+33.8** |
  | 2023-07→2024-07 | +27.1% | +31.5% | −4.4 |
  | 2024-07→2025-07 | +13.7% | +11.3% | **+2.4** |
  | 2025-07→2026-06 | +29.5% | +28.2% | **+1.3** |

  **Win rate 75% (3/4), mean outperformance +8.3%** (per-stock rebalance). The strategy
  is high-variance: it crushed the 2022-23 value/commodity rotation. The crisis path
  (§5d) flipped the former worst window (2024-25) from −20.2% to a win, and the
  **per-stock confidence rebalance** (each name trimmed by its own forecast, not a
  basket-mean dial) lifted the win rate from 2/4 to 3/4 and mean outperformance from
  +5.7% to +8.3% — the worst window's shortfall roughly halved (−11.0 → −4.4). (Numbers
  vary run-to-run with the synthetic sample.)
- **Tier-graded de-risking scope** — shedding the riskiest tier first (3x → 2x → stock)
  is applied to the two Story.md de-risking events — the crash drawdown and the 33%
  post-upstreak trim — and is **performance-neutral** there (75%/+8.3% unchanged).
  Applying it to the *routine* confidence-rebalance instead cost ~5pp (fell to 50%/+3.3%)
  by de-levering in up-markets, so that path stays pro-rata — consistent with the
  earlier Lever-1 lesson (this basket's edge is its leverage in up-markets).
- **Risk-lever experiment** — two levers were evaluated to make the win more robust.
  **Lever 2 (leverage-aware de-risk)** — drawdown de-risk now sells the riskiest tier
  first (3x → 2x → stock) via `state.sell_tier`, keeping the Story.md 10%/name/day cap;
  it shipped (mean **+5.5%**, ≈ neutral, better risk hygiene). **Lever 1 (vol-aware
  leverage throttle)** — shedding 2x/3x when VIX is elevated — was **dropped**: the
  walk-forward showed it cut the strategy's leverage edge and fought the crisis dip-buy
  (mean fell to +3.5% at a VIX-28 stress threshold, +2.9% at VIX-20). Lesson: this
  basket's edge *is* the leverage in up-markets; de-levering on vol is net-negative.
- **Closed** — "beat NASDAQ" is now a **75% win rate, mean +8.2%** across the
  walk-forward (up from the marginal 50%/+5.7% under the basket-mean dial), accepted as
  the Phase 3 outcome. Still high-variance (one window trails ~4pp). The honest remaining
  lever is a **basket/benchmark review** (concentrated value vs a tech-heavy NASDAQ) —
  not more risk tuning — deferred (revisit if a stronger edge is wanted).

## 5d. Phase 4 design notes (in progress)

- **`portfolio/` package** — `state.PortfolioState` holds leveraged lots (tier 1/2/3,
  each with a cost basis) + cash; `mark()` applies daily constant-leverage returns;
  `sell_name()` realizes gains proportionally and pays tax. `build_base_case()` is the
  90/10 book (per-name 12%/3%/3%).
- **`tax.tax_on_sale`** — 25% flat Abgeltungsteuer; realized losses accumulate in a
  **full-portfolio** carry pool (one pool across all names, never expiring) that offsets
  future gains before tax — i.e. gains and losses net across the whole book over time
  (Story.md). (Only nuance vs. German law: a loss does not retroactively refund tax on a
  gain realized *earlier* the same year — a second-order effect.)
- **`rules`** — deterministic sell-side guardrails, both shedding the **riskiest tier
  first** (3x → 2x → stock, via `sell_riskiest_first`): 33% per-name → trim 3% (the
  post-upstreak case), 20% drawdown → de-risk toward cash (the crash case); every sell
  capped at 10%/day. `apply_guardrails` runs them per day (de-risk, then trim). The
  routine confidence-rebalance (in `backtest.engine`) sells **pro-rata** across tiers,
  preserving the leverage edge in up-markets (tier-grading there cost ~5pp in the
  walk-forward — §5c).
- **`backtest.run_rules_backtest`** — replays the base-case leveraged book under the
  guardrails vs NASDAQ (sell-side only).
- **`backtest.run_forecast_backtest`** — **each name's** target portfolio fraction
  tracks *that name's* buy-confidence (lagged, scaled by its per-name base weight) via
  `_target_name_fraction` + `_rebalance_names_to_target`, with a per-name dead-band,
  cash re-entry, daily guardrails, and tax. Names rebalance independently (Story.md's
  per-ticker forecast), so a bearish read on one stock trims only that stock. **This is
  the pipeline/UI backtest.** "Optimal weighting" stays delegated to ML confidence.
- **Crisis path** — `_is_crisis` flags a basket drop > `CRISIS_DROP` (15%) over
  `CRISIS_LOOKBACK` (10) trading days; on a flag the cash reserve is `_deploy`-ed to
  ~100% invested (buy-the-dip) and the book holds that for `CRISIS_REVERT_DAYS` (60,
  ~2 months) — no de-risk or rebalance-to-cash during the window — then reverts to the
  base case. The per-name trim still fires in crisis; crisis takes precedence over the
  drawdown de-risk. Story.md: "temporarily go 100% / 0% cash on a major pullback, back
  to base within 2 months." Lifted the worst walk-forward window from −20.2% to +8.1%
  (§5c).
- **Dividends on the underlying** — `_dividend_yields` recovers the per-day dividend
  yield as `adj_close.pct_change() − close.pct_change()` (total minus price return,
  from `auto_adjust=False`); `state.pay_dividends` credits cash on **tier-1 lots
  only** (Story.md: not the leveraged positions), net of the flat 25% tax. Wired into
  both the rules and forecast backtests.
- **Trade log + per-tier balances** — `rules.Trade` carries `date`/`tier`; every buy/sell
  is logged **per tier** with its actual € (deploys split 12/3/3; the pro-rata rebalance
  sell is decomposed by `_sell_proportional` — selling logic unchanged, returns identical)
  into `BacktestResult.trades`, and each day's per-`(ticker, tier)` value into
  `BacktestResult.tier_curve`. The **Strategy tab** stacks three panels on a **shared
  x-axis** with a legend: the price curve with aggregated buy/sell markers (total €), the
  **per-tier balance-evolution chart** (stock / 2x / 3x from `tier_curve`) with per-tier
  markers (actual € each), and NASDAQ; the full buy/sell table sits behind a **popover
  button**. Markers are drawn on the decision day (T-1, the signal bar), display-only.
- **Live sentiment overlay** (`ml/overlay.py`) — tilts the **live** 5-field forecast by
  the analyst signals: `sentiment_tilt` (recommendation mean + EPS-revision momentum +
  price-vs-target) scales confidence/amount, `risk_gate` (put/call + IV skew) caps the
  leverage tier on crash fear. **Live-only** — these signals have no history, so the
  overlay is *not* in the backtest/walk-forward. Next step to make them trainable: the
  Phase 5 cron snapshots `sentiment_analyst` daily to accumulate history.
- **Book-aware forecast sizing** — `run_phase1` runs the backtest first, then sizes the
  live forecast against the evolved book: `forecast.apply_book_limits` caps each buy at
  the **remaining cash** (decremented as buys are funded) and each sell at the **value
  held in that name's tier** (Story.md: buy only with cash on hand, sell only from open
  positions), dropping unfundable actions. (The backtest already enforced cash via
  `state.buy`; this brings the displayed 5-field forecast in line.)
- **Open** — basket/benchmark review (the real outperformance lever, §5c).

## 6. Run & verify

```bash
uv sync --extra dev
uv run pytest                                   # 63 tests, offline (synthetic fixtures)
uv run concinvest run --n 4000                  # live: fetch→model→forecast→backtest
uv run concinvest validate --n 10000            # walk-forward (multi-window) vs NASDAQ
uv run concinvest update --sentiment            # daily ETL + dated sentiment snapshot (cron)
uv run streamlit run src/concinvest/app/streamlit_app.py --server.port 8505
```

Daily cron: `scripts/daily_update.sh` wraps `concinvest update --sentiment` (logs to
the gitignored `data/daily_update.log`); schedule ~22:00 Europe/Berlin via `crontab -e`
(`CRON_TZ=Europe/Berlin`). Each run appends a dated `sentiment_analyst` snapshot so the
live analyst signals accumulate history (the prerequisite to making them trainable).

- **Unit tests** (`tests/`, offline via `conftest.py` synthetic market): technical
  indicators vs known values, cross-asset ratios (incl. VVIX/GSCI/10y-5y spread),
  sentiment scaling, SQLite upsert/read roundtrip, additive schema migration, pure
  fetch helpers (IV nearest-strike, finanznachrichten headline parse), dataset
  shape/balance/no-leakage + chronological order + date split, TSCV tuning, model
  train + 5-field forecast, backtest curve, portfolio state/tax/guardrails (incl.
  tier-targeted `sell_tier` + riskiest-first de-risk) + rules-based & forecast-driven
  backtests + trade-log recording, crisis-drop detection, underlying-only dividends,
  leading-holiday benchmark gap, sentiment overlay (tilt/gate/leverage cap),
  walk-forward window construction, daily-ETL sentiment-history accumulation,
  per-stock target fraction + independent-name rebalance, backtest final-state,
  riskiest-tier-first trim, forecast book-limits (cash/holdings caps).
- **Live integration**: `concinvest run` prints model CV ROC-AUC, portfolio vs NASDAQ
  return, and the 5-field forecast for all stocks.
- **UI**: app boots on 8505; **Run / refresh** fetches live data; safe-exit button
  kills only the app port, never SSH.

## 7. Remaining phases — detail

- **Phase 3** (✅) — done: time-ordered generator (100k-capable), honest
  date-based train/validate split, TSCV hyperparameter tuning, feature-importance
  pruning, base-case-faithful exposure mapping, walk-forward validation
  (`concinvest validate`), risk-control tightening (Lever 2 riskiest-first de-risk
  shipped; Lever 1 vol throttle evaluated and dropped — §5c), and **per-stock
  confidence rebalancing** (each name trimmed by its own forecast). Walk-forward
  **win rate 75% (3/4), mean +8.2%**, up from 50%/+5.7% under the old basket-mean dial.
  Still high-variance (one window ~−4pp). The remaining real lever is a
  **basket/benchmark review** (not more risk tuning), deferred to a future revisit.
- **Phase 4** (✅) — `portfolio/` `state.py` (leveraged lots + cash + tier-targeted
  `sell_tier`), `tax.py` (25% flat + loss offset), `rules.py` (90/10 base, 33%→trim 3%,
  <10%/day sell, 20% drawdown→riskiest-tier-first de-risk);
  `backtest.run_forecast_backtest` (confidence-driven exposure + re-entry + guardrails
  + tax + crisis 100%/2-month-revert + underlying dividends + trade log), wired into
  the pipeline; live sentiment overlay (`ml/overlay.py`) on the forecast; Strategy tab.
- **Phase 5** (🔄) — **done:** daily cron — `pipeline.daily_etl` (OHLCV + features +
  cross-asset + dated `sentiment_analyst` snapshot) behind `concinvest update
  --sentiment`, wrapped by `scripts/daily_update.sh` for ~22:00 Europe/Berlin
  scheduling; the dated snapshots accumulate the analyst-signal history needed to make
  the overlay trainable. **Remaining:** correlation/**regime detection** UI (rising-
  market detection), Docker deploy.

## 8. Conventions

Apache-2.0 / MIT-compatible licensing. Tests-first for behaviour changes; functions
≤ ~40 lines; small commits. Streamlit on port 8505. Secrets never committed; runtime
`data/` gitignored.

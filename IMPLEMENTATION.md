# IMPLEMENTATION.md

Compact, current-state reference for **ConcentratedInvestment**. Domain spec:
[`Story.md`](./Story.md). Deep docs: [`docs/architecture.md`](docs/architecture.md)
(modules + data flow) and [`docs/SCHEMA.md`](docs/SCHEMA.md) (database).

---

## 1. Goal

ML recommendation system for a **fixed concentrated 5-stock portfolio** fed by daily
Yahoo Finance data, surfaced through a Streamlit UI.

**Stocks (v1, max 5):** `SIE.DE`, `MUV2.DE`, `FCX`, `TSLA`, `005930.KS` (later
user-configurable; `005930.KS` Samsung Electronics replaced `8001.T` ITOCHU, which is
not 3x-tradable at the broker).

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
| **3** | Full 100k synthetic dataset, TimeSeriesSplit tuning, feature-importance selection — **tune to beat NASDAQ** | ✅ done (beats NASDAQ on walk-forward — see §5c) |
| **4** | Full rules engine (allocation/risk/leverage/drawdown/trim/crisis) + German tax + dividends, in backtest | ✅ done |
| **5** | UI polish (regime detection), daily cron (~22:00 CET), Docker deploy | 🔄 in progress (cron + sentiment snapshot + regime badge done) |

## 4. Architecture (implemented)

Package `src/concinvest/`. Full detail in [`docs/architecture.md`](docs/architecture.md).

```
data/      tickers.py · fetch.py (yfinance, all network) · store.py (SQLite) · portfolio_store.py (Live-tab CSV portfolios)
features/  technical.py · cross_asset.py · sentiment.py (VADER/FinBERT) · analyst.py · options.py
ml/        dataset.py (panel + synthetic gen) · model.py (RF+TSCV) · forecast.py (5 fields) · overlay.py (live sentiment tilt)
portfolio/ state.py (leveraged lots+cash) · tax.py (Abgeltungsteuer) · rules.py (guardrails)
backtest/  engine.py (forecast-driven + rules-based portfolio) · walkforward.py (multi-window)
app/       streamlit_app.py · exit_button.py
config.py · pipeline.py (run_phase1 / fetch_and_store) · cli.py
```

**Data flow:** `fetch → features → store (SQLite) → ml.dataset panel → ml.model →
{forecast, backtest} → app`. Orchestrated by `pipeline.run_phase1`;
`pipeline.fetch_and_store` is the reusable daily-ETL block. **Fetching is incremental
per ticker** (stored tickers pull only the recent tail newer than `store.latest_date`;
unstored tickers self-heal a full pull from `START_DATE`; `--full` forces a full
re-fetch) and **rate-limit-resilient** (a ticker the threaded batch silently drops is
retried individually, then skipped-with-warning). Fetched bars merge with the full
stored history read back via `store.read_ohlcv` so feature windows keep full depth. Full
mechanism + rationale in [docs/architecture.md](docs/architecture.md).

**Database:** 4 tables — `ohlcv_raw`, `daily_market` (Table 1), `sentiment_analyst`
(Table 2), `cross_asset` (Table 3). Raw OHLCV kept separate from derived features.
See [`docs/SCHEMA.md`](docs/SCHEMA.md).

**Model contract:** `ml.dataset.FEATURE_COLS` = technical + cross-asset + **momentum
lags** (each of those carried at `_lag{3,10,30,100}` trading days back) + sentiment
placeholders + action encoding (`is_sell`, `leverage`). Lags give the trees recent
trajectory (strictly past data — no leakage); low-importance ones are pruned. Sentiment
is neutral over history (no historical news feed) and filled live at forecast time.

## 4b. Strategies (selectable in the UI / `concinvest run --strategy`)

Two backtest/forecast books share the model, panel, tax, dividends, and
`BacktestResult` shape (so every UI tab renders either). **Default** is selected
unless changed.

- **Default (balanced)** — `backtest.run_forecast_backtest`: the guardrailed Story.md
  90/10 book (per-name 9% stock / 4.5% 2x / 4.5% 3x) with per-stock confidence
  rebalancing, 33% trim, underlying-dominance, 20%-drawdown de-risk, and the crisis
  100%/2-month-revert path. This is the project's main strategy (Phases 3–4).
- **Aggressive (3x)** — `backtest.run_aggressive_backtest`: an all-3x book with
  **minimal rules** (no underlying-dominance / drawdown-de-risk guardrails, no
  daily-sell cap). Start 90% in 3x (per-name 18%) + 10% cash. Each 3x lot is **stopped
  out fully at −60%** (value ≤ 40% of cost → underlying −20% vs entry; proceeds → cash).
  Once a lot is **+60%** (value ≥ 160% of its take-profit reference) an **ML amount
  (≥30%)** is skimmed, split **50/50** into cash and a permanent **tier-1 underlying**
  buy-and-hold lot (earns dividends, never re-leveraged); the skimmed lot's reference is
  **re-based** to its remainder (needs another +60% to re-trigger). **ML buy events**
  (3x only, confidence ≥ `AGG_ENTRY_THRESHOLD`) deploy a fixed `AGG_ENTRY_CHUNK` (10%)
  of portfolio per name; a **crisis** (`_is_crisis`) deploys the accumulated cash
  hoard into 3x (buy-the-dip). A **per-name concentration cap** (`_agg_cap_overweight`,
  reusing the default's `PER_NAME_CAP` = 33% + `rules.sell_riskiest_first`) trims any name
  whose total (underlying + 3x) exceeds 33% back to the cap, shedding 3x first → cash, so
  a single-stock blow-up can't dominate the book (entries also skip already-capped names).
  Constants live in `config.AGG_*`. Lot-level stop-loss/take-profit reuse
  `state.sell_lot`; the per-lot reference is `Lot.tp_basis`.

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

## 5c. Phase 3 design notes

Durable decisions below; the full run-by-run experiment log (superseded numbers, dropped
levers) lives in git history, not here.

- **Time-honest dataset** — `generate_dataset` returns rows sorted by snapshot date, so
  `TimeSeriesSplit` CV is valid; `n` is arbitrary (Story.md 100k = `concinvest run --n 100000`).
- **Honest validation** — `train_validate_split` carves the last `VALIDATION_YEARS` off by
  calendar date; `run_phase1` trains **only on the pre-validation split**, so the
  validation-window backtest is true out-of-sample.
- **Tuning** — `model.tune` / `tune_and_train` pick the best `PARAM_GRID` entry by mean TSCV
  ROC-AUC (`concinvest run` tunes by default; `--no-tune` to skip; params on `TrainedModel.params`).
- **Feature pruning** — `select_features` drops features below a cutoff that **scales with
  feature count** (`min(MIN_IMPORTANCE, KEEP_UNIFORM_FRAC / n)`, action encoding always kept).
  The absolute floor alone let the 52 momentum-lag columns dilute every importance and
  collapse the model to the action encoding — a degenerate constant-confidence forecaster;
  the count-scaled cutoff keeps ~67/69 features. `TrainedModel.features` records what's used.
- **Exposure mapping** — `_target_name_fraction` holds each name's per-name base weight while
  that name's buy-confidence is ≥ 0.5 and de-risks only below 0.5; names rebalance
  independently (`_rebalance_names_to_target`), so a bearish read on one stock trims only that
  stock (Story.md's per-ticker forecast). The routine confidence-rebalance sells **pro-rata**
  across tiers — tier-grading it cost ~5pp by de-levering in up-markets. Tier-graded
  (riskiest-first) de-risk is reserved for the two Story.md de-risking events (crash drawdown,
  33% trim), where it is performance-neutral.
- **Base case** — re-tilted to **9%/4.5%/4.5%** per name (same 18%/name, more leverage than
  the old 12/3/3), with guardrails **underlying ≥ 2x+3x** (`enforce_underlying_dominance`) and
  a **6% per-name floor / cash < 70%** (`MIN_NAME_WEIGHT`/`MAX_CASH`; book ≥ 30% invested at
  all times, even in a drawdown).
- **Momentum lags** — each technical + cross-asset feature is also carried at
  `_lag{3,10,30,100}` (strictly past data, leading edge → 0); they rank among the top signals.
- **Risk-lever outcome** — Lever 2 (leverage-aware riskiest-first de-risk, `state.sell_tier`)
  shipped; **Lever 1 (vol-aware leverage throttle) was dropped** — de-levering on high VIX cut
  the leverage edge and fought the crisis dip-buy. **Lesson: this basket's edge *is* leverage
  in up-markets** — de-levering on vol is net-negative.
- **Current headline** (walk-forward, `concinvest validate --n 10000`): **≈ 75% win rate (3/4),
  mean outperformance ≈ +11%** vs NASDAQ; high-variance (one window trails ~4pp). **Numbers
  vary run-to-run with the synthetic sample — indicative, not exact.** The remaining real lever
  is a **basket/benchmark review** (concentrated value vs a tech-heavy NASDAQ), not more risk
  tuning — deferred.

## 5d. Phase 4 design notes (in progress)

- **`portfolio/` package** — `state.PortfolioState` holds leveraged lots (tier 1/2/3,
  each with a cost basis) + cash; `mark()` applies daily constant-leverage returns;
  `sell_name()` realizes gains proportionally and pays tax. `build_base_case()` is the
  90/10 book (per-name 9%/4.5%/4.5%).
- **`tax.tax_on_sale`** — 25% flat Abgeltungsteuer; realized losses accumulate in a
  **full-portfolio** carry pool (one pool across all names, never expiring) that offsets
  future gains before tax — i.e. gains and losses net across the whole book over time
  (Story.md). (Only nuance vs. German law: a loss does not retroactively refund tax on a
  gain realized *earlier* the same year — a second-order effect.)
- **`rules`** — deterministic sell-side guardrails, all shedding the **riskiest tier
  first** (3x → 2x → stock, via `sell_riskiest_first`): 33% per-name → trim 3% (the
  post-upstreak case); **underlying ≥ 2x+3x** (`enforce_underlying_dominance`) → sell
  leverage excess when a rally lets the leveraged tiers outgrow the underlying; 20%
  drawdown → de-risk toward cash (the crash case) but **never below the 6% per-name
  floor** (`MIN_NAME_WEIGHT`; the retained floor is underlying-only). Every sell capped
  at 10%/day. `apply_guardrails` runs them per day (de-risk, dominance, then trim). The
  routine confidence-rebalance (in `backtest.engine`) sells **pro-rata** across tiers,
  preserving the leverage edge in up-markets (tier-grading there cost ~5pp in the
  walk-forward — §5c). The 6% floor across 5 names keeps **cash < 70%** (`MAX_CASH`)
  structurally (book ≥ 30% invested at all times). **No order below `MIN_TRADE_EUR`
  (€500)** is placed — gated in `sell_riskiest_first`, `_deploy_name`, `_sell_proportional`
  and the forecast's `apply_book_limits`; this also suppresses the tiny daily
  underlying-dominance micro-trims at the 9≡9 boundary base case.
- **`backtest.run_rules_backtest`** — replays the base-case leveraged book under the
  guardrails vs NASDAQ (sell-side only).
- **`backtest.run_forecast_backtest`** — **each name's** target portfolio fraction
  tracks *that name's* buy-confidence (lagged, scaled by its per-name base weight,
  **floored at the 6% `MIN_NAME_WEIGHT`**) via `_target_name_fraction` +
  `_rebalance_names_to_target`, with a per-name dead-band, cash re-entry, daily
  guardrails (incl. underlying-dominance), and tax. Names rebalance independently
  (Story.md's per-ticker forecast), so a bearish read on one stock trims only that
  stock. **This is the pipeline/UI backtest.** "Optimal weighting" stays delegated to
  ML confidence.
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
  is logged **per tier** with its actual € (deploys split 9/4.5/4.5; the pro-rata rebalance
  sell is decomposed by `_sell_proportional` — selling logic unchanged, returns identical)
  into `BacktestResult.trades`, each day's per-`(ticker, tier)` value into
  `BacktestResult.tier_curve`, and each day's cash balance into `BacktestResult.cash_curve`.
  The **ML: Strategy tab** stacks three panels on a **shared
  x-axis** with a legend: the price curve with aggregated buy/sell markers (total €), the
  **per-tier balance-evolution chart** (stock / 2x / 3x from `tier_curve`) with per-tier
  markers (actual € each), and NASDAQ; the full buy/sell table sits behind a **popover
  button**. Markers are drawn on the decision day (T-1, the signal bar), display-only. The
  asset selector also offers **Cash** as a 6th option — cash (€) over NASDAQ on a shared
  x-axis, from `cash_curve`.
- **Live: Sample Portfolio tab** (first tab) — a persisted, selectable user book (named CSV
  files under `data/portfolios/`, one row per position + a cash row; each lot its own buy date,
  defaulting to today). `pipeline.build_dated_book` derives each lot's current value via the
  daily-rebalanced Nx-leverage path (`_lot_value_path`, matching the backtest's `state.mark`),
  the real cost basis, and a high-water ≥ current (drawdown ≥ 0). **Run live analysis** calls
  `pipeline.recommend_for_portfolio` (reusing the trained model, side-effect-free on a deep
  copy): live news/sentiment → 5-field forecast sized to the book (`apply_book_limits`) +
  sentiment overlay + the strategy's value/cost-aware actions (`_strategy_actions`; default:
  drawdown de-risk / dominance / 33% trim; aggressive: −60% stop-loss / +60% take-profit /
  33% cap). Full UI/chart detail in [docs/architecture.md](docs/architecture.md); the other
  three tabs are the ML views (prefixed **ML:**).
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
uv run pytest                                   # offline (synthetic fixtures)
uv run concinvest run --n 4000                  # live: fetch→model→forecast→backtest
uv run concinvest run --n 4000 --strategy aggressive   # the all-3x book (default: balanced)
uv run concinvest validate --n 10000            # walk-forward (multi-window) vs NASDAQ
uv run concinvest update --sentiment            # daily ETL + dated sentiment snapshot (cron)
uv run concinvest update --sentiment --full     # force full re-fetch (else incremental tail)
uv run concinvest notify --portfolio <name>     # email if that saved book has a trigger
uv run streamlit run src/concinvest/app/streamlit_app.py --server.port 8505
```

Daily cron: `scripts/daily_update.sh` wraps `concinvest update --sentiment` (logs to
the gitignored `data/daily_update.log`); schedule ~22:00 Europe/Berlin via `crontab -e`
(`CRON_TZ=Europe/Berlin`). Each run appends a dated `sentiment_analyst` snapshot so the
live analyst signals accumulate history (the prerequisite to making them trainable).
If `ALERT_PORTFOLIO` is set the script also runs `concinvest notify` (non-fatal),
which emails a buy/sell alert **only when something fires** (`notify.build_alert` +
Resend API; config via env, see `.env.example`). **Deploy:** [`DEPLOY.md`](DEPLOY.md)
(Railway — app + persistent SQLite volume + cron service; `railway.json`); SQLite runs
in WAL mode so the app reads while the cron writes.

- **Unit tests** (`tests/`, offline via `conftest.py` synthetic market): technical
  indicators vs known values, cross-asset ratios (incl. VVIX/GSCI/10y-5y spread),
  sentiment scaling, SQLite upsert/read roundtrip, `latest_date`/`read_ohlcv` helpers +
  incremental `fetch_and_store` (tail-only second fetch, `--full` force, partial-DB
  self-heal, merged full-depth history), additive schema migration, pure
  fetch helpers (IV nearest-strike, finanznachrichten headline parse), dataset
  shape/balance/no-leakage + chronological order + date split, TSCV tuning, model
  train + 5-field forecast, backtest curve, portfolio state/tax/guardrails (incl.
  tier-targeted `sell_tier` + riskiest-first de-risk) + rules-based & forecast-driven
  backtests + trade-log recording, crisis-drop detection, underlying-only dividends,
  leading-holiday benchmark gap, sentiment overlay (tilt/gate/leverage cap),
  walk-forward window construction, daily-ETL sentiment-history accumulation,
  per-stock target fraction (6% floor) + independent-name rebalance, backtest
  final-state, riskiest-tier-first trim, 6% per-name drawdown floor (cash < 70%),
  underlying-dominance leverage trim, €500 min-trade skip (guardrails + forecast),
  forecast book-limits (cash/holdings caps), dated-book derivation
  (`build_dated_book` — per-tier buy dates → daily-rebalanced Nx-leverage current value +
  cost basis + high-water never below current → drawdown ≥ 0), portfolio-store CSV save/load round-trip
  (per-position dates preserved),
  and user-book live recommendations (`recommend_for_portfolio` — actions fire +
  side-effect-free), safe-exit self-SIGTERM (own PID only),
  aggressive-strategy state (`sell_lot`, `tp_basis`, all-3x base case) + helpers
  (stop-loss exit, take-profit skim/re-base/underlying seed, fixed-chunk entries,
  per-name 33% cap + capped-name entry skip) + `run_aggressive_backtest` (3x/stock-only
  book, end-of-window cap holds) + 3x-only forecast restriction, regime classifier
  (`detect_regime` — Rising/Neutral/Falling votes, per-signal explainability, short-history
  robustness), and email-on-trigger alert building (`notify.build_alert` — no-trigger →
  `None`, forecast/strategy-action → rendered subject + body).
- **Live integration**: `concinvest run` prints model CV ROC-AUC, portfolio vs NASDAQ
  return, and the 5-field forecast for all stocks.
- **UI**: app boots on 8505; **Run / refresh** fetches live data; safe-exit button
  SIGTERMs only the app's own process, never the port (so a shared/forwarded port or SSH
  is untouched).

## 7. Remaining phases — detail

- **Phase 3** (✅) — done: time-ordered generator (100k-capable), honest
  date-based train/validate split, TSCV hyperparameter tuning, feature-importance
  pruning, base-case-faithful exposure mapping, walk-forward validation
  (`concinvest validate`), risk-control tightening (Lever 2 riskiest-first de-risk
  shipped; Lever 1 vol throttle evaluated and dropped — §5c), and **per-stock
  confidence rebalancing** (each name trimmed by its own forecast). Current walk-forward
  headline is in **§5c** (single source of truth — not restated here). The remaining real
  lever is a **basket/benchmark review** (not more risk tuning), deferred to a future revisit.
- **Phase 4** (✅) — `portfolio/` `state.py` (leveraged lots + cash + tier-targeted
  `sell_tier`), `tax.py` (25% flat + loss offset), `rules.py` (90/10 base, 33%→trim 3%,
  underlying≥2x+3x, <10%/day sell, 20% drawdown→riskiest-tier-first de-risk to a 6%
  per-name floor / cash<70%);
  `backtest.run_forecast_backtest` (confidence-driven exposure + re-entry + guardrails
  + tax + crisis 100%/2-month-revert + underlying dividends + trade log), wired into
  the pipeline; live sentiment overlay (`ml/overlay.py`) on the forecast; Strategy tab.
- **Phase 5** (🔄) — **done:** daily cron — `pipeline.daily_etl` (OHLCV + features +
  cross-asset + dated `sentiment_analyst` snapshot) behind `concinvest update
  --sentiment`, wrapped by `scripts/daily_update.sh` for ~22:00 Europe/Berlin
  scheduling; the dated snapshots accumulate the analyst-signal history needed to make
  the overlay trainable. **Live: Sample Portfolio tab** — users get strategy-based,
  news/sentiment-aware action recommendations for their own book
  (`pipeline.recommend_for_portfolio`; §5d). **Rising-market regime badge** —
  `features.regime.detect_regime` casts six explainable Fear&Greed-style votes
  (S&P vs 50d MA · S&P vs 125d MA · breadth: stocks above their 125d MA · VIX vs 50d MA ·
  gold vs 50d MA · oil vs 50d MA — for VIX/gold/oil, *below* their MA = bullish, i.e. a
  rising gold/oil is risk-off); bullish fraction > 0.6 = Rising, < 0.4 = Falling, else
  Neutral. Computed in `run_phase1` (`Phase1Result.regime`, from `^GSPC`/`^VIX` + the
  5 stocks + `GC=F`/`CL=F` closes — the dense raw closes, not the NaN-pocked cross-asset
  ratio) and shown atop the **ML: Current market** tab as a **Plotly vote gauge**
  (a verdict-coloured title over a red→white→green gradient arc, the bullish-vote count
  marked by a dark threshold line) + a **diverging
  vote bar** per component (green = bullish, short metric on the bar — breadth as
  "N > 125d MA" — full reason on hover). Pure/offline.
  **Remaining:** Docker deploy.

## 8. Conventions

Apache-2.0 / MIT-compatible licensing. Tests-first for behaviour changes; functions
≤ ~40 lines; small commits. Streamlit on port 8505. Secrets never committed; runtime
`data/` gitignored.

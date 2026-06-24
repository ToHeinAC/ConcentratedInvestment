# Building a Yahoo Finance ML Portfolio Forecasting Database: A Deep Dive

## Executive Summary

This report provides a comprehensive blueprint for building a daily-updated database using Yahoo Finance data mainly via the `yfinance` Python library to power a machine learning–based portfolio recommendation system. It covers the optimal selection of global indices, commodities, macro indicators, and sentiment proxies — all accessible through Yahoo Finance ticker symbols. The report also maps out all available data types from `yfinance` that serve as trend-setting features for short-term, mid-term, and long-term forecasting, including derived features like moving averages and quotients.

The key idea is to build a recommendation system with daily data and daily updates to achieve an optimal perfomance while mitigating major risks.

The app follows the "keep it simple stupid" principle, is graphically modern and user-friendly.

## Basic idea

The following portfolio rules must be applied:
- 5 Stocks (yahoo finance ticker), for now must be:
    - Siemens Aktiengesellschaft (SIE.DE)
    - Münchener Rückversicherungs-Gesellschaft Aktiengesellschaft in München (MUV2.DE)
    - Freeport-McMoRan Inc. (FCX)
    - Tesla, Inc. (TSLA)
    - Samsung Electronics Co., Ltd. (005930.KS)  (replaced ITOCHU Corporation / 8001.T, which is not 3x-tradable at the broker)
    later, a user may be able to set his own 5 stocks (not more than 5!)
- Base case scenario is to be invested in the 5 stocks above with 90% allocation and 10% cash; also the base case is that most of the time i.e. most of the days no trades are made (there have to be severe reasons from the data that trigger a trade)
- Allocations must be obey the following rules:
    -   Best allocoation for optimized performance
    -   You are allowed to open 2x and 3x leveraged positions (only long)
    -   Optimal weighting of all positions such that a drawdown of max 20% of the portfolio value occurs; when this is reached, adjust the allocations based on the forecast to cash
    -   You must trim each stock position (in total with leveraged positions included) as soon as one stock position (with leveraged positions included) exceeds 33% of the total portfolio value. Trimming must be 3% of the portfolio values
    -   Each individual selling must be less than 10% portfolio size per day
    -   Take dividend payments into account, those are possible only for the underlying stocks themselves, not leveraged positions
    -   You can temporarily go 100% allocation with 0% cash in case a major pullback or black swan occurred, but within 2 month you must go back to the base case scenario
    -   Take German capital gain taxes into account: gains taxed at a flat 25% “Abgeltungsteuer”, realised capital losses from investments can generally be used to offset capital gains from other investments so that you are taxed only on the net result

The following tech ideas must be met:
- Catch > 5 years of data for all (stocks, indices, commodities etc.), start at 01.01.2020
- From this database, build a ML-Learning dataset consisting of 100.000 stochastic datapoints (50.000 buys, 50.000 sells; each datapoint is a snapshot of the market at a given time with all available features): Random buys (random time, random stock including 2x and 3x positions, fixed position size 10% portfolio), and sellings of random portfolio allocations that are there and that meet the above rules, especially you can only buy if you have cash at hand and you can only sell out of open positions that you hold, not more; each datapoint is a snapshot of the current market conditions. Then the evaluation of any selling must check the performance of this selling with respect to former buys; this dataset shall be used for the ML in order to build a good forecasting model  
- Build a ML model that uses the dataset to predict the optimal selling strategy based on the current market conditions; use the first 4 years as data and the latest one year for validations, Starting point at 01.01.2020 is 100k Euro, 80% stocks, 20% cash, each stocks dollar allocation is in absolute percentages 12% stock, 3% leveraged 2x, 3% leveraged 3x. The model must be able to outperform the market over a period of 5 years, i.e. better than NASDAQ over the last 5 years
- Forecast must consist of those 5 fields: stock ticker, buy or sell, amount in Euro, stock or 2x or 3x, confidence
- Build a basic streamlit web interface to display the data and the portfolio performance
- Build a cron job to update the data daily

***

## Recommended Global Indices (5 Picks)

For a globally diversified ML model, the indices should cover the major economic regions (US, Europe, Asia) and represent different market segments. Based on research into cross-market correlations and ML forecasting literature, the following five indices provide the best geographic and economic coverage:[1][2]

| Index | Yahoo Ticker | Region | Rationale |
|-------|-------------|--------|-----------|
| S&P 500 | `^GSPC` | US | Benchmark for global risk appetite; most liquid market globally[3][4] |
| NASDAQ Composite | `^IXIC` | US | Tech-heavy; captures growth/innovation sector dynamics[3] |
| DAX | `^GDAXI` | Europe (Germany) | Europe's largest economy; export-oriented, sensitive to global trade[3][5] |
| FTSE 100 | `^FTSE` | UK | Commodity- and finance-heavy; different sector composition from DAX[3][1] |
| Nikkei 225 | `^N225` | Japan | Major Asian benchmark; captures yen dynamics and Asian risk[3][4] |

**Optional 6th Index:** Shanghai Composite (`000001.SS`) or Hang Seng (`^HSI`) for direct China/emerging market exposure. The Shanghai Composite is particularly useful since China drives much of the global commodity demand cycle.[6][7]

**Why these five?** Academic research on ML-based stock prediction consistently uses S&P 500, FTSE 100, DAX 30, NASDAQ, and Nikkei 225 as the core global indices for cross-market correlation analysis. These indices span the three major trading time zones (Asia → Europe → US), creating a natural information flow that your model can exploit.[5][8][1]

***

## Complete Yahoo Finance Ticker Universe for Your Database

### Core Commodities & Macro Assets

| Asset | Yahoo Ticker | Category | Trend Horizon |
|-------|-------------|----------|---------------|
| Gold Futures | `GC=F` | Precious Metal / Safe Haven | Mid-to-Long term[9] |
| Bitcoin | `BTC-USD` | Crypto / Risk Sentiment | Short-to-Mid term |
| WTI Crude Oil | `CL=F` | Energy | All horizons[9] |
| Wheat Futures | `ZW=F` | Agriculture | Mid-to-Long term |
| Copper Futures | `HG=F` | Industrial Metal | Mid-to-Long term (economic bellwether) |
| US Dollar Index | `DX-Y.NYB` | Currency | All horizons[10][11] |
| CBOE VIX | `^VIX` | Volatility / Fear Gauge | Short-to-Mid term[12][13] |

### Additional Trend-Setting Data (Highly Recommended)

These assets and indicators are available via Yahoo Finance and significantly enhance forecasting power:

| Asset | Yahoo Ticker | Category | Why It Matters |
|-------|-------------|----------|----------------|
| 10-Year Treasury Yield | `^TNX` | Bonds / Rate Expectations | Yield direction sets equity risk premium[14][11] |
| 30-Year Treasury Yield | `^TYX` | Bonds / Long-Term Rates | Long-term inflation expectations[4] |
| 5-Year Treasury Yield | `^FVX` | Bonds / Mid-Term Rates | Mid-term rate cycle[4] |
| 13-Week T-Bill | `^IRX` | Bonds / Short-Term Rates | Fed policy proxy / risk-free rate[3] |
| Brent Crude Oil | `BZ=F` | Energy | European/global oil benchmark |
| Silver Futures | `SI=F` | Precious Metal | Industrial + monetary metal |
| Ethereum | `ETH-USD` | Crypto | Risk-on/DeFi sentiment proxy |
| Solana | `SOL-USD` | Crypto | Financial Infrastructure + Altcoin sentiment |
| VIX of VIX (VVIX) | `^VVIX` | Volatility of Volatility | Extreme tail-risk signal[3] |
| S&P GSCI Commodity Index | `^SPGSCI` | Broad Commodities | Overall commodity cycle[3] |

### Yield Curve Spread (Derived Feature)

The 10Y-2Y Treasury spread is one of the most powerful recession/expansion indicators. As of March 9, 2026, it stands at 0.56%. You can compute this as `^TNX - (2-year yield)`. Since Yahoo Finance does not carry a direct 2-year yield ticker, you can either:[15]
- Derive it from the FRED API (`fredapi` Python package, series `T10Y2Y`)[16][17]
- Use the 5-year (`^FVX`) minus 13-week (`^IRX`) as an alternative slope measure

***

## All Available yfinance Data Types for Your ML Features

The `yfinance` library's `Ticker` object provides an extensive set of attributes and methods. Here is the complete mapping organized by use case:[18][19]

### Price & Volume Data (Primary OHLCV)

Retrieved via `ticker.history()` with customizable parameters:[20][21]

- **Fields:** Open, High, Low, Close, Volume, Dividends, Stock Splits
- **Intervals:** `1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo`
- **Periods:** `1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max`
- **Custom date ranges:** via `start` and `end` parameters

For your daily database, use `interval='1d'` and fetch with `start/end` date parameters for incremental updates. Intraday data (1m-90m) is limited to the last 60 days.[20]

### Analyst Sentiment & Consensus Data

This is the closest thing to "sentiment data" directly available from Yahoo Finance:[22][23]

| Attribute | Data Returned | Update Frequency |
|-----------|--------------|------------------|
| `recommendations` | period, strongBuy, buy, hold, sell, strongSell (last 4 months)[24][25] | Monthly |
| `recommendations_summary` | Same as above, aggregated | Monthly |
| `upgrades_downgrades` | date, firm, toGrade, fromGrade, action[22][26] | As they occur |
| `analyst_price_targets` | current, low, high, mean, median[19] | Rolling |
| `earnings_estimate` | numberOfAnalysts, avg, low, high, yearAgoEps, growth[23] | Quarterly cycle |
| `revenue_estimate` | numberOfAnalysts, avg, low, high, yearAgoRevenue, growth[23] | Quarterly cycle |
| `eps_trend` | current, 7daysAgo, 30daysAgo, 60daysAgo, 90daysAgo[23] | Weekly |
| `eps_revisions` | upLast7days, upLast30days, downLast7days, downLast30days[23] | Weekly |
| `growth_estimates` | stock, industry, sector, index (for 0q, +1q, 0y, +1y, +5y, -5y)[23] | Monthly |

The `eps_trend` and `eps_revisions` are particularly valuable for ML — they capture the *direction and velocity* of analyst estimate changes, which are strong leading indicators.

### News Data (for NLP Sentiment)

`ticker.get_news(count=10, tab='news'|'all'|'press releases')` returns up to 10 recent news articles per ticker with titles, URLs, and related tickers[27][28]. You can:

- Run NLP sentiment analysis (VADER, TextBlob, or FinBERT) on the headlines[29][30]
- Aggregate daily sentiment scores per stock (integer scale from -3 (strong negative) to +3 (strong positive))
- Track sentiment momentum (rolling average of sentiment scores)

**Caveat:** The news feed returns only ~8-10 most recent articles and relevance can vary. For production-grade sentiment, consider supplementing with a dedicated news API (e.g., NewsAPI, Benzinga).[27][28]

### ESG / Sustainability Data

`ticker.sustainability` returns ESG risk scores from Sustainalytics:[31][32]

- Total Score, Environment Score (E), Social Score (S), Governance Score (G)
- Controversy level, peer comparisons
- Note: Lower scores = lower risk (methodology changed in Nov 2019)[31]

ESG data is useful as a longer-term structural feature, not for daily trading signals.

### Options Data (Sentiment Proxy)

`ticker.option_chain(date)` provides full options chains (calls and puts) with strike prices, volume, open interest, and implied volatility. From this you can derive:[33][34]

- **Put/Call Ratio (PCR):** Sum of put open interest / sum of call open interest — a classic fear gauge[34]
- **Implied Volatility Skew:** Difference between OTM put IV and ATM call IV
- **Volume/Open Interest Ratio:** Identifies unusual options activity[34]
- **Max Pain Price:** Where most options expire worthless[33]

### Fundamental Data (Longer-Term Features)

| Attribute | Description |
|-----------|-------------|
| `info` | Dictionary with 100+ fields: sector, industry, marketCap, trailingPE, forwardPE, beta, dividendYield, profitMargins, revenueGrowth, earningsGrowth, debtToEquity, currentRatio, returnOnEquity, etc.[35][36] |
| `financials` / `quarterly_financials` | Income statement[20] |
| `balance_sheet` / `quarterly_balance_sheet` | Balance sheet[20] |
| `cash_flow` / `quarterly_cash_flow` | Cash flow statement[20] |
| `earnings_dates` | Upcoming and past earnings dates with EPS estimates/actuals[19] |
| `earnings_history` | epsEstimate, epsActual, epsDifference, surprisePercent[23] |

### Ownership / Insider Activity

| Attribute | Use Case |
|-----------|----------|
| `institutional_holders` | Tracks institutional ownership changes[20] |
| `mutualfund_holders` | Mutual fund positioning |
| `insider_transactions` | Insider buy/sell signals[18] |
| `insider_purchases` | Net insider buying[18] |
| `major_holders` | Ownership concentration |

***

## Feature Engineering: Moving Averages & Quotients

### Moving Averages (Short / Mid / Long Term)

All of these are computed from the daily OHLCV data retrieved via `ticker.history()`:

| Feature | Formula | Horizon | Purpose |
|---------|---------|---------|---------|
| SMA(5) | 5-day simple moving average | Short-term | Weekly trend |
| SMA(10) | 10-day SMA | Short-term | Bi-weekly momentum |
| SMA(20) | 20-day SMA | Short-term | ~1 month trend |
| SMA(50) | 50-day SMA | Mid-term | Quarterly trend |
| SMA(100) | 100-day SMA | Mid-term | Semi-annual trend |
| SMA(200) | 200-day SMA | Long-term | Bull/bear regime |
| EMA(12) | 12-day exponential MA | Short-term | MACD fast line |
| EMA(26) | 26-day exponential MA | Mid-term | MACD slow line |
| EMA(50) | 50-day EMA | Mid-term | Responsive quarterly trend[37] |

Research shows that ML models achieve significantly better accuracy predicting changes in moving averages than raw price levels — predicting the MA 40 steps ahead can be as accurate as predicting price 1 step ahead.[38]

### Quotients / Ratios (Key Derived Features)

| Feature | Formula | Signal |
|---------|---------|--------|
| Price/SMA(50) | Close / SMA(50) | >1 = above trend, <1 = below trend[37] |
| Price/SMA(200) | Close / SMA(200) | Bull/bear regime indicator |
| SMA(50)/SMA(200) | Golden/Death Cross ratio | Trend reversal signal |
| Volume/SMA_Vol(20) | Volume / 20-day avg volume | Volume surge detection |
| Gold/Oil Ratio | GC=F / CL=F | Risk/growth sentiment |
| Gold/Dollar Ratio | GC=F / DX-Y.NYB | Monetary policy proxy |
| Copper/Gold Ratio | HG=F / GC=F | Economic growth expectations |
| VIX/VIX_SMA(20) | VIX / 20-day VIX average | Fear spike detection |
| Stock/Index Ratio | Stock price / ^GSPC | Relative strength vs. market |

### Additional Technical Indicators (from OHLCV)

| Indicator | Inputs | Horizon |
|-----------|--------|---------|
| RSI(14) | Close prices | Short-term momentum |
| MACD | EMA(12) - EMA(26), signal = EMA(9) of MACD | Mid-term trend |
| Bollinger Bands | SMA(20) ± 2×StdDev(20) | Volatility regime[39] |
| ATR(14) | High, Low, Close | Volatility measure |
| OBV | Volume × direction | Volume-price divergence |
| Rate of Change (ROC) | (Close - Close_n) / Close_n | Momentum at various periods |

***

## Sentiment Data: What Yahoo Finance Can (and Cannot) Provide

### Direct Sentiment from yfinance

Yahoo Finance provides several *proxy* sentiment measures directly through `yfinance`:

- **Analyst Consensus:** `recommendations` gives you the buy/hold/sell distribution. A shift from "hold" to "buy" across multiple firms is a bullish signal.[24][25]
- **EPS Revision Momentum:** `eps_revisions` tracks the number of upward vs. downward revisions in the last 7 and 30 days — one of the strongest alpha signals in quantitative finance.[23]
- **Upgrade/Downgrade Flow:** `upgrades_downgrades` provides timestamped analyst rating changes with firm names.[26][22]
- **Earnings Surprise History:** `earnings_history` shows how much actual EPS deviated from estimates — persistent beat/miss patterns predict future behavior.[23]
- **Options Put/Call Ratio:** Derived from `option_chain()` data — a classic contrarian sentiment indicator.[33][34]

### VIX as Sentiment

The VIX (`^VIX`) is the market's "fear gauge," measuring expected 30-day volatility of the S&P 500. The CNN Fear & Greed Index itself uses the VIX as one of its seven components, alongside put/call ratio, safe haven demand, junk bond demand, stock price breadth, stock price strength, and market momentum.[40][41][42][13]

You can replicate several Fear & Greed Index components using Yahoo Finance data:
- **Market Momentum:** S&P 500 (`^GSPC`) price vs. 125-day MA
- **Stock Price Strength:** Track 52-week highs vs. lows across your stock universe
- **Market Volatility:** VIX (`^VIX`) vs. its 50-day MA[41]
- **Safe Haven Demand:** Compare `^TNX` (treasury yield) changes vs. `^GSPC` returns over 20 days[40]

### News Sentiment (Requires NLP Processing)

`ticker.get_news()` provides headlines that can be processed with NLP tools:[30][29]
- **VADER** (from NLTK): Fast, rule-based sentiment scoring
- **TextBlob**: Simple polarity/subjectivity scoring
- **FinBERT**: Transformer-based, trained on financial text — most accurate for financial sentiment

The workflow is: fetch news → extract titles/descriptions → run sentiment model → aggregate daily score → store in database.[43][29]

### What Yahoo Finance Cannot Provide Directly

- **Social media sentiment** (Reddit, Twitter/X) — requires separate APIs
- **CNN Fear & Greed Index** as a single number — must be replicated from components
- **Real-time order flow** or **institutional trade data**
- **Historical news archives** — only recent articles are available[28][44]

***

## Database Schema Recommendation

### Table 1: Daily Market Data (updated daily)

| Column | Source | Example Tickers |
|--------|--------|-----------------|
| date | - | - |
| ticker | - | ^GSPC, GC=F, BTC-USD, etc. |
| open, high, low, close, volume | `history()` | All tickers |
| sma_5, sma_10, sma_20, sma_50, sma_100, sma_200 | Computed | All tickers |
| ema_12, ema_26, ema_50 | Computed | All tickers |
| rsi_14, macd, macd_signal, bollinger_upper, bollinger_lower | Computed | All tickers |
| price_sma50_ratio, price_sma200_ratio, sma50_sma200_ratio | Computed | All tickers |
| volume_sma20_ratio | Computed | All tickers |

### Table 2: Sentiment & Analyst Data (updated daily/weekly)

| Column | Source |
|--------|--------|
| date, ticker | - |
| recommendation_mean | `info['recommendationMean']` |
| strong_buy, buy, hold, sell, strong_sell | `recommendations` |
| eps_revision_up_7d, eps_revision_down_7d | `eps_revisions` |
| analyst_target_mean, analyst_target_high, analyst_target_low | `analyst_price_targets` |
| news_sentiment_score | NLP on `get_news()` |
| put_call_ratio | Derived from `option_chain()` |

### Table 3: Cross-Asset Features (updated daily)

| Column | Source |
|--------|--------|
| date | - |
| gold_oil_ratio | GC=F / CL=F |
| copper_gold_ratio | HG=F / GC=F |
| vix_level, vix_sma20_ratio | ^VIX |
| yield_10y, yield_spread_10y_5y | ^TNX, ^FVX |
| dollar_index | DX-Y.NYB |
| btc_sma20_ratio | BTC-USD |

***

## Supplementary Data Sources (Beyond Yahoo Finance)

For a production-grade ML system, consider augmenting with:

| Source | Data | Python Package | Free Tier |
|--------|------|----------------|-----------|
| FRED (Federal Reserve) | Yield curve, unemployment, CPI, Fed Funds rate, GDP[16][17] | `fredapi` | Yes (API key required) |
| NewsAPI | Historical news for sentiment analysis[29] | `newsapi-python` | Limited (1 month) |
| CBOE | Put/Call ratios, VIX term structure[45] | Web scraping | Yes |
| Finviz | Stock screener data, analyst ratings | `finvizfinance` | Yes |
| Alpha Vantage | Technical indicators pre-computed | `alpha_vantage` | Yes (rate limited) |

***

## Implementation Notes

- **Daily Update Schedule:** Run your data pipeline after US market close (~22:00 CET for you in Hamburg). Use `yf.download()` with `threads=True` for batch downloading multiple tickers efficiently.[20]
- **Rate Limiting:** Yahoo Finance may throttle requests. Use `session` parameter with retries and respect a ~0.5s delay between requests for non-OHLCV data.
- **Data Storage:** SQLite for prototyping, PostgreSQL for production. Store raw OHLCV separately from computed features to allow re-computation when changing parameters.
- **Moving Average Computation:** Use `pandas.rolling()` for SMA and `pandas.ewm()` for EMA. Compute these as part of your ETL pipeline, not on-the-fly during model training.[37]
- **Feature Scaling:** Moving average quotients (price/SMA ratios) are naturally normalized around 1.0, making them excellent ML features without additional scaling.[39]

# Additional Data
- sentiment: https://www.finanzen.net/news/news_suchergebnis.asp
- sentiment: https://www.finanznachrichten.de/suche/uebersicht.htm?suche=munich%20re
Run NLP sentiment analysis (VADER, TextBlob, or FinBERT) on the headlines

# The Forecast Part
 
- Machine Learning sklearn random forest 
- Cross-validation with time series split
- Feature importance analysis
- Forecast must consist of those 5 fields: stock ticker, buy or sell, amount in Euro, stock or 2x or 3x, confidence
- The web-UI must display all important forecast data, e.g. a 10x10 correlation matrix that shows the most significant correlations and anticorrelations of the performance for the current (last days) condition of the market and if the conditions of raising markets are detected

# Tech guidance
- Python 3.11+
- streamlit frontend
- uv for package management
- pytest for testing
- pandas for data manipulation
- docker for deployment

# Later: 
- app payment integration
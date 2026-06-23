"""Command-line entrypoint.

Phase 0: only scaffolding commands exist. Subcommands (update, train, forecast,
backtest) are wired in later phases.
"""

from __future__ import annotations

import argparse

from . import __version__, config
from .data import tickers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="concinvest", description=__doc__)
    parser.add_argument("--version", action="version", version=f"concinvest {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Print configuration and ticker universe summary")

    p_update = sub.add_parser("update", help="Fetch core tickers and store features (daily ETL)")
    p_update.add_argument("--start", default=str(config.START_DATE))
    p_update.add_argument("--sentiment", action="store_true",
                          help="also snapshot dated analyst/sentiment rows (daily cron)")

    p_run = sub.add_parser("run", help="Run the full Phase 1 slice and print results")
    p_run.add_argument("--start", default=str(config.START_DATE))
    p_run.add_argument("--n", type=int, default=4000,
                       help="synthetic datapoints (Story.md target: 100000)")
    p_run.add_argument("--sentiment", action="store_true", help="fetch live sentiment")
    p_run.add_argument("--no-tune", dest="tune", action="store_false",
                       help="skip TimeSeriesSplit hyperparameter tuning")

    p_val = sub.add_parser("validate", help="Walk-forward (multi-window) validation vs NASDAQ")
    p_val.add_argument("--start", default=str(config.START_DATE))
    p_val.add_argument("--n", type=int, default=10000, help="synthetic datapoints")
    p_val.add_argument("--windows", type=int, default=4, help="number of validation windows")
    p_val.add_argument("--window", type=int, default=252, help="window length in trading days")
    p_val.add_argument("--no-tune", dest="tune", action="store_false",
                       help="skip per-fold hyperparameter tuning (faster)")

    args = parser.parse_args(argv)

    if args.command == "info":
        config.ensure_dirs()
        print(f"concinvest {__version__}")
        print(f"db: {config.DB_PATH}")
        print(f"start date: {config.START_DATE}")
        print(f"portfolio stocks: {', '.join(tickers.STOCKS)}")
        print(f"core slice: {len(tickers.CORE_TICKERS)} tickers")
        print(f"full universe: {len(tickers.ALL_TICKERS)} tickers")
        return 0

    if args.command == "update":
        if args.sentiment:
            from .pipeline import daily_etl

            s = daily_etl(start=args.start)
            print(f"daily ETL {s['as_of']}: {s['tickers']} tickers, {s['stocks']} "
                  f"stocks, {s['cross_rows']} cross rows, {s['sentiment_rows']} "
                  f"sentiment rows; db: {config.DB_PATH}")
            return 0
        from .pipeline import fetch_and_store

        market, cross, raw = fetch_and_store(tickers.ALL_TICKERS, start=args.start)
        print(f"stored {len(raw)} tickers; {len(market)} stocks with features")
        print(f"cross-asset rows: {len(cross)}; db: {config.DB_PATH}")
        return 0

    if args.command == "run":
        from .ml.forecast import forecasts_to_frame
        from .pipeline import run_phase1

        res = run_phase1(start=args.start, n_dataset=args.n,
                         with_sentiment=args.sentiment, tune=args.tune)
        bt = res.backtest
        print(f"model CV ROC-AUC (mean): {res.model.mean_cv:.3f} | params: {res.model.params}")
        print(f"portfolio return: {bt.portfolio_return:+.1%} | "
              f"NASDAQ: {bt.benchmark_return:+.1%} | "
              f"{'BEATS' if bt.beats_benchmark else 'below'} benchmark")
        print("forecast:")
        fc = forecasts_to_frame(res.forecasts)
        print(fc.to_string(index=False) if not fc.empty else "  hold (no trade triggered)")
        return 0

    if args.command == "validate":
        from .pipeline import run_walkforward

        res = run_walkforward(start=args.start, n_dataset=args.n,
                              n_windows=args.windows, window=args.window, tune=args.tune)
        if res.windows.empty:
            print("no validation windows (insufficient history)")
            return 0
        df = res.windows.copy()
        for col in ("portfolio", "benchmark", "outperformance"):
            df[col] = df[col].map(lambda v: f"{v:+.1%}")
        print(df.to_string(index=False))
        print(f"\nwin rate vs NASDAQ: {res.win_rate:.0%} "
              f"({int(res.windows['beats'].sum())}/{len(res.windows)}) | "
              f"mean outperformance: {res.mean_outperformance:+.1%}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

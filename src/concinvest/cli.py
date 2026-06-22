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

    p_run = sub.add_parser("run", help="Run the full Phase 1 slice and print results")
    p_run.add_argument("--start", default=str(config.START_DATE))
    p_run.add_argument("--n", type=int, default=4000,
                       help="synthetic datapoints (Story.md target: 100000)")
    p_run.add_argument("--sentiment", action="store_true", help="fetch live sentiment")
    p_run.add_argument("--no-tune", dest="tune", action="store_false",
                       help="skip TimeSeriesSplit hyperparameter tuning")

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

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

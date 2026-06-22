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

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

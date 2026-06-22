"""Smoke tests for the ticker universe and config (Phase 0)."""

from concinvest import config
from concinvest.data import tickers


def test_exactly_five_portfolio_stocks():
    assert len(tickers.STOCKS) == 5
    assert "SIE.DE" in tickers.STOCKS
    assert "TSLA" in tickers.STOCKS


def test_all_tickers_deduplicated():
    assert len(tickers.ALL_TICKERS) == len(set(tickers.ALL_TICKERS))


def test_core_slice_is_subset_of_universe_or_benchmark():
    # Core slice tickers should all have a known name.
    for t in tickers.CORE_TICKERS:
        assert t in tickers.NAMES


def test_benchmark_is_nasdaq():
    assert config.BENCHMARK_TICKER == "^IXIC"
    assert "^IXIC" in tickers.INDICES


def test_base_allocation_sums_to_one():
    assert config.BASE_STOCK_ALLOCATION + config.BASE_CASH_ALLOCATION == 1.0

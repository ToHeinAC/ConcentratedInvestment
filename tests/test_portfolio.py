"""Tests for the portfolio package: state, German tax, and risk guardrails."""

from concinvest import config
from concinvest.portfolio import rules, state, tax


# --- tax -----------------------------------------------------------------
def test_tax_gain_is_taxed_flat():
    due, carry = tax.tax_on_sale(1000.0, 0.0)
    assert due == 1000.0 * config.CAPITAL_GAINS_TAX_RATE
    assert carry == 0.0


def test_tax_loss_accumulates_and_offsets():
    due, carry = tax.tax_on_sale(-400.0, 0.0)
    assert due == 0.0 and carry == 400.0
    # Next gain is offset by the carried loss before tax.
    due2, carry2 = tax.tax_on_sale(1000.0, carry)
    assert carry2 == 0.0
    assert due2 == (1000.0 - 400.0) * config.CAPITAL_GAINS_TAX_RATE


# --- state ---------------------------------------------------------------
def test_base_case_allocation_90_10():
    st = state.build_base_case(100_000.0, stocks=["A", "B", "C", "D", "E"])
    assert abs(st.total_value() - 100_000.0) < 1e-6
    assert abs(st.cash - 10_000.0) < 1e-6  # 90% invested, 10% cash
    assert abs(st.name_value("A") - 18_000.0) < 1e-6  # 12% + 3% + 3%


def test_mark_applies_leverage_multiplier():
    st = state.PortfolioState(cash=0.0, high_water=300.0)
    st.lots = [state.Lot("A", 1, 100, 100), state.Lot("A", 3, 100, 100)]
    st.mark({"A": 0.10})  # +10% underlying
    # stock lot +10%, 3x lot +30%
    vals = sorted(lot.value for lot in st.lots)
    assert abs(vals[0] - 110.0) < 1e-9
    assert abs(vals[1] - 130.0) < 1e-9


def test_sell_realizes_gain_net_of_tax():
    st = state.PortfolioState(cash=0.0, high_water=200.0)
    st.lots = [state.Lot("A", 1, 100.0, 200.0)]  # cost 100, now worth 200
    net = st.sell_name("A", 200.0)  # sell all; gain 100 -> tax 25
    assert abs(net - 175.0) < 1e-6
    assert abs(st.cash - 175.0) < 1e-6
    assert st.lots == []


def test_dividends_pay_underlying_only_net_of_tax():
    st = state.PortfolioState(cash=0.0, high_water=200.0)
    st.lots = [state.Lot("A", 1, 100.0, 1000.0), state.Lot("A", 3, 100.0, 1000.0)]
    net = st.pay_dividends({"A": 0.02})  # 2% dividend day
    # Only the tier-1 lot pays: 1000 * 0.02 = 20 gross, net of 25% tax = 15.
    expected = 1000.0 * 0.02 * (1.0 - config.CAPITAL_GAINS_TAX_RATE)
    assert abs(net - expected) < 1e-9
    assert abs(st.cash - expected) < 1e-9


# --- rules ---------------------------------------------------------------
def test_trim_fires_above_per_name_cap():
    st = state.PortfolioState(cash=30.0, high_water=100.0)
    st.lots = [state.Lot("A", 1, 50.0, 50.0), state.Lot("B", 1, 20.0, 20.0)]
    total = st.total_value()  # 100; A is 50% > 33%, B is 20% < 33%
    trades = rules.trim_overweight(st)
    assert len(trades) == 1 and trades[0].ticker == "A"
    assert abs(trades[0].amount_eur - config.TRIM_FRACTION * total) < 1e-6  # 3% of 100


def test_no_trim_when_balanced():
    st = state.build_base_case(100_000.0, stocks=["A", "B", "C", "D", "E"])
    assert rules.trim_overweight(st) == []  # each name 18% < 33%


def test_drawdown_derisk_caps_each_sell_at_ten_percent():
    st = state.PortfolioState(cash=0.0, high_water=100.0)
    st.lots = [state.Lot("A", 1, 75.0, 75.0)]  # now 75 after a 25% drawdown
    trades = rules.drawdown_derisk(st)
    assert len(trades) == 1
    # Single sell capped at 10% of portfolio value (75), so 7.5 sold.
    assert abs(trades[0].amount_eur - 7.5) < 1e-6
    assert st.cash > 0.0


def test_no_derisk_within_threshold():
    st = state.PortfolioState(cash=0.0, high_water=100.0)
    st.lots = [state.Lot("A", 1, 90.0, 90.0)]  # 10% drawdown < 20%
    assert rules.drawdown_derisk(st) == []

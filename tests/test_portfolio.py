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
    assert abs(st.name_value("A") - 18_000.0) < 1e-6  # 9% + 4.5% + 4.5%


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


# --- aggressive strategy state helpers -----------------------------------
def test_buy_sets_take_profit_basis():
    st = state.PortfolioState(cash=10_000.0, high_water=10_000.0)
    st.buy("A", 3, 5_000.0)
    assert st.lots[0].tp_basis == 5_000.0  # re-base reference initialised to entry cost


def test_sell_lot_sells_single_lot_net_of_tax():
    st = state.PortfolioState(cash=0.0, high_water=200.0)
    st.lots = [state.Lot("A", 3, 100.0, 200.0), state.Lot("A", 3, 100.0, 100.0)]
    net = st.sell_lot(st.lots[0], 200.0)  # sell the first lot fully; gain 100 -> tax 25
    assert abs(net - 175.0) < 1e-6
    assert abs(st.cash - 175.0) < 1e-6
    assert [(lot.cost_basis, lot.value) for lot in st.lots] == [(100.0, 100.0)]


def test_aggressive_base_case_is_all_3x():
    st = state.build_base_case(100_000.0, stocks=["A", "B", "C", "D", "E"],
                               split=config.AGG_BASE_SPLIT)
    assert abs(st.cash - 10_000.0) < 1e-6  # 90% invested, 10% cash
    assert all(lot.tier == 3 for lot in st.lots)  # only 3x
    assert abs(st.name_value("A") - 18_000.0) < 1e-6  # per-name 18%


# --- rules ---------------------------------------------------------------
def test_trim_fires_above_per_name_cap():
    st = state.PortfolioState(cash=30_000.0, high_water=100_000.0)
    st.lots = [state.Lot("A", 1, 50_000.0, 50_000.0), state.Lot("B", 1, 20_000.0, 20_000.0)]
    total = st.total_value()  # 100k; A is 50% > 33%, B is 20% < 33%
    trades = rules.trim_overweight(st)
    assert len(trades) == 1 and trades[0].ticker == "A"
    assert abs(trades[0].amount_eur - config.TRIM_FRACTION * total) < 1e-6  # 3% of 100k


def test_trim_sheds_riskiest_tier_first():
    st = state.PortfolioState(cash=0.0, high_water=100_000.0)
    # A is 40% (>33%): stock 30k + 3x 10k; the trim should come out of the 3x first.
    st.lots = [state.Lot("A", 1, 30_000.0, 30_000.0), state.Lot("A", 3, 10_000.0, 10_000.0)]
    trades = rules.trim_overweight(st)
    assert trades[0].tier == 3


def test_no_trim_when_balanced():
    st = state.build_base_case(100_000.0, stocks=["A", "B", "C", "D", "E"])
    assert rules.trim_overweight(st) == []  # each name 18% < 33%


def test_drawdown_derisk_caps_each_sell_at_ten_percent():
    st = state.PortfolioState(cash=0.0, high_water=100_000.0)
    st.lots = [state.Lot("A", 1, 75_000.0, 75_000.0)]  # now 75k after a 25% drawdown
    trades = rules.drawdown_derisk(st)
    assert len(trades) == 1
    # Single sell capped at 10% of portfolio value (75k), so 7.5k sold.
    assert abs(trades[0].amount_eur - 7_500.0) < 1e-6
    assert st.cash > 0.0


def test_no_derisk_within_threshold():
    st = state.PortfolioState(cash=0.0, high_water=100_000.0)
    st.lots = [state.Lot("A", 1, 90_000.0, 90_000.0)]  # 10% drawdown < 20%
    assert rules.drawdown_derisk(st) == []


# --- leverage-aware de-risk ----------------------------------------------
def test_sell_tier_sells_only_that_tier():
    st = state.PortfolioState(cash=0.0, high_water=300.0)
    st.lots = [state.Lot("A", 1, 100.0, 100.0), state.Lot("A", 3, 100.0, 100.0)]
    st.sell_tier("A", 3, 100.0)  # sell the whole 3x lot
    assert [(lot.tier, lot.value) for lot in st.lots] == [(1, 100.0)]
    assert abs(st.cash - 100.0) < 1e-9  # no gain (cost==value) -> no tax


def test_drawdown_derisk_sells_riskiest_tier_first():
    st = state.PortfolioState(cash=0.0, high_water=100_000.0)
    st.lots = [state.Lot("A", 1, 30_000.0, 30_000.0),
               state.Lot("A", 3, 45_000.0, 45_000.0)]  # 25% drawdown
    trades = rules.drawdown_derisk(st)
    assert trades[0].tier == 3  # 3x cut before stock
    # First sell capped at 10% of portfolio (75k) = 7.5k, taken from the 3x lot.
    assert abs(trades[0].amount_eur - 7_500.0) < 1e-6


def test_drawdown_derisk_respects_min_name_floor():
    st = state.PortfolioState(cash=0.0, high_water=140_000.0)
    # total=100k (28% drawdown). A sits near the 6% floor; the floor (not the 10%/day
    # cap) binds for it, so A is sold down to exactly 6% and no further.
    st.lots = ([state.Lot("A", 1, 8_000.0, 8_000.0)]
               + [state.Lot(c, 1, 23_000.0, 23_000.0) for c in "BCDE"])
    rules.drawdown_derisk(st)
    for c in "ABCDE":
        assert st.name_value(c) >= config.MIN_NAME_WEIGHT * 100_000.0 - 1e-6  # >= 6% floor
    assert abs(st.name_value("A") - config.MIN_NAME_WEIGHT * 100_000.0) < 1e-6  # floored 6%
    # 5 names x 6% floor => at least 30% invested => cash stays < 70% (Story.md).
    assert st.cash <= config.MAX_CASH * 100_000.0 + 1e-6


# --- underlying dominance (underlying >= 2x + 3x) -------------------------
def test_enforce_underlying_dominance_trims_leverage_riskiest_first():
    st = state.PortfolioState(cash=0.0, high_water=100_000.0)
    # underlying 10k < leveraged (2x 5k + 3x 15k = 20k); excess 10k, capped at 10% of 30k.
    st.lots = [state.Lot("A", 1, 10_000.0, 10_000.0), state.Lot("A", 2, 5_000.0, 5_000.0),
               state.Lot("A", 3, 15_000.0, 15_000.0)]
    trades = rules.enforce_underlying_dominance(st)
    assert trades[0].tier == 3  # riskiest tier shed first
    assert abs(trades[0].amount_eur - 3_000.0) < 1e-6  # capped at 10%/day of total (30k)


def test_no_dominance_trim_in_base_case():
    st = state.build_base_case(100_000.0, stocks=["A", "B", "C", "D", "E"])
    # Base case is 9% underlying == 9% leveraged per name -> exactly on the boundary,
    # so dominance holds (excess == 0, no trim) until an up-move tips leverage ahead.
    assert rules.enforce_underlying_dominance(st) == []


def test_orders_below_min_trade_are_skipped():
    st = state.PortfolioState(cash=0.0, high_water=100_000.0)
    # underlying 10k vs 3x 10.3k -> excess 300 € (< MIN_TRADE_EUR): no micro-trim fires.
    st.lots = [state.Lot("A", 1, 10_000.0, 10_000.0),
               state.Lot("A", 3, 10_300.0, 10_300.0)]
    assert rules.enforce_underlying_dominance(st) == []  # 300 € order suppressed

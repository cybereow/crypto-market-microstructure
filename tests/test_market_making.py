import numpy as np
import pandas as pd

from src.market_making import compute_quotes, simulate_market_making


def test_compute_quotes_symmetric_around_reservation_mid():
    bid, ask = compute_quotes(prev_close=100.0, vol=2.0, inventory=0.0, obi_prev=0.5,
                               k_spread=1.5, k_inventory=0.0, k_obi=0.0)
    assert bid == 100.0 - 1.5 * 2.0
    assert ask == 100.0 + 1.5 * 2.0


def test_inventory_skew_pushes_reservation_price_down_when_long():
    """Positive inventory (net long) should shift BOTH quotes down: makes
    the bid less attractive (less likely to buy more) and the ask more
    attractive (more likely to sell off the position).
    """
    flat_bid, flat_ask = compute_quotes(100.0, 2.0, inventory=0.0, obi_prev=0.5,
                                         k_spread=1.0, k_inventory=0.5, k_obi=0.0)
    long_bid, long_ask = compute_quotes(100.0, 2.0, inventory=4.0, obi_prev=0.5,
                                         k_spread=1.0, k_inventory=0.5, k_obi=0.0)
    assert long_bid < flat_bid
    assert long_ask < flat_ask
    # Spread width itself is unaffected by inventory -- only the reservation
    # price moves.
    assert (long_ask - long_bid) == (flat_ask - flat_bid)


def test_obi_skew_pushes_reservation_price_up_on_buying_pressure():
    """obi > 0.5 (bid-heavy order flow, per obi_momentum_entries's
    convention) should shift both quotes UP -- less likely to get the ask
    lifted right before an up-move, more willing to buy ahead of it.
    """
    neutral_bid, neutral_ask = compute_quotes(100.0, 2.0, inventory=0.0, obi_prev=0.5,
                                               k_spread=1.0, k_inventory=0.0, k_obi=1.0)
    bullish_bid, bullish_ask = compute_quotes(100.0, 2.0, inventory=0.0, obi_prev=0.9,
                                               k_spread=1.0, k_inventory=0.0, k_obi=1.0)
    assert bullish_bid > neutral_bid
    assert bullish_ask > neutral_ask


def test_simulate_fills_both_sides_and_captures_spread():
    """Flat price with one wide-range bar: both quotes get touched, netting
    inventory back to zero but capturing the spread as PnL (less fees).
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="5s")
    df = pd.DataFrame({
        'close': [100.0, 100.0, 100.0, 100.0],
        'high':  [100.0, 100.0, 106.0, 100.0],
        'low':   [100.0, 100.0, 94.0, 100.0],
    }, index=dates)

    result = simulate_market_making(df, vol_lookback=1, k_spread=1.0, k_inventory=0.0,
                                     k_obi=0.0, max_inventory=5.0, fee_pct=0.0)

    # vol[1] = high[1]-low[1] = 0 (bar 1 is flat) -> quotes at bar 2 are
    # bid=ask=100 (zero spread, since bar 1's range was 0). Bar 2's own
    # wide range (94-106) doesn't affect bar 2's OWN quotes (no lookahead)
    # but DOES set vol for bar 3's quotes.
    assert result['n_bid_fills'] >= 1
    assert result['n_ask_fills'] >= 1
    # No captured spread on the zero-vol quote, no fees -> PnL should be
    # exactly zero for that round trip.
    assert result['total_pnl'] == 0.0


def test_simulate_captures_positive_spread_when_quotes_are_wide():
    """With a real (nonzero) prior-bar range setting the quote width, a
    round trip should capture the spread minus fees -- and MORE fee should
    mean LESS captured PnL, monotonically.
    """
    dates = pd.date_range("2024-01-01", periods=3, freq="5s")
    df = pd.DataFrame({
        'close': [100.0, 100.0, 100.0],
        # bar 1's range (4) sets bar 2's quote width (bid=96, ask=104 at
        # k_spread=1); bar 2's OWN much wider range (90-110) is what
        # actually touches both of those quotes.
        'high':  [100.0, 102.0, 110.0],
        'low':   [100.0, 98.0, 90.0],
    }, index=dates)

    no_fee = simulate_market_making(df, vol_lookback=1, k_spread=1.0, fee_pct=0.0)
    with_fee = simulate_market_making(df, vol_lookback=1, k_spread=1.0, fee_pct=0.001)

    assert no_fee['total_pnl'] > 0
    assert with_fee['total_pnl'] < no_fee['total_pnl']


def test_max_inventory_cap_stops_further_same_side_fills():
    """A long streak of bid-only fills (ask never touched) should stop
    adding inventory once max_inventory is reached, not run past it.
    """
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="5s")
    # Every bar's low dips to touch a generous bid; high never reaches the ask.
    df = pd.DataFrame({
        'close': [100.0] * n,
        'high': [100.0] * n,
        'low': [80.0] * n,
    }, index=dates)

    result = simulate_market_making(df, vol_lookback=1, k_spread=1.0, max_inventory=3.0, fee_pct=0.0)

    assert max(result['inventory']) <= 3.0
    assert result['n_ask_fills'] == 0

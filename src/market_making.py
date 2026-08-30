"""Two-sided market-making simulation -- a fundamentally different bet from
every other strategy in this repo.

Everything else here (breakout, reversion, vol_breakout, trend_pullback,
range_fade, obi_momentum) is DIRECTIONAL: pick a side, pay to cross or wait
to be filled, and profit only if price moves the way you bet. Sections 8-11
of the README all converge on the same wall: transaction cost is
consistently the same order of magnitude as, or larger than, any
directional edge this repo could measure.

Market making inverts the bet: quote BOTH sides at once and profit from the
spread you capture on round trips, largely independent of which way price
goes, at the cost of carrying inventory risk when one side fills more than
the other. The real top-of-book spread on BTC perpetuals turns out to be
far too thin to clear even a retail maker fee (see README section 12) --
median ~0.0002% of price, an order of magnitude below the 0.02% maker fee
this repo has assumed everywhere else. So this module does NOT join the
tightest quote; it prices a synthetic quote a multiple of recent realized
volatility away from mid (the standard practice for exactly this reason:
give up some fill rate to capture a spread that's actually worth the fee),
with two independent skews:

  - inventory skew: the more one-sided the current position, the more both
    quotes shift to lean the next fill back toward flat -- the classic
    inventory-risk control every real market maker needs, without which a
    trending market accumulates an ever-growing losing position on one side.
  - OBI skew (optional): shift both quotes in the direction the order-flow-
    imbalance signal (src.labeling.obi_momentum_entries's same `obi` input)
    says price is about to move, so the quote on the wrong side of an
    incoming move is less likely to be the one that fills.

Fills use the same OHLC-touch convention as `src.execution.simulate_maker_fills`
(a bar's low/high crossing the quote), with the same OPTIMISTIC-upper-bound
caveat: no real queue-position data, so this is a ceiling on real
performance, not a floor.
"""
import numpy as np
import pandas as pd


def compute_quotes(prev_close: float, vol: float, inventory: float, obi_prev: float,
                    k_spread: float, k_inventory: float, k_obi: float) -> tuple:
    """Price this bar's bid/ask from PREVIOUS-bar information only (no
    lookahead): a volatility-scaled half-spread around a reservation price
    that's been shifted off the raw mid by two independent skews.

    Inventory skew: `-k_inventory * inventory` on the reservation price.
    Positive inventory (net long) pushes the reservation price DOWN, which
    pulls both quotes down -- the bid gets less attractive (less likely to
    buy more) and the ask gets more attractive (more likely to sell down
    the position). This is the standard inventory-risk control; without it
    a run of fills on one side just keeps accumulating.

    OBI skew: `+k_obi * (obi_prev - 0.5)` on the reservation price. When
    OBI signals buying pressure (obi_prev > 0.5, per
    src.labeling.obi_momentum_entries's same convention), this pushes the
    reservation price UP -- both quotes rise, making the ask less likely to
    be lifted right before price moves up, and the bid more willing to buy
    ahead of it. Note the opposite sign convention from inventory skew:
    inventory skew reacts to OUR position, OBI skew reacts to the MARKET.

    Returns (bid_quote, ask_quote).
    """
    half_spread = k_spread * vol
    reservation_mid = prev_close - k_inventory * inventory + k_obi * (obi_prev - 0.5)
    return reservation_mid - half_spread, reservation_mid + half_spread


def simulate_market_making(df: pd.DataFrame, vol_lookback: int = 60, k_spread: float = 2.0,
                            k_inventory: float = 0.0, k_obi: float = 0.0,
                            max_inventory: float = 5.0, fee_pct: float = 0.0002,
                            obi_col: str = 'obi') -> dict:
    """Run the quoting/fill/inventory loop bar by bar.

    At each bar `t`, quotes are priced from information available at the
    CLOSE of bar `t-1` (previous close, previous realized vol, previous
    inventory, previous obi) and checked against bar `t`'s own high/low --
    no lookahead. `vol` is a simple range-based realized-volatility proxy
    (mean high-low over `vol_lookback` bars), not the ATR-with-gaps used
    elsewhere in this repo, to keep this module dependency-free of the
    feature pipeline; it plays the same role.

    `k_spread` sets the half-spread in vol multiples, `k_inventory` and
    `k_obi` are the two skews described in the module docstring (in price
    units per unit of inventory / per unit of obi-0.5), `max_inventory`
    caps how far one-sided the book is allowed to run (a side stops being
    quoted once its fill would breach the cap), and `fee_pct` is charged
    (or, if negative, rebated) on the notional of every fill.

    Returns a dict with the bar-level equity curve (`equity`, `inventory`,
    `cash`), the fill log (`fills`: list of {bar, side, price}), and
    summary stats (`n_bid_fills`, `n_ask_fills`, `total_pnl`,
    `avg_captured_spread`, `max_drawdown`).
    """
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()
    obi = df[obi_col].to_numpy() if obi_col in df.columns else np.full(len(df), 0.5)
    n = len(df)

    vol = pd.Series(highs - lows).rolling(vol_lookback).mean().to_numpy()

    cash = 0.0
    inventory = 0.0
    equity_curve = np.zeros(n)
    inventory_curve = np.zeros(n)
    fills = []
    bid_fill_prices, ask_fill_prices = [], []

    for t in range(n):
        if t == 0 or np.isnan(vol[t - 1]):
            equity_curve[t] = cash + inventory * closes[t]
            inventory_curve[t] = inventory
            continue

        bid_quote, ask_quote = compute_quotes(closes[t - 1], vol[t - 1], inventory, obi[t - 1],
                                               k_spread, k_inventory, k_obi)

        if inventory < max_inventory and lows[t] <= bid_quote:
            inventory += 1.0
            cash -= bid_quote * (1.0 + fee_pct)
            fills.append({'bar': t, 'side': 'bid', 'price': bid_quote})
            bid_fill_prices.append(bid_quote)

        if inventory > -max_inventory and highs[t] >= ask_quote:
            inventory -= 1.0
            cash += ask_quote * (1.0 - fee_pct)
            fills.append({'bar': t, 'side': 'ask', 'price': ask_quote})
            ask_fill_prices.append(ask_quote)

        equity_curve[t] = cash + inventory * closes[t]
        inventory_curve[t] = inventory

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max
    n_round_trips = min(len(bid_fill_prices), len(ask_fill_prices))
    avg_captured_spread = (float(np.mean(ask_fill_prices[:n_round_trips]) -
                                  np.mean(bid_fill_prices[:n_round_trips]))
                            if n_round_trips else float('nan'))

    return {
        'equity': equity_curve, 'inventory': inventory_curve, 'fills': fills,
        'n_bid_fills': len(bid_fill_prices), 'n_ask_fills': len(ask_fill_prices),
        'total_pnl': float(equity_curve[-1]), 'avg_captured_spread': avg_captured_spread,
        'max_drawdown': float(drawdown.min()),
    }

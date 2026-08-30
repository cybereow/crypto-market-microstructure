"""Maker-fill (queue) simulation for the primary signal.

Every cost figure in this repo up to this point (see README section 8) is a
FLAT assumption: every candidate trade pays `cost_per_trade` and is entered
at the signal bar's close, as if a market order always fills instantly at
that price. That is true for a taker order. It is not true for a maker
(limit) order, which is the only way to actually reach the ~0.08% round-trip
cost the edge needs to survive (see README): a resting limit order only
earns the maker fee if the market trades back to it, and it may not — a
breakout that keeps going away from the entry level never fills the passive
order at all.

This module replaces the "instant fill" assumption with an explicit,
OHLC-only queue simulation:

  1. At the signal bar, price a passive limit order a small ATR-multiple
     BETTER than the signal price, in the maker direction (a long buys
     below the signal close, a short sells above it) — the price the
     market must trade back to for the order to earn the maker fee instead
     of paying to cross the spread.
  2. Wait up to `queue_timeout` bars for a subsequent bar's high/low to
     touch that price. If it never touches, the order is cancelled: no
     position was ever opened, so there is no trade to count.
  3. If it fills, re-run the SAME triple-barrier logic used everywhere else
     in this repo (`src.labeling.scan_triple_barrier`), anchored at the
     fill bar and fill price instead of the signal bar and its close.

Caveat this cannot remove: OHLC bars carry no true book/queue-position data.
"The bar's low touched the limit price" is an OPTIMISTIC fill assumption —
it ignores how much size was already resting ahead of this order at that
price level. A real limit order can see its price touched and still not
fill if the queue ahead of it absorbs the whole move. This simulation can
therefore only ever produce an upper bound on the real maker fill rate; if
the strategy's edge does not survive even this optimistic assumption, it
certainly will not survive a real order book.
"""
import numpy as np
import pandas as pd

from src.labeling import scan_triple_barrier


def simulate_maker_fills(df: pd.DataFrame, entries: pd.Series, atr: pd.Series,
                          offset_mult: float = 0.15, queue_timeout: int = 3) -> pd.DataFrame:
    """For each non-zero entry, simulate resting a passive limit order
    instead of assuming an instant market fill at the signal bar's close.

    `offset_mult` is how much better than the signal price the resting
    order is priced, in ATR multiples of the signal bar (0.15 is a small
    fraction of a bar's typical range — a near-touch, not a deep limit that
    would trivially never fill). `queue_timeout` bars is how long the order
    stays resting before being cancelled.

    Returns a DataFrame indexed like `entries` (rows where entries == 0 are
    dropped) with columns: side, signal_pos (bar position of the signal),
    signal_price, limit_price, filled (bool), fill_price (NaN if unfilled),
    wait_bars (bars from signal to fill, NaN if unfilled).
    """
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()
    atr_vals = atr.to_numpy()
    entry_vals = entries.to_numpy()
    n = len(df)

    rows = []
    idx = []
    for i in range(n):
        side = entry_vals[i]
        if side == 0 or np.isnan(atr_vals[i]):
            continue

        signal_price = closes[i]
        offset = offset_mult * atr_vals[i]
        # Better than market in the maker direction: a long's bid sits
        # BELOW the signal price, a short's ask sits ABOVE it.
        limit_price = signal_price - side * offset

        filled = False
        fill_price = np.nan
        wait_bars = np.nan
        end = min(i + queue_timeout, n - 1)
        for j in range(i + 1, end + 1):
            touched = lows[j] <= limit_price if side > 0 else highs[j] >= limit_price
            if touched:
                filled = True
                fill_price = limit_price
                wait_bars = j - i
                break

        rows.append({'side': side, 'signal_pos': i, 'signal_price': signal_price,
                     'limit_price': limit_price, 'filled': filled,
                     'fill_price': fill_price, 'wait_bars': wait_bars})
        idx.append(df.index[i])

    return pd.DataFrame(rows, index=pd.Index(idx, name=df.index.name))


def triple_barrier_from_fill(df: pd.DataFrame, fills: pd.DataFrame, atr: pd.Series,
                              pt_mult: float, sl_mult: float, max_holding: int) -> pd.DataFrame:
    """Label only the FILLED orders from `simulate_maker_fills`, anchored at
    the actual fill bar/price rather than the signal bar/price. Unfilled
    orders are excluded — no position was ever opened, so there is no
    outcome to label.

    Uses `atr` at the SIGNAL bar (not the fill bar) for the barrier
    distances, matching `triple_barrier_labels`' convention that the
    position's risk is sized off the volatility observed when the setup was
    identified.

    Returns a DataFrame with the same columns as `triple_barrier_labels`
    (side, label, ret, hold, entry_pos, exit_pos), plus `wait_bars`.
    """
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()
    atr_vals = atr.to_numpy()

    filled = fills[fills['filled']]
    rows = []
    idx = []
    for ts, row in filled.iterrows():
        side = row['side']
        signal_pos = int(row['signal_pos'])
        fill_pos = signal_pos + int(row['wait_bars'])
        entry_price = row['fill_price']
        pt_dist = pt_mult * atr_vals[signal_pos]
        sl_dist = sl_mult * atr_vals[signal_pos]

        outcome = scan_triple_barrier(highs, lows, closes, fill_pos, side, entry_price,
                                       pt_dist, sl_dist, max_holding)
        ret = side * (outcome['exit_price'] / entry_price - 1)
        rows.append({'side': side, 'label': outcome['label'], 'ret': ret,
                     'hold': outcome['hold'], 'wait_bars': row['wait_bars'],
                     'entry_pos': fill_pos, 'exit_pos': fill_pos + outcome['hold']})
        idx.append(ts)

    return pd.DataFrame(rows, index=pd.Index(idx, name=fills.index.name))

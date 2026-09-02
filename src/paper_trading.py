"""Shadow paper-testing for the maker-fill hypothesis (README section 9):
watch REAL live quotes for the `vol_breakout` signal's candidate trades and
record whether a resting limit order would actually have filled, and how
it would have resolved -- without ever placing a real order or risking
real capital.

Why this exists, and how it differs from `src.execution.simulate_maker_fills`:
that module's fill rule is "did a historical 4h candle's high/low touch the
limit price", which is an OPTIMISTIC upper bound with no real queue-position
information. This module still can't see the real order book queue either,
but it checks against actually-observed LIVE trade prices polled between
candle closes, which is a strictly finer-grained (though still imperfect --
polling, not streaming) approximation than a single candle's high/low. The
gap between what section 9's backtest predicted and what this records is
itself evidence about how much the OHLC approximation was worth.

All decision logic lives here as small, pure, testable functions. The
orchestrating script (scripts/paper_test_live.py) owns every ccxt/network
call and the on-disk state file, and calls into these.
"""
import numpy as np
import pandas as pd

STATE_COLUMNS = [
    'id', 'asset', 'side', 'signal_time', 'signal_price', 'atr',
    'limit_price', 'pt_price', 'sl_price', 'deadline', 'status',
    'fill_time', 'fill_price', 'vertical_deadline', 'close_time',
    'close_price', 'ret',
]

# Same defaults as scripts/backtest_maker_fill.py's vol_breakout run, so the
# live numbers land on the same axis as the historical ones in README
# section 9 -- this is a check on that result, not a new experiment.
LOOKBACK = 20
OFFSET_MULT = 0.15
PT_MULT = 2.0
SL_MULT = 1.0
QUEUE_TIMEOUT_BARS = 3
MAX_HOLDING_BARS = 18
BAR_HOURS = 4


def new_state() -> pd.DataFrame:
    return pd.DataFrame(columns=STATE_COLUMNS)


def make_pending_order(asset: str, side: int, signal_time: pd.Timestamp,
                        signal_price: float, atr: float,
                        offset_mult: float = OFFSET_MULT, pt_mult: float = PT_MULT,
                        sl_mult: float = SL_MULT) -> dict:
    """A new candidate trade just fired on a freshly-closed candle. Price
    the resting limit the same way simulate_maker_fills does: `offset_mult`
    ATR better than the signal price, in the maker direction.

    `offset_mult`/`pt_mult`/`sl_mult` default to this module's own
    vol_breakout-tuned constants (scripts/paper_test_live.py's signal);
    pass a different signal's own tuned geometry explicitly (e.g.
    scripts/paper_test_funding_live.py's pt_mult=2.0, sl_mult=2.0,
    matching scripts/backtest_funding_reversion.py) rather than editing
    these module constants, which stay vol_breakout's own defaults.
    """
    offset = offset_mult * atr
    limit_price = signal_price - side * offset
    pt_dist = pt_mult * atr
    sl_dist = sl_mult * atr
    if side > 0:
        pt_price, sl_price = limit_price + pt_dist, limit_price - sl_dist
    else:
        pt_price, sl_price = limit_price - pt_dist, limit_price + sl_dist

    return {
        'id': f"{asset}_{signal_time.isoformat()}",
        'asset': asset, 'side': side, 'signal_time': signal_time,
        'signal_price': signal_price, 'atr': atr, 'limit_price': limit_price,
        'pt_price': pt_price, 'sl_price': sl_price,
        'deadline': signal_time + pd.Timedelta(hours=BAR_HOURS * QUEUE_TIMEOUT_BARS),
        'status': 'pending_fill', 'fill_time': pd.NaT, 'fill_price': np.nan,
        'vertical_deadline': pd.NaT, 'close_time': pd.NaT, 'close_price': np.nan,
        'ret': np.nan,
    }


def check_fill(side: int, limit_price: float, last_price: float) -> bool:
    """Would a resting limit at `limit_price` have been touched by a trade
    at `last_price`? Long rests below market (fills on a dip to/through
    it); short rests above (fills on a rally to/through it).
    """
    return last_price <= limit_price if side > 0 else last_price >= limit_price


def check_barrier(side: int, pt_price: float, sl_price: float, last_price: float) -> str:
    """Which barrier (if any) a live trade price has touched. Returns
    'pt', 'sl', or None. Matches src.labeling.scan_triple_barrier's
    convention: for a long, pt is above entry and sl is below (and the
    reverse for a short).
    """
    if side > 0:
        if last_price >= pt_price:
            return 'pt'
        if last_price <= sl_price:
            return 'sl'
    else:
        if last_price <= pt_price:
            return 'pt'
        if last_price >= sl_price:
            return 'sl'
    return None


def step_pending_order(row: dict, last_price: float, now: pd.Timestamp) -> dict:
    """Advance one 'pending_fill' order given the current live price.
    Returns the (possibly updated) row; does not mutate the input.
    """
    row = dict(row)
    if check_fill(row['side'], row['limit_price'], last_price):
        row['status'] = 'filled'
        row['fill_time'] = now
        row['fill_price'] = row['limit_price']
        row['vertical_deadline'] = now + pd.Timedelta(hours=BAR_HOURS * MAX_HOLDING_BARS)
    elif now >= row['deadline']:
        row['status'] = 'expired_unfilled'
    return row


def step_open_order(row: dict, last_price: float, now: pd.Timestamp) -> dict:
    """Advance one 'filled' (open) order given the current live price."""
    row = dict(row)
    hit = check_barrier(row['side'], row['pt_price'], row['sl_price'], last_price)
    if hit == 'pt':
        row['status'] = 'closed_win'
        row['close_time'], row['close_price'] = now, last_price
        row['ret'] = row['side'] * (last_price / row['fill_price'] - 1)
    elif hit == 'sl':
        row['status'] = 'closed_loss'
        row['close_time'], row['close_price'] = now, last_price
        row['ret'] = row['side'] * (last_price / row['fill_price'] - 1)
    elif now >= row['vertical_deadline']:
        row['status'] = 'closed_vertical'
        row['close_time'], row['close_price'] = now, last_price
        row['ret'] = row['side'] * (last_price / row['fill_price'] - 1)
    return row

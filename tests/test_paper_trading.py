import pandas as pd

from src.paper_trading import (make_pending_order, check_fill, check_barrier,
                               step_pending_order, step_open_order)


def test_make_pending_order_long_prices_limit_below_and_barriers_correctly():
    t = pd.Timestamp("2026-01-01 00:00:00")
    order = make_pending_order('BTC/USDT', 1, t, signal_price=100.0, atr=10.0)

    assert order['limit_price'] == 100.0 - 0.15 * 10.0  # 98.5, below signal
    assert order['pt_price'] == order['limit_price'] + 2.0 * 10.0
    assert order['sl_price'] == order['limit_price'] - 1.0 * 10.0
    assert order['status'] == 'pending_fill'
    assert order['deadline'] == t + pd.Timedelta(hours=12)  # 3 bars * 4h


def test_make_pending_order_short_prices_limit_above_and_barriers_correctly():
    t = pd.Timestamp("2026-01-01 00:00:00")
    order = make_pending_order('BTC/USDT', -1, t, signal_price=100.0, atr=10.0)

    assert order['limit_price'] == 100.0 + 0.15 * 10.0  # 101.5, above signal
    assert order['pt_price'] == order['limit_price'] - 2.0 * 10.0
    assert order['sl_price'] == order['limit_price'] + 1.0 * 10.0


def test_make_pending_order_accepts_a_different_signals_own_geometry():
    """scripts/paper_test_funding_live.py needs pt_mult=2.0/sl_mult=2.0
    (symmetric, matching backtest_funding_reversion.py) instead of
    vol_breakout's own 2.0/1.0 module defaults -- overriding must not
    require editing the module constants other signals still rely on.
    """
    t = pd.Timestamp("2026-01-01 00:00:00")
    order = make_pending_order('BTC/USDT', 1, t, signal_price=100.0, atr=10.0,
                               offset_mult=0.2, pt_mult=2.0, sl_mult=2.0)

    assert order['limit_price'] == 100.0 - 0.2 * 10.0
    assert order['pt_price'] == order['limit_price'] + 2.0 * 10.0
    assert order['sl_price'] == order['limit_price'] - 2.0 * 10.0


def test_check_fill_direction():
    assert check_fill(1, limit_price=99.0, last_price=98.5) is True
    assert check_fill(1, limit_price=99.0, last_price=99.5) is False
    assert check_fill(-1, limit_price=101.0, last_price=101.5) is True
    assert check_fill(-1, limit_price=101.0, last_price=100.5) is False


def test_check_barrier_direction():
    assert check_barrier(1, pt_price=110.0, sl_price=95.0, last_price=111.0) == 'pt'
    assert check_barrier(1, pt_price=110.0, sl_price=95.0, last_price=94.0) == 'sl'
    assert check_barrier(1, pt_price=110.0, sl_price=95.0, last_price=100.0) is None
    assert check_barrier(-1, pt_price=90.0, sl_price=105.0, last_price=89.0) == 'pt'
    assert check_barrier(-1, pt_price=90.0, sl_price=105.0, last_price=106.0) == 'sl'


def test_step_pending_order_fills_when_price_touches_limit():
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    order = make_pending_order('BTC/USDT', 1, t0, signal_price=100.0, atr=10.0)

    now = t0 + pd.Timedelta(hours=1)
    updated = step_pending_order(order, last_price=98.0, now=now)  # <= limit 98.5

    assert updated['status'] == 'filled'
    assert updated['fill_time'] == now
    assert updated['fill_price'] == order['limit_price']
    assert updated['vertical_deadline'] == now + pd.Timedelta(hours=72)  # 18 bars * 4h


def test_step_pending_order_expires_when_deadline_passes_without_fill():
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    order = make_pending_order('BTC/USDT', 1, t0, signal_price=100.0, atr=10.0)

    now = order['deadline'] + pd.Timedelta(minutes=1)
    updated = step_pending_order(order, last_price=99.9, now=now)  # never touched limit

    assert updated['status'] == 'expired_unfilled'


def test_step_pending_order_stays_pending_when_neither_condition_holds():
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    order = make_pending_order('BTC/USDT', 1, t0, signal_price=100.0, atr=10.0)

    now = t0 + pd.Timedelta(hours=1)
    updated = step_pending_order(order, last_price=99.9, now=now)

    assert updated['status'] == 'pending_fill'


def test_step_open_order_closes_win_on_profit_target():
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    order = make_pending_order('BTC/USDT', 1, t0, signal_price=100.0, atr=10.0)
    filled = step_pending_order(order, last_price=98.0, now=t0 + pd.Timedelta(hours=1))

    now = t0 + pd.Timedelta(hours=2)
    closed = step_open_order(filled, last_price=filled['pt_price'] + 1, now=now)

    assert closed['status'] == 'closed_win'
    assert closed['ret'] > 0


def test_step_open_order_closes_loss_on_stop():
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    order = make_pending_order('BTC/USDT', 1, t0, signal_price=100.0, atr=10.0)
    filled = step_pending_order(order, last_price=98.0, now=t0 + pd.Timedelta(hours=1))

    now = t0 + pd.Timedelta(hours=2)
    closed = step_open_order(filled, last_price=filled['sl_price'] - 1, now=now)

    assert closed['status'] == 'closed_loss'
    assert closed['ret'] < 0


def test_step_open_order_closes_vertical_after_max_holding():
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    order = make_pending_order('BTC/USDT', 1, t0, signal_price=100.0, atr=10.0)
    filled = step_pending_order(order, last_price=98.0, now=t0 + pd.Timedelta(hours=1))

    now = filled['vertical_deadline'] + pd.Timedelta(minutes=1)
    mid_price = (filled['pt_price'] + filled['sl_price']) / 2  # neither barrier touched
    closed = step_open_order(filled, last_price=mid_price, now=now)

    assert closed['status'] == 'closed_vertical'
    assert closed['close_price'] == mid_price

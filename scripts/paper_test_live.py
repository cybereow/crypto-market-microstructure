"""Shadow paper-test runner: detects the `vol_breakout` signal on freshly
closed 4h candles, prices a resting maker limit exactly like
`src.execution.simulate_maker_fills`, and polls REAL live quotes to see
whether it fills and how it resolves. No order is ever sent -- see
`src/paper_trading.py` for why this is a strictly finer-grained check on
README section 9's maker-fill hypothesis than the OHLC-based backtest.

Safe to run repeatedly (e.g. from a Routine): state lives in
data/paper_trades.csv and every step is idempotent -- a signal already
logged for a given (asset, candle) is never logged twice, and each order
only ever advances forward through pending_fill -> filled -> closed_*.

Venue: Kraken, not Binance. Binance's REST API returns HTTP 451 from this
environment's network egress (the same restriction documented in the
README for scripts/download_data.py), so this reuses Kraken exactly as
download_data.py already does for the same reason. Kraken/Binance prices
for these pairs are highly correlated but not identical -- a disclosed
approximation, not a silent one.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from src.labeling import volatility_breakout_entries
from src.metrics import net_pf_expectancy
from src.paper_trading import (STATE_COLUMNS, LOOKBACK, make_pending_order,
                               step_pending_order, step_open_order)

ASSETS = ['ETH/USDT', 'SOL/USDT', 'LINK/USDT', 'AVAX/USDT', 'DOT/USDT',
          'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'LTC/USDT', 'ATOM/USDT']

STATE_PATH = os.path.join(OUTPUT_DIR, 'paper_trades.csv')

DATETIME_COLS = ['signal_time', 'deadline', 'fill_time', 'vertical_deadline', 'close_time']


def make_exchange() -> ccxt.Exchange:
    """Some sandboxed dev environments (including the one this was first
    run in) route outbound HTTPS through a policy proxy whose CA ccxt's own
    requests.Session does not pick up by default (ccxt disables
    `trust_env` on its session precisely to avoid surprises from the
    ambient environment). Detected by the CA bundle path such setups
    document (/root/.ccr/README.md): never disable TLS verification --
    point requests at that bundle and let it read HTTPS_PROXY, same as any
    other library is expected to opt in to that environment's trust. A
    normal machine won't have this path, so this is a no-op there.
    """
    ca_bundle = '/root/.ccr/ca-bundle.crt'
    if os.path.exists(ca_bundle):
        os.environ.setdefault('REQUESTS_CA_BUNDLE', ca_bundle)

    exchange = ccxt.kraken({'enableRateLimit': True})
    exchange.session.trust_env = True
    return exchange


def load_state() -> pd.DataFrame:
    if not os.path.exists(STATE_PATH):
        return pd.DataFrame(columns=STATE_COLUMNS)
    df = pd.read_csv(STATE_PATH)
    for col in DATETIME_COLS:
        df[col] = pd.to_datetime(df[col])
    return df


def save_state(state: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    state.to_csv(STATE_PATH, index=False)


def detect_new_signal(exchange: ccxt.Exchange, asset: str, now: pd.Timestamp) -> dict:
    """Fetch recent 4h candles and check whether the LAST FULLY CLOSED one
    just fired a vol_breakout entry. Returns a new pending-order dict, or
    None if there's no fresh signal (or not enough history yet).
    """
    ohlcv = exchange.fetch_ohlcv(asset, '4h', limit=300)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df.set_index('timestamp', inplace=True)

    # Exchanges commonly include the still-forming current candle as the
    # last row; only the previous, fully-closed one is a valid signal bar.
    if df.index[-1] + pd.Timedelta(hours=4) > now:
        df = df.iloc[:-1]
    if len(df) < 150:  # not enough history for the rolling(100) squeeze rank
        return None

    df_features = create_features(df)
    raw_atr = df_features['ATR_14'] * df_features['close']
    entries = volatility_breakout_entries(df_features, lookback=LOOKBACK)

    side = int(entries.iloc[-1])
    atr = raw_atr.iloc[-1]
    if side == 0 or np.isnan(atr):
        return None

    signal_time = df_features.index[-1]
    signal_price = float(df_features['close'].iloc[-1])
    return make_pending_order(asset, side, signal_time, signal_price, float(atr))


def main():
    parser = argparse.ArgumentParser(
        description="Shadow paper-test the vol_breakout maker-fill hypothesis against live quotes.")
    parser.add_argument("--assets", type=str, nargs='+', default=ASSETS)
    args = parser.parse_args()

    exchange = make_exchange()
    state = load_state()
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)

    new_rows = []
    for asset in args.assets:
        try:
            ticker = exchange.fetch_ticker(asset)
            last_price = float(ticker['last'])
        except Exception as exc:
            print(f"  {asset}: ticker fetch failed ({exc}), skipping this run.")
            continue

        try:
            candidate = detect_new_signal(exchange, asset, now)
        except Exception as exc:
            print(f"  {asset}: signal detection failed ({exc}), skipping new-signal check.")
            candidate = None
        if candidate is not None:
            already_logged = ((state['asset'] == asset) &
                               (state['signal_time'] == candidate['signal_time'])).any()
            if not already_logged:
                new_rows.append(candidate)
                print(f"  {asset}: NEW signal side={candidate['side']:+d} "
                      f"at {candidate['signal_price']:.6g}, limit {candidate['limit_price']:.6g}")

        mask = state['asset'] == asset
        for i in state[mask].index:
            row = state.loc[i].to_dict()
            if row['status'] == 'pending_fill':
                state.loc[i] = step_pending_order(row, last_price, now)
            elif row['status'] == 'filled':
                state.loc[i] = step_open_order(row, last_price, now)

    if new_rows:
        state = pd.concat([state, pd.DataFrame(new_rows)], ignore_index=True)
    save_state(state)

    pending = (state['status'] == 'pending_fill').sum()
    open_ = (state['status'] == 'filled').sum()
    expired = (state['status'] == 'expired_unfilled').sum()
    closed = state[state['status'].isin(['closed_win', 'closed_loss', 'closed_vertical'])]

    print(f"\nState: {pending} pending fill, {open_} open, {expired} expired unfilled, "
          f"{len(closed)} closed.")
    if len(closed):
        rets = closed['ret'].to_numpy()
        win_rate = (closed['status'] == 'closed_win').mean()
        taker_pf, taker_exp = net_pf_expectancy(rets, 0.004)
        maker_pf, maker_exp = net_pf_expectancy(rets, 0.0008)
        print(f"  closed trades: win rate {win_rate:.1%}")
        print(f"  TAKER cost 0.40%: PF {taker_pf:.2f}, exp/trade {taker_exp:+.4%}")
        print(f"  MAKER cost 0.08%: PF {maker_pf:.2f}, exp/trade {maker_exp:+.4%}")
    if (expired + open_ + len(closed)) > 0:
        fill_rate = (open_ + len(closed)) / (expired + open_ + len(closed))
        print(f"  observed fill rate so far: {fill_rate:.1%} "
              f"({open_ + len(closed)}/{expired + open_ + len(closed)})")


if __name__ == "__main__":
    main()

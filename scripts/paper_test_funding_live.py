"""Live shadow paper-test for the funding-rate-extreme-reversion signal,
regime-filtered variant (README sections 14 and 20 -- the strongest
result in this repo's alt-data line: pooled maker PF 1.15, exp +0.26%,
p=0.013, p_deflated=0.0385 for the 3 configurations tried; still only
the OPTIMISTIC OHLC-based maker-fill upper bound, section 9's caveat).

Detects `src.labeling.funding_reversion_regime_filtered_entries` firing
on freshly-closed 4h candles, prices a resting maker limit exactly like
`src.execution.simulate_maker_fills`, and polls REAL live quotes to see
whether it fills and how it resolves -- same "no order is ever sent,
zero capital at risk" guarantee as scripts/paper_test_live.py, and same
reason this is a strictly finer-grained check on the historical
maker-fill hypothesis than an OHLC backtest (see src/paper_trading.py).

Venue: Kraken Futures (`ccxt.krakenfutures`), not Binance -- Binance's
futures API returns HTTP 451 from this environment's network egress
(same restriction documented for scripts/download_data.py and
scripts/paper_test_live.py), and Binance's own perpetual funding rate is
what section 14-20's backtests were built on, so Kraken Futures is a
disclosed, not silent, approximation on TWO axes at once here: Kraken
prices track Binance prices closely but not exactly (as already noted
for paper_test_live.py), AND Kraken accrues funding roughly hourly at a
much smaller per-observation magnitude than Binance's nominal 8h rate
(confirmed directly: Kraken's live funding readings are ~1e-5 to 1e-6 vs
Binance's typical ~1e-4 to 1e-2). This does not break the signal's
mechanism -- `funding_extreme_reversion_entries` only ever compares
funding against its OWN rolling quantile, never an absolute magnitude --
but it does mean the logged funding_rate values here are on a different
scale than the historical Binance-based backtest data, and is exactly
why this script exists as an independent live check rather than a
guaranteed replay of the same numbers.

Safe to run repeatedly (e.g. from a Routine): state lives in
data/paper_trades_funding.csv and every step is idempotent -- a signal
already logged for a given (asset, candle) is never logged twice, and
each order only ever advances forward through
pending_fill -> filled -> closed_*.
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
from src.labeling import funding_reversion_regime_filtered_entries
from src.metrics import net_pf_expectancy
from src.paper_trading import (STATE_COLUMNS, make_pending_order,
                               step_pending_order, step_open_order)

ASSETS = ['BTC/USD:USD', 'ETH/USD:USD', 'SOL/USD:USD']

# Matches scripts/backtest_funding_reversion.py's defaults for
# --signal funding_reversion_regime_filtered, so the live numbers land
# on the same axis as sections 14/20's historical ones.
LOOKBACK = 90
QUANTILE = 0.90
OFFSET_MULT = 0.15
PT_MULT = 2.0
SL_MULT = 2.0

STATE_PATH = os.path.join(OUTPUT_DIR, 'paper_trades_funding.csv')

DATETIME_COLS = ['signal_time', 'deadline', 'fill_time', 'vertical_deadline', 'close_time']


def make_exchange() -> ccxt.Exchange:
    """See scripts/paper_test_live.py's make_exchange for why this CA
    bundle detection exists -- same sandboxed-environment proxy handling,
    a no-op on a normal machine.
    """
    ca_bundle = '/root/.ccr/ca-bundle.crt'
    if os.path.exists(ca_bundle):
        os.environ.setdefault('REQUESTS_CA_BUNDLE', ca_bundle)

    exchange = ccxt.krakenfutures({'enableRateLimit': True})
    exchange.session.trust_env = True
    return exchange


def load_state() -> pd.DataFrame:
    if not os.path.exists(STATE_PATH):
        return pd.DataFrame(columns=STATE_COLUMNS)
    df = pd.read_csv(STATE_PATH)
    for col in DATETIME_COLS:
        # See scripts/paper_test_live.py's load_state for why this forces
        # a fixed (ns) resolution rather than whatever pandas infers.
        df[col] = pd.to_datetime(df[col]).astype('datetime64[ns]')
    return df


def save_state(state: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    state.to_csv(STATE_PATH, index=False)


def fetch_funding_series(exchange: ccxt.Exchange, asset: str, limit: int = 1000) -> pd.Series:
    """Hourly funding-rate history, resampled/forward-filled to a
    gapless hourly series -- the same shape scripts/download_funding_vision.py
    produces for the historical backtests, so joining it onto 4h OHLCV
    by exact timestamp works the same way (every 4h bar's timestamp is
    also an hourly-series timestamp).
    """
    history = exchange.fetch_funding_rate_history(asset, limit=limit)
    if not history:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([h['timestamp'] for h in history], unit='ms', utc=True).tz_localize(None)
    rates = pd.Series([h['fundingRate'] for h in history], index=idx).sort_index()
    rates = rates[~rates.index.duplicated(keep='last')]
    return rates.resample('h').ffill()


def detect_new_signal(exchange: ccxt.Exchange, asset: str, now: pd.Timestamp) -> tuple:
    """Fetch recent 4h candles + funding history and check whether the
    LAST FULLY CLOSED candle just fired a funding_reversion_regime_filtered
    entry. Returns (candidate_order_dict, features_dict), or (None, None)
    if there's no fresh signal (or not enough history yet).
    """
    ohlcv = exchange.fetch_ohlcv(asset, '4h', limit=300)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df.set_index('timestamp', inplace=True)

    if df.index[-1] + pd.Timedelta(hours=4) > now:
        df = df.iloc[:-1]
    if len(df) < 150:  # not enough history for the rolling(90) funding quantile plus warm-up
        return None, None

    funding = fetch_funding_series(exchange, asset)
    if funding.empty:
        return None, None
    df = df.join(funding.rename('funding_rate'), how='left')
    df['funding_rate'] = df['funding_rate'].ffill()
    if df['funding_rate'].isna().all():
        return None, None

    df_features = create_features(df)
    raw_atr = df_features['ATR_14'] * df_features['close']
    entries = funding_reversion_regime_filtered_entries(df_features, lookback=LOOKBACK,
                                                         quantile=QUANTILE)

    side = int(entries.iloc[-1])
    atr = raw_atr.iloc[-1]
    if side == 0 or np.isnan(atr):
        return None, None

    signal_time = df_features.index[-1]
    signal_price = float(df_features['close'].iloc[-1])
    order = make_pending_order(asset, side, signal_time, signal_price, float(atr),
                               offset_mult=OFFSET_MULT, pt_mult=PT_MULT, sl_mult=SL_MULT)

    last_row = df_features.iloc[-1]
    features = {'funding_rate': float(last_row['funding_rate']) if pd.notna(last_row['funding_rate']) else None,
                'ATR_ratio': float(last_row['ATR_ratio']) if pd.notna(last_row['ATR_ratio']) else None}
    return order, features


def main():
    parser = argparse.ArgumentParser(
        description="Live shadow paper-test for the regime-filtered funding-reversion signal "
                    "(README sections 14, 20). No order is ever sent; zero capital at risk.")
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
            candidate, features = detect_new_signal(exchange, asset, now)
        except Exception as exc:
            print(f"  {asset}: signal detection failed ({exc}), skipping new-signal check.")
            candidate = None

        if candidate is not None:
            already_logged = ((state['asset'] == asset) &
                               (state['signal_time'] == candidate['signal_time'])).any()
            if not already_logged:
                new_rows.append(candidate)
                print(f"  {asset}: NEW signal side={candidate['side']:+d} "
                      f"at {candidate['signal_price']:.6g}, limit {candidate['limit_price']:.6g} "
                      f"(funding_rate={features['funding_rate']}, ATR_ratio={features['ATR_ratio']})")

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
        print(f"  (section 20 historical reference: maker PF 1.15, exp/trade +0.26%, n=1287)")
    if (expired + open_ + len(closed)) > 0:
        fill_rate = (open_ + len(closed)) / (expired + open_ + len(closed))
        print(f"  observed fill rate so far: {fill_rate:.1%} "
              f"({open_ + len(closed)}/{expired + open_ + len(closed)})")


if __name__ == "__main__":
    main()

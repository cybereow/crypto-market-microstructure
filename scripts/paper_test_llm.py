"""LLM-gated shadow paper-test runner: same `vol_breakout` candidate
detection and live-quote polling as `scripts/paper_test_live.py`, but every
candidate must also be approved by an LLM (Claude, via the Messages API)
before it becomes a paper trade. Restricted by default to BTC, ETH, SOL.

This is a research probe, not a live-execution bot -- exactly like
`paper_test_live.py`, no order is ever sent and no capital is ever at
risk. See `src/llm_decision.py` for why an LLM gate can't be held to the
same statistical standard (`src/significance.py`) as the rest of this
repo, and is offered here only as something to test empirically.

Every candidate signal is logged either way (see the `llm_decision`,
`llm_confidence`, `llm_reason` columns), not just the approved ones --
rejected candidates get `status='llm_rejected'` and are never turned into
paper trades, but stay in the state file so the gate's effect (would this
candidate have won or lost, had it been taken anyway?) can be compared
against `data/paper_trades.csv` (the ungated shadow trader) later.

Safe to run repeatedly (e.g. from a Routine): state lives in
data/paper_trades_llm.csv and every step is idempotent -- a signal already
logged for a given (asset, candle) is never re-sent to the LLM, and each
approved order only ever advances forward through
pending_fill -> filled -> closed_*.

Needs ANTHROPIC_API_KEY in the environment. Venue is Kraken, for the same
reason documented in paper_test_live.py (Binance's REST API returns HTTP
451 from this environment's network egress).
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import anthropic
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from scripts.paper_test_live import make_exchange
from src.labeling import volatility_breakout_entries
from src.metrics import net_pf_expectancy
from src.paper_trading import (STATE_COLUMNS as BASE_STATE_COLUMNS, LOOKBACK,
                               make_pending_order, step_pending_order, step_open_order)
from src.llm_decision import FEATURE_KEYS, get_llm_decision

ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
MODEL = 'claude-sonnet-5'

STATE_COLUMNS = BASE_STATE_COLUMNS + ['llm_decision', 'llm_confidence', 'llm_reason']

STATE_PATH = os.path.join(OUTPUT_DIR, 'paper_trades_llm.csv')

DATETIME_COLS = ['signal_time', 'deadline', 'fill_time', 'vertical_deadline', 'close_time']


def make_llm_client() -> anthropic.Anthropic:
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. This script calls the real Claude "
            "API to gate candidate trades (no capital is ever risked, but "
            "the API call itself needs a key).")
    return anthropic.Anthropic(api_key=api_key)


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


def detect_new_signal(exchange, asset: str, now: pd.Timestamp) -> tuple:
    """Fetch recent 4h candles and check whether the LAST FULLY CLOSED one
    just fired a vol_breakout entry. Returns (candidate_order_dict,
    features_dict), or (None, None) if there's no fresh signal (or not
    enough history yet). `features_dict` is the indicator snapshot handed
    to the LLM gate.
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
        return None, None

    df_features = create_features(df)
    raw_atr = df_features['ATR_14'] * df_features['close']
    entries = volatility_breakout_entries(df_features, lookback=LOOKBACK)

    side = int(entries.iloc[-1])
    atr = raw_atr.iloc[-1]
    if side == 0 or np.isnan(atr):
        return None, None

    signal_time = df_features.index[-1]
    signal_price = float(df_features['close'].iloc[-1])
    order = make_pending_order(asset, side, signal_time, signal_price, float(atr))

    last_row = df_features.iloc[-1]
    features = {key: (float(last_row[key]) if key in last_row.index and pd.notna(last_row[key]) else None)
                for key in FEATURE_KEYS}
    return order, features


def main():
    parser = argparse.ArgumentParser(
        description="LLM-gated shadow paper-test: vol_breakout candidates need Claude's "
                    "approval before becoming a paper trade.")
    parser.add_argument("--assets", type=str, nargs='+', default=ASSETS)
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    client = make_llm_client()
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
            candidate, features = None, None

        if candidate is not None:
            already_logged = ((state['asset'] == asset) &
                               (state['signal_time'] == candidate['signal_time'])).any()
            if not already_logged:
                decision = get_llm_decision(client, args.model, asset, candidate['side'],
                                             candidate['signal_price'], candidate['atr'], features)
                candidate['llm_decision'] = decision['decision']
                candidate['llm_confidence'] = decision['confidence']
                candidate['llm_reason'] = decision['reason']
                if decision['decision'] == 'approve':
                    new_rows.append(candidate)
                    print(f"  {asset}: LLM APPROVED side={candidate['side']:+d} "
                          f"(confidence {decision['confidence']:.2f}) - {decision['reason']}")
                else:
                    candidate['status'] = 'llm_rejected'
                    new_rows.append(candidate)
                    print(f"  {asset}: LLM REJECTED side={candidate['side']:+d} "
                          f"(confidence {decision['confidence']:.2f}) - {decision['reason']}")

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

    approved = (state['llm_decision'] == 'approve').sum() if len(state) else 0
    rejected = (state['status'] == 'llm_rejected').sum()
    pending = (state['status'] == 'pending_fill').sum()
    open_ = (state['status'] == 'filled').sum()
    expired = (state['status'] == 'expired_unfilled').sum()
    closed = state[state['status'].isin(['closed_win', 'closed_loss', 'closed_vertical'])]

    print(f"\nState: {approved} LLM-approved total, {rejected} LLM-rejected, "
          f"{pending} pending fill, {open_} open, {expired} expired unfilled, "
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

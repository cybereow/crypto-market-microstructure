"""Triple-barrier labeling and simple primary (rule-based) signal generators
for meta-labeling.

Meta-labeling (Lopez de Prado, "Advances in Financial Machine Learning"):
instead of asking an ML model "which direction will price go next?" (a
question with essentially no answerable signal at the single-candle level —
see the ~50% out-of-sample directional accuracy this repo measured on all
six assets), ask a much narrower, structured question: "if a simple rule
already flagged this bar as a candidate trade, will it actually hit its
profit target before its stop-loss?" That has real structure (a defined
risk/reward, a bounded holding period) for a classifier to learn from,
instead of raw next-tick noise.
"""
import numpy as np
import pandas as pd


def donchian_breakout_entries(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Primary (rule-based) signal: +1 when close breaks above the prior
    `lookback`-bar high (long candidate), -1 when it breaks below the prior
    `lookback`-bar low (short candidate), 0 otherwise. This is the classic
    Donchian-channel breakout — simple, well-known, not fit to this data.
    """
    prior_high = df['high'].rolling(lookback).max().shift(1)
    prior_low = df['low'].rolling(lookback).min().shift(1)

    entries = pd.Series(0, index=df.index)
    entries[df['close'] > prior_high] = 1
    entries[df['close'] < prior_low] = -1
    return entries


def rsi_reversion_entries(df: pd.DataFrame, rsi_col: str = 'RSI_14',
                           oversold: float = 30, overbought: float = 70) -> pd.Series:
    """Primary (rule-based) signal: +1 the bar RSI crosses back up through
    `oversold` (a long candidate — buy the bounce), -1 the bar it crosses
    back down through `overbought` (a short candidate — fade the spike).
    Classic mean-reversion trigger, structurally higher base win rate than
    breakout/trend-following (many small reversions vs a few large trends).
    """
    rsi = df[rsi_col]
    prev_rsi = rsi.shift(1)

    entries = pd.Series(0, index=df.index)
    entries[(prev_rsi < oversold) & (rsi >= oversold)] = 1
    entries[(prev_rsi > overbought) & (rsi <= overbought)] = -1
    return entries


def triple_barrier_labels(df: pd.DataFrame, entries: pd.Series, atr: pd.Series,
                           pt_mult: float = 1.5, sl_mult: float = 1.0,
                           max_holding: int = 18) -> pd.DataFrame:
    """For each non-zero entry, scan forward up to `max_holding` bars and
    label it by whichever barrier is touched first:
      - profit-take at entry_price +/- pt_mult * ATR (in the entry's direction)
      - stop-loss at entry_price -/+ sl_mult * ATR
      - vertical barrier (max_holding reached): label by the sign of the
        realized return at exit instead of leaving it undefined.

    If a bar's high/low touch both barriers simultaneously, the stop-loss is
    assumed to have hit first — the conservative assumption, since intra-bar
    order is unknown.

    Returns a DataFrame indexed like `entries` (rows where entries == 0 are
    dropped) with columns: side (the primary signal's direction), label (1
    win / 0 loss), ret (realized return of the trade), hold (bars held).
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

        entry_price = closes[i]
        pt_dist = pt_mult * atr_vals[i]
        sl_dist = sl_mult * atr_vals[i]
        if side > 0:
            upper = entry_price + pt_dist
            lower = entry_price - sl_dist
        else:
            upper = entry_price + sl_dist
            lower = entry_price - pt_dist

        label = None
        exit_price = None
        hold = 0
        end = min(i + max_holding, n - 1)
        for j in range(i + 1, end + 1):
            hold = j - i
            hit_upper = highs[j] >= upper
            hit_lower = lows[j] <= lower
            if hit_upper and hit_lower:
                # Ambiguous within-bar order — assume the adverse barrier hit first.
                exit_price = lower if side > 0 else upper
                label = 0
                break
            if hit_upper:
                exit_price = upper
                label = 1 if side > 0 else 0
                break
            if hit_lower:
                exit_price = lower
                label = 0 if side > 0 else 1
                break
        if label is None:
            # Vertical barrier: neither touched within max_holding.
            exit_price = closes[end]
            ret_at_end = side * (exit_price / entry_price - 1)
            label = 1 if ret_at_end > 0 else 0

        ret = side * (exit_price / entry_price - 1)
        rows.append({'side': side, 'label': label, 'ret': ret, 'hold': hold,
                     'entry_pos': i, 'exit_pos': i + hold})
        idx.append(df.index[i])

    return pd.DataFrame(rows, index=pd.Index(idx, name=df.index.name))

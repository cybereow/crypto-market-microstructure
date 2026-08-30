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


def volatility_breakout_entries(df: pd.DataFrame, lookback: int = 20,
                                 squeeze_pct: float = 0.35) -> pd.Series:
    """Breakout, but only out of a volatility SQUEEZE.

    Rationale distinct from plain Donchian: most breakouts fail because
    they fire in already-expanded, chopping markets where the "range" is
    just noise. Volatility is strongly autocorrelated and mean-reverting
    in *level*, so a breakout that occurs while Bollinger width sits in
    its own historical bottom quantile is far more likely to be the start
    of a genuine expansion rather than another whipsaw. This is a
    conditional filter on WHEN to trust a breakout, not a new direction rule.
    """
    prior_high = df['high'].rolling(lookback).max().shift(1)
    prior_low = df['low'].rolling(lookback).min().shift(1)

    # Squeeze: current BB width in the bottom quantile of its own recent history.
    width = df['bb_width']
    width_rank = width.rolling(100).rank(pct=True).shift(1)
    squeezed = width_rank <= squeeze_pct

    entries = pd.Series(0, index=df.index)
    entries[(df['close'] > prior_high) & squeezed] = 1
    entries[(df['close'] < prior_low) & squeezed] = -1
    return entries


def trend_pullback_entries(df: pd.DataFrame, lookback: int = 20,
                            rsi_col: str = 'RSI_14') -> pd.Series:
    """Buy dips *within* an established uptrend; sell rips within a downtrend.

    Rationale: plain RSI reversion fades every extreme, including the ones
    that are extreme because a strong trend is under way — those are exactly
    the losing half. Conditioning the reversion on longer-horizon trend
    direction keeps the "buy fear" edge but removes the trades that fight a
    dominant move. Trend from the 20/50 SMA relationship (slow, few
    parameters); pullback from a moderate RSI level rather than a deep one,
    because in a real trend price rarely reaches 30.
    """
    uptrend = (df['close_to_sma20'] > 0) & (df['close_to_sma50'] > 0)
    downtrend = (df['close_to_sma20'] < 0) & (df['close_to_sma50'] < 0)

    rsi = df[rsi_col]
    prev = rsi.shift(1)
    dip = (prev < 45) & (rsi >= 45)
    rip = (prev > 55) & (rsi <= 55)

    entries = pd.Series(0, index=df.index)
    entries[uptrend & dip] = 1
    entries[downtrend & rip] = -1
    return entries


def range_fade_entries(df: pd.DataFrame, lookback: int = 20,
                        band: float = 0.9) -> pd.Series:
    """Fade the edges of a Bollinger range while volatility is NOT expanding.

    Rationale: the mirror image of volatility_breakout_entries, and the
    reason both are worth testing — they are mutually exclusive regimes.
    Mean reversion at band edges pays when volatility is stable or
    contracting, and is precisely what gets run over when it expands. The
    ATR_ratio guard (short-horizon ATR vs long-horizon) is what separates
    the two cases; without it this is a coin flip.

    NOTE on units: `bb_position` from create_features() is
    (close - sma20) / (2*std20), i.e. centred on 0 and roughly spanning
    [-1, +1] — NOT a 0..1 percentile. Comparing it against 0.1/0.9 as if
    it were a percentile fires on ~38% of all bars with a 5:1 long skew,
    which is a range-fade signal in name only. Band edges are therefore
    |pos| >= band, and the trade is a *cross back inside* the band so the
    entry is a reversion rather than a stand-in for "price is extended".
    """
    pos = df['bb_position']
    prev = pos.shift(1)
    not_expanding = df['ATR_ratio'] < 1.05

    entries = pd.Series(0, index=df.index)
    # Was at/below the lower band, now crossing back up into it -> long.
    entries[(prev <= -band) & (pos > -band) & not_expanding] = 1
    # Was at/above the upper band, now crossing back down into it -> short.
    entries[(prev >= band) & (pos < band) & not_expanding] = -1
    return entries


def scan_triple_barrier(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                         start: int, side: int, entry_price: float,
                         pt_dist: float, sl_dist: float, max_holding: int) -> dict:
    """Scan forward from bar `start` (exclusive) up to `max_holding` bars and
    report whichever barrier is touched first:
      - profit-take at entry_price +/- pt_dist (in the entry's direction)
      - stop-loss at entry_price -/+ sl_dist
      - vertical barrier (max_holding reached): label by the sign of the
        realized return at exit instead of leaving it undefined.

    If a bar's high/low touch both barriers simultaneously, the stop-loss is
    assumed to have hit first — the conservative assumption, since intra-bar
    order is unknown.

    Shared by `triple_barrier_labels` (entry at the signal bar's close) and
    `src.execution.triple_barrier_from_fill` (entry at a simulated limit
    fill), so both agree on exactly how a barrier touch is resolved.

    Returns {'label', 'exit_price', 'hold'}.
    """
    n = len(closes)
    if side > 0:
        upper = entry_price + pt_dist
        lower = entry_price - sl_dist
    else:
        upper = entry_price + sl_dist
        lower = entry_price - pt_dist

    label = None
    exit_price = None
    hold = 0
    end = min(start + max_holding, n - 1)
    for j in range(start + 1, end + 1):
        hold = j - start
        hit_upper = highs[j] >= upper
        hit_lower = lows[j] <= lower
        if hit_upper and hit_lower:
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

    return {'label': label, 'exit_price': exit_price, 'hold': hold}


def triple_barrier_labels(df: pd.DataFrame, entries: pd.Series, atr: pd.Series,
                           pt_mult: float = 1.5, sl_mult: float = 1.0,
                           max_holding: int = 18) -> pd.DataFrame:
    """For each non-zero entry, label it with `scan_triple_barrier` using the
    signal bar's own close as the (assumed instant/market-fill) entry price.

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

        outcome = scan_triple_barrier(highs, lows, closes, i, side, entry_price,
                                       pt_dist, sl_dist, max_holding)
        ret = side * (outcome['exit_price'] / entry_price - 1)
        rows.append({'side': side, 'label': outcome['label'], 'ret': ret,
                     'hold': outcome['hold'], 'entry_pos': i, 'exit_pos': i + outcome['hold']})
        idx.append(df.index[i])

    return pd.DataFrame(rows, index=pd.Index(idx, name=df.index.name))

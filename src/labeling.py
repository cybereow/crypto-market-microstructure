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


def obi_momentum_entries(df: pd.DataFrame, lookback: int = 288, obi_col: str = 'obi',
                          quantile: float = 0.90) -> pd.Series:
    """Primary signal from top-of-book order-flow imbalance, not price
    action: +1 the bar `obi` crosses UP through its own rolling upper
    quantile (resting bid size has come to dominate resting ask size), -1
    the bar it crosses DOWN through the rolling lower quantile (symmetric
    selling pressure).

    Rationale, and why this is a genuinely different bet from every other
    signal in this repo: order-flow-imbalance literature (e.g. Cont,
    Kukanov & Stoikov 2014, "The Price Impact of Order Book Events") links
    a skewed resting bid/ask ratio to near-term price PRESSURE — the book
    itself, not its history. Every other signal here (Donchian breakout,
    RSI reversion, volatility squeeze) is a function of OHLC alone, which
    is exactly the kind of pattern a market maker or arb bot can already
    see and trade against before this repo's own bar even closes. OBI is
    derived from the live order book, not the printed price series.

    `lookback` sets the window OBI's own extremes are measured against
    (default 288 = 1 day of 5-minute bars), so "extreme" adapts to each
    period's baseline resting-size skew rather than a fixed 0..1 cutoff
    that would drift with whatever ratio happens to be typical.
    """
    obi = df[obi_col]
    upper = obi.rolling(lookback).quantile(quantile).shift(1)
    lower = obi.rolling(lookback).quantile(1 - quantile).shift(1)
    prev = obi.shift(1)

    entries = pd.Series(0, index=df.index)
    entries[(prev <= upper) & (obi > upper)] = 1
    entries[(prev >= lower) & (obi < lower)] = -1
    return entries


def funding_extreme_reversion_entries(df: pd.DataFrame, lookback: int = 90,
                                       quantile: float = 0.90,
                                       funding_col: str = 'funding_rate') -> pd.Series:
    """Primary signal from perpetual-futures FUNDING RATE, not price action
    or the order book: fade the crowd when funding sits at its own recent
    extreme.

    Rationale, and why this is a genuinely different bet from every other
    signal in this repo (see scripts/download_funding_vision.py, which
    already documents this hypothesis but had never been wired into an
    actual entry rule until this function): funding is a periodic payment
    from longs to shorts (when positive) or shorts to longs (when
    negative), sized to keep the perpetual's price tracking spot. An
    unusually high positive funding rate means longs are paying an
    unusually large premium to stay long -- direct evidence of crowded,
    leveraged long positioning, which is unusually exposed to a
    liquidation cascade on any dip (the same mechanism behind "extreme
    funding precedes local tops" being a standard piece of crypto-trading
    lore). The symmetric case (extreme negative funding) is crowded
    shorts, prone to a short-squeeze bounce. This is therefore a FADE
    (mean-reversion) signal, the opposite direction from
    `obi_momentum_entries`'s follow-the-pressure rule, despite the
    similar "cross a rolling quantile" mechanics: -1 (short) fires when
    funding crosses UP through its own rolling upper quantile (crowded
    longs -> fade with a short); +1 (long) fires on the symmetric
    downside cross (crowded shorts -> fade with a long).

    `lookback` sets the window funding's own recent extremes are measured
    against (default 90 bars = 15 days at 4h), so "extreme" adapts to
    each period's prevailing funding regime (which shifts with the
    market's overall bull/bear structure) instead of a fixed absolute
    cutoff.
    """
    funding = df[funding_col]
    upper = funding.rolling(lookback).quantile(quantile).shift(1)
    lower = funding.rolling(lookback).quantile(1 - quantile).shift(1)
    prev = funding.shift(1)

    entries = pd.Series(0, index=df.index)
    entries[(prev <= upper) & (funding > upper)] = -1
    entries[(prev >= lower) & (funding < lower)] = 1
    return entries


def obv_divergence_entries(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Primary signal: fade a price breakout that volume flow doesn't confirm.

    Rationale: on-balance volume (OBV, a running sum of each bar's volume
    signed by that bar's price direction) is meant to track buying/selling
    PRESSURE behind a move, not the price move itself. When price prints a
    new `lookback`-bar high but OBV does NOT also print a new
    `lookback`-bar high, the rally is happening without proportional
    volume behind it -- classic bearish divergence, and evidence the move
    is thinner/more exhausted than the price chart alone suggests. The
    symmetric case (a new price low without a new OBV low) is bullish
    divergence. Like `funding_extreme_reversion_entries`, this is
    therefore a FADE signal -- but built from this asset's OWN volume
    flow rather than perpetual-futures positioning, so it's a third,
    independent bet alongside funding and OBI (order-book imbalance,
    section 11): three different alt-data sources testing the same
    underlying idea (crowd positioning/participation, not price history
    alone, is where an edge might still be findable) from three different
    angles.

    Mirrors `donchian_breakout_entries`'s no-lookahead convention exactly
    (compare the current bar against the PRIOR `lookback` bars' extreme,
    via `.shift(1)`) so a "new high/low" here means the same thing it
    means everywhere else in this repo.
    """
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    prior_price_high = df['close'].rolling(lookback).max().shift(1)
    prior_price_low = df['close'].rolling(lookback).min().shift(1)
    prior_obv_high = obv.rolling(lookback).max().shift(1)
    prior_obv_low = obv.rolling(lookback).min().shift(1)

    price_breaks_high = df['close'] > prior_price_high
    price_breaks_low = df['close'] < prior_price_low
    obv_confirms_high = obv > prior_obv_high
    obv_confirms_low = obv < prior_obv_low

    entries = pd.Series(0, index=df.index)
    entries[price_breaks_high & ~obv_confirms_high] = -1
    entries[price_breaks_low & ~obv_confirms_low] = 1
    return entries


def btc_lead_lag_entries(df: pd.DataFrame, threshold: float = 0.03,
                          btc_ret_col: str = 'btc_ret_5') -> pd.Series:
    """Primary signal: BTC's own recent momentum, traded on a DIFFERENT
    asset (an altcoin) -- a cross-asset lead-lag bet, structurally unlike
    every other signal in this repo (all of which condition only on the
    traded asset's own price/volume/positioning history).

    Rationale: BTC is crypto's dominant liquidity venue and the asset
    most capital flows through first; altcoins routinely play "catch up"
    with a short lag rather than moving in perfect lockstep, so a strong
    BTC move not yet mirrored in an altcoin's own price is a candidate
    signal that the alt hasn't finished repricing. +1 (long) fires the
    bar BTC's trailing return crosses UP through `threshold`; -1 (short)
    on the symmetric downside cross. Fires on the CROSSING bar only (like
    `obi_momentum_entries`/`funding_extreme_reversion_entries`), not
    every bar the condition holds, so a sustained BTC move doesn't spawn
    one overlapping candidate per bar.

    `df` must already carry `btc_ret_col` -- BTC's own trailing return,
    reindexed onto this asset's own timestamps and forward-filled. Reuses
    `src.regime.build_btc_regime`'s `btc_ret_5` column by default rather
    than recomputing BTC's return from scratch (see
    scripts/backtest_btc_lead_lag.py for how the join is built).
    """
    btc_ret = df[btc_ret_col]
    prev = btc_ret.shift(1)

    entries = pd.Series(0, index=df.index)
    entries[(prev <= threshold) & (btc_ret > threshold)] = 1
    entries[(prev >= -threshold) & (btc_ret < -threshold)] = -1
    return entries


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

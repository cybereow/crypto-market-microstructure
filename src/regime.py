"""Cross-asset market-regime context (BTC as the crypto market factor).

Why an explicit alignment feature instead of just another raw column:
crypto is a one-factor market — BTC's trend explains a large share of every
altcoin's move. The naive way to give a model that information is to append
`BTC_MACD` as a feature and hope the trees discover the interaction
`sign(trade_direction) x sign(BTC_trend)` on their own. With only a few
hundred pooled training trades that discovery is unlikely: it requires the
tree to split on the primary signal's side first and then on BTC's trend
within each branch, spending scarce depth and samples on a relationship we
already know from domain knowledge.

So we compute the interaction directly. `btc_alignment` answers "is this
candidate trade WITH the market or AGAINST it?" as a single feature that
needs no depth to exploit. It is also kept as a separate, non-model column
because the decision layer (src/gating.py) uses it to move the confidence
threshold explicitly rather than relying on the model to internalize it.

Leakage note: every column here is built from BTC data at or before the
bar in question (rolling/EWM means, shifted comparisons), then joined on
the candidate trade's own entry timestamp. Nothing peeks forward.
"""
import numpy as np
import pandas as pd


def build_btc_regime(btc_df: pd.DataFrame, fast: int = 12, slow: int = 26,
                     trend_len: int = 50, vol_len: int = 30) -> pd.DataFrame:
    """Reduce a BTC OHLCV frame to a handful of market-regime columns.

    Returns (indexed like btc_df):
      btc_trend        : +1 bullish / -1 bearish  (MACD line above/below its signal)
      btc_trend_strength: normalized MACD histogram — how decisive that trend is
      btc_above_sma    : +1/-1, close above/below its `trend_len` SMA (slower,
                         less whippy confirmation of the same idea)
      btc_ret_5        : BTC's own recent 5-bar return (market beta context)
      btc_vol          : BTC realized volatility (risk-on/risk-off regime)
      btc_vol_ratio    : short vs long realized vol — is volatility expanding?
    """
    close = btc_df['close'].astype(float)

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    # Trend DIRECTION comes from the raw EMA difference, not from a
    # price-normalized MACD compared against its own signal line. Dividing by
    # `close` before taking that comparison introduces a spurious drift: in a
    # steady uptrend the raw EMA gap is roughly constant while `close` keeps
    # growing, so the normalized series *falls* and the comparison flips the
    # regime to bearish in the middle of a rally. Direction is scale-free, so
    # it is read from the unnormalized gap.
    ema_gap = ema_fast - ema_slow

    # Trend STRENGTH is normalized, because its magnitude does need to be
    # comparable across price levels and across assets.
    macd_norm = ema_gap / (close + 1e-9)
    macd_signal = macd_norm.ewm(span=9, adjust=False).mean()

    sma_trend = close.rolling(trend_len).mean()
    log_ret = np.log(close / close.shift(1))
    vol_short = log_ret.rolling(10).std()
    vol_long = log_ret.rolling(vol_len).std()

    out = pd.DataFrame(index=btc_df.index)
    out['btc_trend'] = np.where(ema_gap > 0, 1.0, -1.0)
    out['btc_trend_strength'] = macd_norm - macd_signal
    out['btc_above_sma'] = np.where(close > sma_trend, 1.0, -1.0)
    out['btc_ret_5'] = close.pct_change(5)
    out['btc_vol'] = vol_short * np.sqrt(365 * 6)  # 6 four-hour bars per day
    out['btc_vol_ratio'] = vol_short / (vol_long + 1e-9)

    # A regime is only meaningful once its longest lookback is warm.
    return out


def add_alignment_features(trades: pd.DataFrame, regime: pd.DataFrame,
                           side_col: str = 'side') -> pd.DataFrame:
    """Join BTC regime onto candidate trades and derive the alignment
    interaction features.

    `trades` must be indexed by entry timestamp and carry a `side` column
    (+1 long / -1 short) — i.e. the output of triple_barrier_labels.

    Adds:
      btc_alignment          : 1.0 if the trade's direction agrees with
                               BTC's trend, else 0.0. THE key feature —
                               "am I trading with the market or against it".
      btc_alignment_sma      : same against the slower SMA regime.
      btc_alignment_strength : signed trend strength multiplied by the
                               trade's side, so the model sees not just
                               agreement but how strong the agreement is
                               (a weakly bullish BTC is not the same
                               tailwind as a decisively bullish one).
      btc_ret_5_aligned      : BTC's recent return in the trade's own
                               direction.

    Rows whose timestamp has no warm regime data are left as NaN for the
    caller to drop — never silently filled, since a fabricated "neutral"
    regime would be indistinguishable from a real one.
    """
    joined = trades.join(regime, how='left')
    side = joined[side_col].astype(float)

    joined['btc_alignment'] = (np.sign(side) == np.sign(joined['btc_trend'])).astype(float)
    joined.loc[joined['btc_trend'].isna(), 'btc_alignment'] = np.nan

    joined['btc_alignment_sma'] = (np.sign(side) == np.sign(joined['btc_above_sma'])).astype(float)
    joined.loc[joined['btc_above_sma'].isna(), 'btc_alignment_sma'] = np.nan

    joined['btc_alignment_strength'] = side * joined['btc_trend_strength']
    joined['btc_ret_5_aligned'] = side * joined['btc_ret_5']
    return joined


REGIME_FEATURE_COLS = [
    'btc_trend', 'btc_trend_strength', 'btc_above_sma', 'btc_ret_5',
    'btc_vol', 'btc_vol_ratio',
    'btc_alignment', 'btc_alignment_sma', 'btc_alignment_strength',
    'btc_ret_5_aligned',
]

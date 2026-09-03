"""A daily signal digest built on top of this repo's existing, validated
primitives — the answer to "give me 3-4 tradeable signals a day."

Why this module exists, and what it deliberately does *not* claim
--------------------------------------------------------------------
Every experiment in `docs/RESEARCH_LOG.md` converges on one wall: on OHLC
data at retail cost the raw per-trade edge (~0.15%) is smaller than a taker
round-trip (~0.40%), so *firing more often makes you lose faster*, not
richer. So a "3-4 signals a day" product cannot be built by loosening a
filter until enough trades appear — that just buys more negative-expectancy
trades. It has to be built the other way round: generate a *wide* candidate
pool, then spend a fixed daily budget of N slots on the highest-conviction
candidates only.

This module therefore does two separate jobs, kept separate on purpose:

1.  **Candidate generation** — pool every rule-based primary signal already
    in `src/labeling.py` across a universe of assets, and attach each one's
    triple-barrier outcome. This is unchanged, already-tested machinery.

2.  **Conviction ranking** — score each candidate with a *fixed, a-priori*
    formula built only from the three effects the research log actually
    *measured* as real (not the ones it retracted):
      - BTC-regime alignment (RESEARCH_LOG §8: trades aligned with BTC won
        53.4% vs 48.9% against — a confirmed, out-of-sample effect);
      - volatility state matched to the signal's economic logic (a squeeze
        for breakouts, contracting vol for reversions — the rationale each
        signal in labeling.py is built on);
      - own-asset trend agreement, signed by whether the signal is a
        pro-trend or a counter-trend bet.
    There is **no fitted parameter** in the score. Nothing here is tuned to
    maximise a backtest number, which is exactly the failure mode §7
    documents (four tuned ideas that raised win rate 48.8%->62.4% and then
    failed significance testing). A transparent, unfitted score cannot
    p-hack itself.

The honest expectation, stated up front so the code can be judged against
it: at taker cost this will lose money (the §8 wall is arithmetic, not a
model flaw); at maker cost the ranked top slice is where a small, real edge
can survive. `scripts/daily_signal_report.py` reports both, side by side,
with the same significance tests the rest of the repo uses. The point of
this module is to spend a daily signal budget *well*, not to pretend the
wall isn't there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.labeling import triple_barrier_labels
from src.labeling import (
    donchian_breakout_entries,
    rsi_reversion_entries,
    volatility_breakout_entries,
    trend_pullback_entries,
    range_fade_entries,
)

# Each primary signal, tagged by whether its economic logic is a *trend*
# bet (wants price to keep going) or a *reversion* bet (wants price to snap
# back). The tag decides the sign of two ranking components below, so a
# reversion signal is not scored as if it were a breakout.
SIGNAL_FAMILY = {
    'breakout': 'trend',
    'vol_breakout': 'trend',
    'trend_pullback': 'trend',
    'reversion': 'reversion',
    'range_fade': 'reversion',
}

SIGNAL_FUNCS = {
    'breakout': lambda df, lb: donchian_breakout_entries(df, lookback=lb),
    'vol_breakout': lambda df, lb: volatility_breakout_entries(df, lookback=lb),
    'trend_pullback': lambda df, lb: trend_pullback_entries(df, lookback=lb),
    'reversion': lambda df, lb: rsi_reversion_entries(df),
    'range_fade': lambda df, lb: range_fade_entries(df, lookback=lb),
}

# Fixed ranking weights. Declared here, once, a priori. BTC alignment gets
# the largest share because it is the effect with the most out-of-sample
# support in the log; the other two are the per-signal economic rationale.
# These are NOT swept — changing them to chase a backtest number would
# reintroduce exactly the multiple-testing problem significance.py exists
# to catch.
W_BTC = 0.40
W_REGIME = 0.30
W_TREND = 0.30


def _rolling_pct_rank(s: pd.Series, window: int = 100) -> pd.Series:
    """Point-in-time percentile of the current value within its own trailing
    `window`, shifted by one bar so the current bar is never used to rank
    itself (no lookahead)."""
    return s.rolling(window).rank(pct=True).shift(1)


def btc_trend_strength(btc_df: pd.DataFrame, sma: int = 50) -> pd.Series:
    """Signed, bounded BTC regime in [-1, +1]: how far BTC's close sits above
    (+) or below (-) its own `sma`, squashed by tanh so a runaway move
    doesn't dominate. Computed only from data available at each bar.

    RESEARCH_LOG §8 found BTC-regime features were 5 of the model's 10 most
    important, and that alignment with BTC's direction is a real (if small)
    edge. This exposes that regime as a continuous number to align against.
    """
    sma_series = btc_df['close'].rolling(sma).mean()
    dist = btc_df['close'] / (sma_series + 1e-9) - 1.0
    # Scale so a ~10% distance from the SMA maps to ~tanh(1)=0.76; crypto
    # trends routinely run further, and we want the tail bounded, not linear.
    return np.tanh(dist * 10.0)


def conviction_score(cand: pd.DataFrame, btc_strength_at: pd.Series) -> pd.Series:
    """Fixed-weight conviction in [0, 1] for a frame of candidate signals.

    Required columns on `cand`:
      side          : +1 long / -1 short (the primary signal's direction)
      signal        : signal name (keys SIGNAL_FAMILY)
      bb_width_rank : trailing percentile of Bollinger width at the bar
      atr_ratio     : short/long ATR at the bar (>1 expanding, <1 contracting)
      own_trend     : signed own-asset trend, e.g. sign(close_to_sma50)

    `btc_strength_at` is BTC regime strength aligned to each candidate's
    timestamp. All inputs are point-in-time; the score adds no new lookahead.
    """
    side = cand['side'].to_numpy(dtype=float)
    fam = cand['signal'].map(SIGNAL_FAMILY).to_numpy()
    bb_rank = cand['bb_width_rank'].to_numpy(dtype=float)
    atr_ratio = cand['atr_ratio'].to_numpy(dtype=float)
    own_trend = cand['own_trend'].to_numpy(dtype=float)
    btc = btc_strength_at.to_numpy(dtype=float)

    # 1) BTC alignment: +side in the direction of BTC's regime, mapped to
    #    [0, 1]. Aligned-with-a-strong-BTC -> ~1, against-a-strong-BTC -> ~0.
    s_btc = np.clip(0.5 + 0.5 * side * btc, 0.0, 1.0)

    # 2) Regime fit, per family:
    #    trend signals want a volatility SQUEEZE (low bb-width percentile),
    #    reversion signals want CONTRACTING vol (atr_ratio below 1). This is
    #    just each signal's own documented rationale, turned into a score.
    is_trend = fam == 'trend'
    s_regime = np.where(
        is_trend,
        1.0 - np.nan_to_num(bb_rank, nan=0.5),           # squeeze is good
        np.clip(1.2 - np.nan_to_num(atr_ratio, nan=1.0), 0.0, 1.0),  # contracting is good
    )
    s_regime = np.clip(s_regime, 0.0, 1.0)

    # 3) Own-asset trend agreement. A trend signal wants side aligned with
    #    the asset's own trend; a reversion signal is a counter-trend bet, so
    #    it is rewarded for the OPPOSITE alignment (fading into the trend's
    #    own direction is what gets run over). Neutral (0.5) when flat.
    agree = side * np.sign(own_trend)
    s_trend = np.where(is_trend, 0.5 + 0.5 * agree, 0.5 - 0.5 * agree)
    s_trend = np.clip(s_trend, 0.0, 1.0)

    score = W_BTC * s_btc + W_REGIME * s_regime + W_TREND * s_trend
    return pd.Series(score, index=cand.index)


def build_candidates(df_features: pd.DataFrame, raw_atr: pd.Series, asset: str,
                     signals, lookback: int, pt_mult: float, sl_mult: float,
                     max_holding: int) -> pd.DataFrame:
    """Run every requested primary signal on one asset and return the pooled
    candidate trades with their triple-barrier outcome and the raw feature
    values the conviction score needs. One row per fired candidate."""
    frames = []
    for sig in signals:
        entries = SIGNAL_FUNCS[sig](df_features, lookback)
        labelled = triple_barrier_labels(df_features, entries, raw_atr,
                                          pt_mult=pt_mult, sl_mult=sl_mult,
                                          max_holding=max_holding)
        if labelled.empty:
            continue
        labelled = labelled.copy()
        labelled['signal'] = sig
        labelled['asset'] = asset
        # Attach point-in-time features by position (entry_pos indexes the
        # signal bar). All of these already exist on df_features.
        pos = labelled['entry_pos'].to_numpy()
        bb_width = df_features['bb_width']
        bb_rank = _rolling_pct_rank(bb_width, 100).to_numpy()
        atr_ratio = df_features['ATR_ratio'].to_numpy()
        own_trend = np.sign(df_features['close_to_sma50'].to_numpy())
        labelled['bb_width_rank'] = bb_rank[pos]
        labelled['atr_ratio'] = atr_ratio[pos]
        labelled['own_trend'] = own_trend[pos]
        frames.append(labelled)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def attach_scores(candidates: pd.DataFrame, btc_strength: pd.Series) -> pd.DataFrame:
    """Add a `score` column (conviction in [0,1]) to a pooled candidate frame.
    `btc_strength` is indexed by timestamp; it is reindexed onto each
    candidate's own timestamp (forward-filled so a candidate between BTC bars
    uses the last known regime, never a future one)."""
    cand = candidates.copy()
    ts = cand.index
    # Candidate timestamps repeat (same bar across assets and signals), so a
    # direct reindex onto them is illegal. Build a de-duplicated, forward-
    # filled lookup over the unique timestamps, then map each candidate's
    # (possibly duplicated) timestamp through it. ffill guarantees a
    # candidate never reads a BTC regime from the future.
    btc_sorted = btc_strength.sort_index()
    uniq = pd.Index(ts).unique().sort_values()
    lookup = btc_sorted.reindex(btc_sorted.index.union(uniq)).ffill().reindex(uniq)
    btc_at = pd.Series(pd.Index(ts).map(lookup), index=ts, dtype=float)
    cand['btc_strength'] = btc_at.to_numpy()
    cand['score'] = conviction_score(cand, btc_at).to_numpy()
    return cand


def select_top_n_per_day(scored: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """The daily digest: within each UTC calendar day, keep the `n`
    highest-conviction candidates across the whole universe. Fewer than `n`
    on quiet days; never more. This is the literal "3-4 signals a day"
    product.

    Note: ranking within a day uses that day's full candidate set, so a
    signal early in the day is ranked against ones later the same day — a
    bounded (<=24h) same-day lookahead. `select_by_threshold` below is the
    fully-causal counterpart used for the honest economics headline."""
    if scored.empty:
        return scored
    day = pd.Index(scored.index).normalize()
    out = (scored.assign(_day=day)
                 .sort_values('score', ascending=False)
                 .groupby('_day', group_keys=False)
                 .head(n)
                 .drop(columns='_day')
                 .sort_index())
    return out


def select_by_threshold(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Fully-causal selection: take every candidate whose conviction is at or
    above a fixed `threshold`. No same-day ranking, so nothing here depends
    on future candidates. Used for the trustworthy economics number; the
    threshold is a single constant (analogous to §7's fixed 0.55 gate),
    chosen to hit the target signal frequency, not to maximise returns."""
    if scored.empty:
        return scored
    return scored[scored['score'] >= threshold].sort_index()


def threshold_for_frequency(scored: pd.DataFrame, target_per_day: float) -> float:
    """The score threshold whose admitted count equals ~`target_per_day`
    signals over the sample's span. This calibrates *frequency only* — it
    never looks at trade outcomes — so it cannot leak return information into
    the selection, only volume."""
    if scored.empty:
        return 1.0
    span_days = max((scored.index.max() - scored.index.min()).days, 1)
    target_total = target_per_day * span_days
    if target_total >= len(scored):
        return float(scored['score'].min())
    q = 1.0 - target_total / len(scored)
    return float(scored['score'].quantile(q))

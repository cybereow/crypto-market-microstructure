"""Daily cross-sectional long/short signals — a genuinely different economic
bet from everything in §1-14, and the natural home for a "N signals a day"
quota.

Why this is not just another knob on the dead directional signal
----------------------------------------------------------------
Sections 8-14 all test the same shape of bet: an *absolute*, single-asset,
intraday directional trade whose raw edge (~0.15%) is smaller than cost.
This module bets on something orthogonal — *relative* strength across the
universe, held for a full day:

  - **Relative, not absolute.** We do not predict whether ETH goes up; we
    predict ETH does better/worse than the universe median and trade the
    spread. Long the strongest M, short the weakest M, dollar-neutral. The
    market's common move (the BTC beta that dominates every asset's daily
    return, and that §8's regime features kept re-discovering) cancels in a
    long-short book, leaving the cross-sectional dispersion — a different,
    less-arbitraged signal.
  - **Daily hold, so cost is a small fraction of the move.** A crypto asset
    moves ~2-4% in a day; a round-trip cost of 0.08-0.40% is a far smaller
    tax on that than it is on the 2xATR intraday barrier move §8 fought
    with. And a long/short book that only *re-ranks* each day turns over
    just the fraction of names that actually changed side — so cost scales
    with realised turnover, not with a fixed per-trade charge on every name.
  - **Exactly N signals a day, by construction.** M longs + M shorts = 2M
    positions every day. Set M=2 and the product is 4 signals a day with no
    threshold-fiddling — the quota is structural, not forced.

The honesty bar is the same as the rest of the repo: this file only
*computes* the book and its turnover-aware net return; `scripts/
cross_sectional_report.py` grades it with a bootstrap significance test on
the daily P&L, at taker AND maker cost, across a small a-priori set of
signals (momentum vs short-term reversal, a few lookbacks) with the
multiple-testing deflation §7 insists on. Whether the effect is real is an
output, not an assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def resample_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """1-hour OHLCV -> daily OHLCV. Kept here so every cross-sectional
    experiment resamples identically (a right-open daily bar, UTC)."""
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
           'volume': 'sum'}
    out = df_1h.resample('1D').agg(agg).dropna(how='any')
    return out


def build_close_panel(daily_by_asset: dict) -> pd.DataFrame:
    """Align each asset's daily close onto one shared date index — the panel
    every cross-sectional signal is computed from. Columns = assets, rows =
    days; missing days (an asset listed later) stay NaN and are excluded per
    day by the ranking, never forward-filled (that would invent prices)."""
    closes = {name: d['close'] for name, d in daily_by_asset.items()}
    panel = pd.DataFrame(closes).sort_index()
    return panel


def cross_sectional_feature(close_panel: pd.DataFrame, signal: str,
                            lookback: int) -> pd.DataFrame:
    """Per-day, per-asset score to rank on, using only information available
    at that day's close (shifted so today's own forward return can never
    leak in).

      momentum : past `lookback`-day return  (buy relative strength)
      reversal : negative past `lookback`-day return (buy relative weakness)

    Both are classic, pre-registered cross-sectional factors — momentum
    (Jegadeesh-Titman, and its documented crypto analogue) and short-horizon
    reversal (Lehmann/Lo-MacKinlay). Testing both *is* the hypothesis (does
    crypto's cross-section trend or mean-revert at a daily horizon?), not a
    fishing expedition — the same principled direction check §11 used.
    """
    past_ret = close_panel.pct_change(lookback)
    if signal == 'momentum':
        feat = past_ret
    elif signal == 'reversal':
        feat = -past_ret
    else:
        raise ValueError(f"unknown signal: {signal}")
    # Shift by one day: rank on yesterday's-close information, hold today.
    return feat.shift(1)


def long_short_book(feature: pd.DataFrame, m_per_side: int) -> pd.DataFrame:
    """Turn the per-day feature into a target weight panel: the top
    `m_per_side` names get +1/(2M), the bottom `m_per_side` get -1/(2M),
    everything else 0. Dollar-neutral (weights sum to ~0), gross exposure 1.

    A day with fewer than 2*m_per_side ranked assets is skipped (all-zero
    weights) rather than taking a lopsided book."""
    weights = pd.DataFrame(0.0, index=feature.index, columns=feature.columns)
    m = m_per_side
    w = 1.0 / (2 * m)
    for day, row in feature.iterrows():
        valid = row.dropna()
        if len(valid) < 2 * m:
            continue
        ranked = valid.sort_values(ascending=False)
        longs = ranked.index[:m]
        shorts = ranked.index[-m:]
        weights.loc[day, longs] = w
        weights.loc[day, shorts] = -w
    return weights


def backtest_long_short(close_panel: pd.DataFrame, weights: pd.DataFrame,
                        cost_per_side: float) -> pd.DataFrame:
    """Daily net P&L of the weight panel.

    Timing: weights set at day t's close (from t-shifted features) earn the
    forward return close[t]->close[t+1]. Turnover at t is the rebalance from
    the previous day's weights to today's, charged at `cost_per_side` per
    unit of one-way traded notional (so a name that keeps its side pays
    nothing; a name that flips pays on the full change). This is the honest
    cost model — cost tracks realised turnover, not a flat per-name charge.

    Returns a frame indexed by day with gross, cost, and net daily returns.
    """
    fwd_ret = close_panel.pct_change().shift(-1)  # close[t]->close[t+1] on row t
    gross = (weights * fwd_ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    # First day's turnover is the cost of putting the book on.
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * cost_per_side
    net = gross - cost
    out = pd.DataFrame({'gross': gross, 'cost': cost, 'net': net,
                        'turnover': turnover})
    # Drop the last row (no forward return available for it).
    return out.iloc[:-1].dropna(how='any')


def daily_funding_panel(funding_by_asset: dict, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-day total funding rate for each asset, aligned to `date_index`.

    Binance perps charge funding every 8h (3x/day). The downloader forward-
    fills the 8-hourly rate to an hourly series, so the true daily total is
    the sum of the ~3 distinct charges that day — recovered here as the daily
    mean of the hourly series times 3 (each 8h block is constant under the
    ffill, so mean*3 == sum of the three charges). Positive funding means
    longs pay shorts.
    """
    cols = {}
    for name, f in funding_by_asset.items():
        s = f['funding_rate'] if 'funding_rate' in f.columns else f.iloc[:, 0]
        daily = s.resample('1D').mean() * 3.0
        cols[name] = daily
    panel = pd.DataFrame(cols).reindex(date_index)
    return panel


def apply_funding(bt: pd.DataFrame, weights: pd.DataFrame,
                  funding_daily: pd.DataFrame) -> pd.DataFrame:
    """Subtract funding P&L from an existing backtest frame.

    A signed position weight `w_i` held over day t pays `w_i * f_i,t` in
    funding (long pays positive funding, short receives it), so the book's
    funding P&L is `-Σ w_i * f_i`. On a momentum book the longs are recent
    winners (often richly-funded) and the shorts recent losers, so this can
    be a real drag — which is exactly why it must be measured, not assumed
    away. Returns a copy of `bt` with `funding` and a funding-adjusted `net`.
    """
    f = funding_daily.reindex(index=weights.index, columns=weights.columns)
    funding_pnl = -(weights * f).sum(axis=1)
    out = bt.copy()
    fp = funding_pnl.reindex(out.index).fillna(0.0)
    out['funding'] = fp
    out['net'] = out['net'] + fp
    return out


def equity_stats(daily_net: pd.Series, periods_per_year: int = 365) -> dict:
    """Summary stats for a daily net-return series."""
    r = daily_net.dropna()
    n = len(r)
    if n == 0:
        return {'n_days': 0}
    total = float((1 + r).prod() - 1)
    mean = float(r.mean())
    vol = float(r.std())
    sharpe = float(mean / vol * np.sqrt(periods_per_year)) if vol > 0 else float('nan')
    ann_ret = float((1 + mean) ** periods_per_year - 1)
    # Max drawdown on the compounded curve.
    curve = (1 + r).cumprod()
    dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    return {'n_days': n, 'total_return': total, 'ann_return': ann_ret,
            'sharpe': sharpe, 'daily_mean': mean, 'daily_vol': vol,
            'max_drawdown': dd, 'hit_rate': float((r > 0).mean())}

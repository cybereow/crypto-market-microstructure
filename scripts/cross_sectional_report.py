"""Daily cross-sectional long/short backtest and digest.

Bets on relative strength across the universe (long the strongest M, short
the weakest M, dollar-neutral, re-ranked daily) — a market-neutral, daily-
hold strategy whose cost/move ratio is far friendlier than the intraday
directional bets of §8-14, and whose 2M positions/day *are* the "N signals a
day" product by construction.

It grades a small, pre-registered set of signals (momentum vs short-term
reversal, a few lookbacks) at taker AND maker cost, with a bootstrap
significance test on the daily P&L and the §7 multiple-testing deflation for
having looked at several. Whether the effect clears cost is the output.

Backtest:
  python scripts/cross_sectional_report.py \
      --data binance_ETH_USDT_1h.csv binance_SOL_USDT_1h.csv ... \
      --m-per-side 2

Today's book:
  python scripts/cross_sectional_report.py --data ... --today --signal momentum --lookback 14
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from src.cross_sectional_daily import (
    resample_daily, build_close_panel, cross_sectional_feature,
    long_short_book, backtest_long_short, equity_stats,
    daily_funding_panel, apply_funding,
)
from src.significance import bootstrap_mean_pvalue, deflated_pvalue


def load_daily_panel(data_files):
    daily = {}
    for f in data_files:
        path = os.path.join(OUTPUT_DIR, f)
        if not os.path.exists(path):
            print(f"  {f}: not found, skipping.")
            continue
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        name = f.replace('binance_', '').replace('_1h.csv', '').replace('.csv', '')
        daily[name] = resample_daily(df)
    return build_close_panel(daily)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs='+', required=True)
    p.add_argument("--m-per-side", type=int, default=2,
                   help="M longs + M shorts per day (M=2 -> 4 signals/day).")
    p.add_argument("--taker-cost", type=float, default=0.004,
                   help="Round-trip taker cost; charged per-side as half of this on turnover.")
    p.add_argument("--maker-cost", type=float, default=0.0008)
    p.add_argument("--signals", nargs='+', default=['momentum', 'reversal'])
    p.add_argument("--lookbacks", nargs='+', type=int, default=[3, 7, 14, 30])
    p.add_argument("--funding", action='store_true',
                   help="Subtract perpetual funding P&L (longs pay, shorts receive). Reads "
                        "binance_funding_<SYMBOL>.csv per asset; the honest short-leg cost.")
    p.add_argument("--today", action='store_true',
                   help="Print today's long/short book for a single --signal/--lookback.")
    p.add_argument("--signal", default='reversal', help="For --today.")
    p.add_argument("--lookback", type=int, default=3, help="For --today.")
    args = p.parse_args()

    panel = load_daily_panel(args.data)
    if panel.shape[1] < 2 * args.m_per_side:
        print(f"Need at least {2*args.m_per_side} assets; have {panel.shape[1]}.")
        return
    print(f"  Panel: {panel.shape[1]} assets x {panel.shape[0]} days "
          f"({panel.index.min():%Y-%m-%d} -> {panel.index.max():%Y-%m-%d})")

    funding_daily = None
    if args.funding:
        fby = {}
        for name in panel.columns:
            symbol = name.replace('_', '')  # ETH_USDT -> ETHUSDT
            fpath = os.path.join(OUTPUT_DIR, f"binance_funding_{symbol}.csv")
            if os.path.exists(fpath):
                fby[name] = pd.read_csv(fpath, index_col='timestamp', parse_dates=True)
            else:
                print(f"    funding for {name} not found ({fpath}); its funding treated as 0.")
        funding_daily = daily_funding_panel(fby, panel.index) if fby else None
        if funding_daily is not None:
            print(f"  Funding: modelling perpetual funding P&L on {len(fby)}/{panel.shape[1]} "
                  f"assets (longs pay, shorts receive).")

    if args.today:
        feat = cross_sectional_feature(panel, args.signal, args.lookback)
        weights = long_short_book(feat, args.m_per_side)
        last = weights.iloc[-1]
        longs = last[last > 0].index.tolist()
        shorts = last[last < 0].index.tolist()
        day = weights.index[-1]
        print(f"\n{'='*60}\n  CROSS-SECTIONAL BOOK — {day:%Y-%m-%d} "
              f"({args.signal}, lookback {args.lookback}d)\n{'='*60}")
        for a in longs:
            print(f"    LONG   {a}")
        for a in shorts:
            print(f"    SHORT  {a}")
        print(f"\n  Dollar-neutral, re-rank daily. Not investment advice; "
              f"see the backtest for cost-adjusted stats.")
        return

    # Number of configurations tried, for the deflation correction.
    n_configs = len(args.signals) * len(args.lookbacks)
    print(f"\n  Signals x lookbacks tried: {n_configs} "
          f"(bootstrap p is deflated for this many).")

    for cost, label in [(args.taker_cost, 'TAKER'), (args.maker_cost, 'MAKER')]:
        cost_per_side = cost / 2.0
        print(f"\n{'='*98}")
        print(f"  {label} cost = {cost:.2%} round-trip  ({cost_per_side:.3%}/side on turnover)"
              f"   |  M={args.m_per_side} per side -> {2*args.m_per_side} signals/day")
        print(f"{'='*98}")
        print(f"  {'signal':<10}{'lookback':>9}{'ann_ret':>10}{'sharpe':>8}"
              f"{'total':>9}{'maxDD':>8}{'hit%':>7}{'turn/d':>8}{'boot_p*':>9}")
        print(f"  {'-'*93}")
        rows = []
        for signal, lb in itertools.product(args.signals, args.lookbacks):
            feat = cross_sectional_feature(panel, signal, lb)
            weights = long_short_book(feat, args.m_per_side)
            bt = backtest_long_short(panel, weights, cost_per_side)
            if funding_daily is not None:
                bt = apply_funding(bt, weights, funding_daily)
            st = equity_stats(bt['net'])
            boot_p = bootstrap_mean_pvalue(bt['net'].to_numpy(), n_iter=5000)
            boot_p_def = deflated_pvalue(boot_p, n_configs)
            rows.append((signal, lb, st, boot_p_def, bt['turnover'].mean()))
            print(f"  {signal:<10}{lb:>9}{st['ann_return']*100:>9.1f}%{st['sharpe']:>8.2f}"
                  f"{st['total_return']*100:>8.0f}%{st['max_drawdown']*100:>7.0f}%"
                  f"{st['hit_rate']*100:>6.0f}%{bt['turnover'].mean():>8.2f}{boot_p_def:>9.4f}")

        best = max(rows, key=lambda r: r[2]['sharpe'] if np.isfinite(r[2]['sharpe']) else -9)
        s, lb, st, pdef, _ = best
        sig = np.isfinite(pdef) and pdef < 0.05 and st['sharpe'] > 0
        print(f"\n  Best by Sharpe: {s} lookback {lb}d -> Sharpe {st['sharpe']:.2f}, "
              f"ann {st['ann_return']:.1%}, deflated p={pdef:.4f} "
              f"-> {'SIGNIFICANT' if sig else 'not significant after correction'}")

    funding_note = ("funding P&L IS included above (--funding)."
                    if funding_daily is not None else
                    "shorting perps carries funding cost NOT modelled here — pass --funding "
                    "to include it.")
    print(f"\n  * boot_p: bootstrap p that daily net expectancy > 0, deflated for "
          f"{n_configs} configs.\n  Turnover is one-way daily traded fraction of the book; "
          f"cost already scales with it.\n  Reminder: {funding_note} Maker economics still "
          f"assume passive fills — treat as an optimistic floor (§8-9).")


if __name__ == "__main__":
    main()

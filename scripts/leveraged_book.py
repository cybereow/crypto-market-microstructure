"""§18 — the aggressive configuration: the §17 book run at variable (in-time)
leverage via volatility targeting.

This is the most aggressive strategy in the repo and the write-up in
RESEARCH_LOG §18 states its four hard caveats. It is NOT a free lunch: it
takes the §17 cross-sectional momentum edge (a real but modest Sharpe ~1.0
book) and (1) holds its risk constant through vol-targeting — which honestly
lifts Sharpe to ~1.3 because it spends the edge more evenly — and (2)
multiplies it by a constant leverage to chase a return target. The return
target is bought entirely with leverage; the leverage buys the drawdown too.

  # base vol-targeted book (~1.5x avg leverage)
  python scripts/leveraged_book.py --data <20 1h csvs>

  # scaled toward a ~50%/yr target (adds constant leverage on top)
  python scripts/leveraged_book.py --data <20 1h csvs> --extra-leverage 1.5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from src.cross_sectional_daily import (
    resample_daily, build_close_panel, cross_sectional_feature,
    long_short_book, overlap_rebalance, backtest_long_short, volatility_scale,
    equity_stats,
)


def load_panel(data_files):
    daily = {}
    for f in data_files:
        path = os.path.join(OUTPUT_DIR, f)
        if not os.path.exists(path):
            print(f"  {f}: not found, skipping."); continue
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        name = f.replace('binance_', '').replace('_1h.csv', '').replace('.csv', '')
        daily[name] = resample_daily(df)
    return build_close_panel(daily)


def report(r: pd.Series, tag: str, lev: pd.Series = None):
    st = equity_stats(r.dropna())
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    lev_txt = f"  avg_lev={lev.reindex(r.index).mean():.2f}" if lev is not None else ""
    print(f"\n  {tag}")
    print(f"    avg {st['ann_return']*100:.0f}%/yr   Sharpe {st['sharpe']:.2f}   "
          f"maxDD {st['max_drawdown']*100:.0f}%   hit {st['hit_rate']*100:.0f}%{lev_txt}")
    print("    year-by-year: " + "  ".join(f"{y}:{v:+.0f}%" for y, v in yr.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs='+', required=True)
    p.add_argument("--m-per-side", type=int, default=5)
    p.add_argument("--mom-lb", type=int, default=14)
    p.add_argument("--rebalance-days", type=int, default=14)
    p.add_argument("--maker-cost", type=float, default=0.0008)
    p.add_argument("--taker-cost", type=float, default=0.0040)
    p.add_argument("--target-vol", type=float, default=0.20,
                   help="Annualized volatility target for the variable-leverage overlay.")
    p.add_argument("--vol-lookback", type=int, default=30)
    p.add_argument("--lev-cap", type=float, default=3.0)
    p.add_argument("--extra-leverage", type=float, default=1.0,
                   help="Constant multiplier on top of vol-targeting to chase a return target. "
                        "This is the part that is pure leverage — it scales return AND drawdown.")
    args = p.parse_args()

    panel = load_panel(args.data)
    days = (panel.index.max() - panel.index.min()).days
    print(f"  Universe: {panel.shape[1]} assets x {panel.shape[0]} days "
          f"({panel.index.min():%Y-%m-%d}->{panel.index.max():%Y-%m-%d})")
    print(f"  Base book: §17 (M={args.m_per_side}, {args.mom_lb}d momentum, "
          f"overlapping {args.rebalance_days}d rebalance)")

    mom = cross_sectional_feature(panel, 'momentum', args.mom_lb)
    w = overlap_rebalance(long_short_book(mom, args.m_per_side), args.rebalance_days)

    for cost, label in [(args.maker_cost, 'MAKER (best case — assumes passive fills)'),
                        (args.taker_cost, 'TAKER (realistic retail — the honest floor)')]:
        base = backtest_long_short(panel, w, cost / 2.0)['net']
        report(base, f"[{label}]  §17 base, 1x", None)
        vt, lev = volatility_scale(base, args.target_vol, args.vol_lookback, args.lev_cap,
                                   args.extra_leverage)
        tag = "vol-targeted (§18)" + (f" x{args.extra_leverage:g} extra leverage"
                                      if args.extra_leverage != 1.0 else "")
        report(vt, f"[{label}]  {tag}", lev)

    print(f"\n  {'='*74}")
    print("  Honest caveats (see RESEARCH_LOG §18): (1) the MAKER numbers require passive")
    print("  limit fills — the TAKER block above is what survives if they don't; (2) a")
    print("  backtest maxDD of ~-30% is typically worse live; (3) leverage this size carries")
    print("  real liquidation risk on a sharp move; (4) the edge is concentrated in 2023-25")
    print("  and the vol-target was lightly tuned, so the forward number is below the backtest.")


if __name__ == "__main__":
    main()

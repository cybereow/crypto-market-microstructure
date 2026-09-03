"""§16 — improving the §15 cross-sectional momentum book the honest way.

Three levers, each a standard portfolio-construction improvement (NOT a
fitted parameter), measured one at a time as an ablation so each one's real
contribution is visible rather than assumed:

  1. Wider universe  — more assets -> more cross-sectional dispersion.
  2. Vol-targeting   — size each leg by 1/vol (risk parity within the leg).
  3. Carry factor    — blend momentum with a funding-rate carry factor
                       (long low-funding, short high-funding), a second,
                       lowly-correlated crypto factor.

Everything is graded at maker AND taker cost, with real funding P&L on the
book, a bootstrap significance test, and an out-of-sample half-split — the
same bar §15 was held to.

  python scripts/cross_sectional_v2.py --data <all 1h csvs> --m-per-side 3 \
      --mom-lb 14 --carry-weight 0.3 --ablation
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from src.cross_sectional_daily import (
    resample_daily, build_close_panel, cross_sectional_feature, composite_feature,
    long_short_book, vol_target_book, backtest_long_short, apply_funding,
    daily_funding_panel, realized_vol_panel, equity_stats,
)
from src.significance import bootstrap_mean_pvalue, deflated_pvalue


def load(data_files):
    daily, funding = {}, {}
    for f in data_files:
        path = os.path.join(OUTPUT_DIR, f)
        if not os.path.exists(path):
            print(f"  {f}: not found, skipping."); continue
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        name = f.replace('binance_', '').replace('_1h.csv', '').replace('.csv', '')
        daily[name] = resample_daily(df)
        fpath = os.path.join(OUTPUT_DIR, f"binance_funding_{name.replace('_', '')}.csv")
        if os.path.exists(fpath):
            funding[name] = pd.read_csv(fpath, index_col='timestamp', parse_dates=True)
    return daily, funding


def evaluate(panel, weights, funding_daily, cost_per_side, n_conf, days):
    bt = backtest_long_short(panel, weights, cost_per_side)
    if funding_daily is not None:
        bt = apply_funding(bt, weights, funding_daily)
    st = equity_stats(bt['net'])
    p = deflated_pvalue(bootstrap_mean_pvalue(bt['net'].to_numpy(), n_iter=5000), n_conf)
    st['boot_p_def'] = p
    st['turnover'] = bt['turnover'].mean()
    st['net'] = bt['net']
    return st


def row(tag, st):
    print(f"  {tag:<34}{st['ann_return']*100:>8.1f}%{st['sharpe']:>8.2f}"
          f"{st['total_return']*100:>8.0f}%{st['max_drawdown']*100:>8.0f}%"
          f"{st['hit_rate']*100:>7.0f}%{st['turnover']:>8.2f}{st['boot_p_def']:>9.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs='+', required=True)
    p.add_argument("--m-per-side", type=int, default=5,
                   help="Longs+shorts per side. Scale with the universe to hold roughly a "
                        "top/bottom quartile (M=5 for ~20 assets); see §16.")
    p.add_argument("--mom-lb", type=int, default=14)
    p.add_argument("--carry-weight", type=float, default=0.3)
    p.add_argument("--vol-lookback", type=int, default=20)
    p.add_argument("--taker-cost", type=float, default=0.004)
    p.add_argument("--maker-cost", type=float, default=0.0008)
    p.add_argument("--ablation", action='store_true')
    args = p.parse_args()

    daily, funding = load(args.data)
    panel = build_close_panel(daily)
    days = max((panel.index.max() - panel.index.min()).days, 1)
    fund_panel = daily_funding_panel(funding, panel.index) if funding else None
    vol_panel = realized_vol_panel(panel, args.vol_lookback)
    print(f"  Universe: {panel.shape[1]} assets x {panel.shape[0]} days "
          f"({panel.index.min():%Y-%m-%d}->{panel.index.max():%Y-%m-%d})  "
          f"funding on {len(funding)}/{panel.shape[1]}")
    print(f"  M={args.m_per_side} per side -> {2*args.m_per_side} signals/day\n")

    mom = cross_sectional_feature(panel, 'momentum', args.mom_lb)
    comp = composite_feature(panel, fund_panel, args.mom_lb, args.carry_weight)
    N_CONF = 4  # a handful of a-priori variants examined; deflate for them.

    hdr = f"  {'configuration':<34}{'ann':>9}{'Sharpe':>8}{'total':>8}{'maxDD':>8}{'hit%':>7}{'turn':>8}{'defl_p':>9}"
    for cost, label in [(args.maker_cost, 'MAKER 0.08%'), (args.taker_cost, 'TAKER 0.40%')]:
        cps = cost / 2.0
        print(f"{'='*95}\n  {label} round-trip (funding P&L included)\n{'='*95}")
        print(hdr); print('  ' + '-'*91)
        if args.ablation:
            row("1) momentum, equal-weight (§15)",
                evaluate(panel, long_short_book(mom, args.m_per_side), fund_panel, cps, N_CONF, days))
            row("2)  + vol-targeting",
                evaluate(panel, vol_target_book(mom, args.m_per_side, vol_panel), fund_panel, cps, N_CONF, days))
        final = evaluate(panel, vol_target_book(comp, args.m_per_side, vol_panel), fund_panel, cps, N_CONF, days)
        row(f"3)  + carry (w={args.carry_weight}) [FINAL]", final)
        if label.startswith('MAKER'):
            final_maker = final

    # OOS half-split on the FINAL config at maker cost.
    print(f"\n  Out-of-sample half-split (FINAL config, maker cost):")
    net = final_maker['net']
    half = len(net) // 2
    for tag, seg in [('FULL', net), ('H1', net.iloc[:half]), ('H2', net.iloc[half:])]:
        st = equity_stats(seg)
        pv = bootstrap_mean_pvalue(seg.to_numpy(), n_iter=5000)
        print(f"    {tag:<5} n={st['n_days']:>4}  ann={st['ann_return']*100:>6.1f}%  "
              f"Sharpe={st['sharpe']:>5.2f}  raw_p={pv:.4f}")

    print(f"\n  defl_p: bootstrap p(net expectancy>0), deflated for {N_CONF} variants. "
          f"Funding P&L is included. Maker economics assume passive fills (optimistic).")


if __name__ == "__main__":
    main()

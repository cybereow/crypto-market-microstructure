"""Walk-forward / sub-period stability check for the funding-rate-extreme-
reversion primary signal (README sections 14 and 18): is the pooled,
full-history positive point estimate (section 14: maker PF 1.07, exp
+0.14%, p=0.112) driven by one lucky stretch of calendar time, or does it
hold up reasonably consistently across sequential, non-overlapping chunks
of the same history?

Unlike scripts/backtest_meta_ml_walkforward.py, this signal is a fixed
rule, not a fitted model -- there is nothing to train per fold and
therefore no train/test leakage to purge/embargo. "Walk-forward" here
means something narrower but still real and still worth checking: split
the pooled candidate pool into N sequential, non-overlapping calendar
chunks and report each chunk's economics independently, so a result that
looks good only because one regime (a single strong bull or bear
stretch, say) dominates the pooled sample is visible instead of hidden
inside an average -- and a bootstrap p-value is computed per fold too,
since a positive point estimate on a handful of trades in one fold is
not the same claim as the pooled p=0.112.

Reuses scripts/backtest_funding_reversion.py's `run_asset` (same join,
feature-building, entry rule, and REALISTIC maker-fill simulation from
README section 9) rather than duplicating it, so a result here is
directly comparable to section 14/18's own numbers -- this is a
different SLICING of the identical computation, not a different
computation.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.backtest_funding_reversion import run_asset
from src.significance import bootstrap_mean_pvalue
from src.metrics import net_pf_expectancy


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward / sub-period stability check for a funding-based signal.")
    parser.add_argument("--data", type=str, nargs='+', required=True,
                        help="FUTURES OHLCV files (download_klines_vision.py --market futures).")
    parser.add_argument("--funding-data", type=str, nargs='+', required=True,
                        help="One funding-rate file per --data file, in the SAME order.")
    parser.add_argument("--signal", type=str, default="funding_reversion",
                        choices=["funding_reversion", "funding_reversion_confirmed"])
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--pt-mult", type=float, default=2.0)
    parser.add_argument("--sl-mult", type=float, default=2.0)
    parser.add_argument("--max-holding", type=int, default=18)
    parser.add_argument("--offset-mult", type=float, default=0.15)
    parser.add_argument("--queue-timeout", type=int, default=3)
    parser.add_argument("--maker-cost", type=float, default=0.0008)
    parser.add_argument("--n-folds", type=int, default=6,
                        help="Number of equal-WIDTH calendar chunks the full history is "
                             "split into (candidate density, not chunk width, varies).")
    args = parser.parse_args()

    if len(args.data) != len(args.funding_data):
        raise SystemExit(
            f"--data has {len(args.data)} file(s) but --funding-data has "
            f"{len(args.funding_data)} -- need exactly one funding file per asset, in the "
            f"same order.")

    maker_all = []
    for data_file, funding_file in zip(args.data, args.funding_data):
        data_path = os.path.join(OUTPUT_DIR, data_file)
        funding_path = os.path.join(OUTPUT_DIR, funding_file)
        if not os.path.exists(data_path):
            print(f"  {data_file}: not found, skipping.")
            continue
        if not os.path.exists(funding_path):
            print(f"  {funding_file}: not found, skipping {data_file}.")
            continue

        _, _, maker = run_asset(data_path, funding_path, args.signal, args.lookback,
                                args.pt_mult, args.sl_mult, args.max_holding,
                                args.offset_mult, args.queue_timeout)
        if maker.empty:
            print(f"  {data_file}: no filled candidates, skipping.")
            continue
        maker = maker.copy()
        maker['asset'] = data_file
        maker_all.append(maker)

    if not maker_all:
        print("No results.")
        return

    maker_df = pd.concat(maker_all).sort_index()
    start, end = maker_df.index.min(), maker_df.index.max()
    edges = pd.date_range(start, end, periods=args.n_folds + 1)

    print(f"Walk-forward stability check -- signal={args.signal}, {args.n_folds} folds, "
          f"{start.date()} to {end.date()}\n")

    fold_stats = []
    for i in range(args.n_folds):
        lo, hi = edges[i], edges[i + 1]
        mask = (maker_df.index >= lo) & (maker_df.index <= hi if i == args.n_folds - 1
                                         else maker_df.index < hi)
        fold = maker_df[mask]
        if fold.empty:
            print(f"  Fold {i + 1} ({lo.date()} to {hi.date()}): no candidates.")
            continue

        rets = fold['ret'].to_numpy()
        win_rate = float((fold['label'] == 1).mean())
        pf, exp = net_pf_expectancy(rets, args.maker_cost)
        p = bootstrap_mean_pvalue(rets - args.maker_cost)
        fold_stats.append({'fold': i + 1, 'n': len(fold), 'win_rate': win_rate, 'pf': pf,
                           'exp': exp, 'p': p})
        print(f"  Fold {i + 1} ({lo.date()} to {hi.date()}): n={len(fold):4d}  "
              f"win rate {win_rate:.1%}  PF {pf:.2f}  exp/trade {exp:+.4%}  p={p:.4f}")

    if not fold_stats:
        print("\nNo fold produced any candidates.")
        return

    n_pos = sum(1 for s in fold_stats if s['exp'] > 0)
    n_pf_above_1 = sum(1 for s in fold_stats if s['pf'] > 1.0)
    print(f"\n{'=' * 78}")
    print(f"  Stability across {len(fold_stats)} folds: {n_pos}/{len(fold_stats)} positive "
          f"net expectancy, {n_pf_above_1}/{len(fold_stats)} PF > 1.0")

    rets_all = maker_df['ret'].to_numpy()
    pf_all, exp_all = net_pf_expectancy(rets_all, args.maker_cost)
    p_all = bootstrap_mean_pvalue(rets_all - args.maker_cost)
    print(f"  Full-history pooled reference (non-walk-forward, sections 14/18 style): "
          f"n={len(maker_df)}  PF {pf_all:.2f}  exp/trade {exp_all:+.4%}  p={p_all:.4f}")
    print(f"{'=' * 78}")
    print("\n  A signal driven by one regime shows up here as most folds flat/negative and "
          "one fold carrying the whole pooled result -- that is NOT the same claim as a "
          "consistently small positive edge across folds, even if the pooled p-value looks "
          "identical either way.")


if __name__ == "__main__":
    main()

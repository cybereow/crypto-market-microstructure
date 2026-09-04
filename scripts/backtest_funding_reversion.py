"""Does perpetual-futures FUNDING RATE -- alt-data, not price/order-book
action -- carry a real, cost-surviving edge as a standalone primary
signal?

`scripts/download_funding_vision.py`'s own docstring already states the
hypothesis (extreme positive funding, crowded longs, has historically
preceded local tops, and vice versa), but nothing in this repo had
actually turned that into a tradeable entry rule and tested it until
`src.labeling.funding_extreme_reversion_entries`. This script holds it to
exactly the same bar as every other primary signal here: taker AND maker
economics, the maker-order queue fill simulation from README section 9
(`src.execution.simulate_maker_fills`), and a bootstrap significance
test on the net-of-cost mean return. New does not mean exempt -- see
README section 7 for what happens to a result that skips this step.

Funding rate only exists for perpetual futures, so this pairs each
asset's --data FUTURES klines (download_klines_vision.py --market
futures) with its own --funding-data file (download_funding_vision.py),
matched by position. Unlike --obi-data on backtest_maker_fill.py (one
file shared across every asset), funding is asset-specific -- BTC's
funding history is not ETH's -- so each asset needs its own file, in the
same order as --data.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from scripts.train_meta_ml import SIGNAL_BUILDERS
from src.labeling import triple_barrier_labels
from src.execution import simulate_maker_fills, triple_barrier_from_fill
from src.significance import bootstrap_mean_pvalue
from src.metrics import net_pf_expectancy


def load_asset(data_path: str, funding_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    funding_df = pd.read_csv(funding_path, index_col='timestamp', parse_dates=True)
    df = df.join(funding_df[['funding_rate']], how='left')
    df['funding_rate'] = df['funding_rate'].ffill()
    return df


def run_asset(data_path: str, funding_path: str, signal: str, lookback: int,
              pt_mult: float, sl_mult: float, max_holding: int,
              offset_mult: float, queue_timeout: int) -> tuple:
    df = load_asset(data_path, funding_path)
    df_features = create_features(df)
    raw_atr = df_features['ATR_14'] * df_features['close']
    entries = SIGNAL_BUILDERS[signal](df_features, lookback)

    market = triple_barrier_labels(df_features, entries, raw_atr, pt_mult=pt_mult,
                                    sl_mult=sl_mult, max_holding=max_holding)
    fills = simulate_maker_fills(df_features, entries, raw_atr, offset_mult=offset_mult,
                                 queue_timeout=queue_timeout)
    maker = triple_barrier_from_fill(df_features, fills, raw_atr, pt_mult=pt_mult,
                                     sl_mult=sl_mult, max_holding=max_holding)
    return market, fills, maker


def main():
    parser = argparse.ArgumentParser(
        description="Backtest the funding-rate-extreme-reversion primary signal.")
    parser.add_argument("--data", type=str, nargs='+', required=True,
                        help="FUTURES OHLCV files (download_klines_vision.py --market futures).")
    parser.add_argument("--funding-data", type=str, nargs='+', required=True,
                        help="One funding-rate file per --data file, in the SAME order "
                             "(download_funding_vision.py).")
    parser.add_argument("--signal", type=str, default="funding_reversion",
                        choices=["funding_reversion", "funding_reversion_confirmed", "funding_reversion_regime_filtered"],
                        help="'funding_reversion_confirmed' additionally requires price "
                             "(bb_position) to independently confirm the crowding thesis "
                             "(see docs/RESEARCH_LOG.md section 18).")
    parser.add_argument("--lookback", type=int, default=90,
                        help="Bars funding's own extremes are measured against "
                             "(default 90 = 15 days at 4h).")
    parser.add_argument("--pt-mult", type=float, default=2.0)
    parser.add_argument("--sl-mult", type=float, default=2.0)
    parser.add_argument("--max-holding", type=int, default=18)
    parser.add_argument("--offset-mult", type=float, default=0.15,
                        help="How much better than the signal price the resting maker "
                             "limit is priced, in ATR multiples.")
    parser.add_argument("--queue-timeout", type=int, default=3,
                        help="Bars the resting limit order stays before being cancelled.")
    parser.add_argument("--taker-cost", type=float, default=0.004)
    parser.add_argument("--maker-cost", type=float, default=0.0008)
    parser.add_argument("--out", type=str, default="funding_reversion_backtest.csv")
    args = parser.parse_args()

    if len(args.data) != len(args.funding_data):
        raise SystemExit(
            f"--data has {len(args.data)} file(s) but --funding-data has "
            f"{len(args.funding_data)} -- need exactly one funding file per asset, in the "
            f"same order.")

    print(f"Funding-extreme-reversion backtest -- signal={args.signal}, "
          f"lookback={args.lookback} bars, pt/sl={args.pt_mult}/{args.sl_mult}\n")

    market_all, fills_all, maker_all = [], [], []
    for data_file, funding_file in zip(args.data, args.funding_data):
        data_path = os.path.join(OUTPUT_DIR, data_file)
        funding_path = os.path.join(OUTPUT_DIR, funding_file)
        if not os.path.exists(data_path):
            print(f"  {data_file}: not found, skipping.")
            continue
        if not os.path.exists(funding_path):
            print(f"  {funding_file}: not found, skipping {data_file}.")
            continue

        market, fills, maker = run_asset(data_path, funding_path, args.signal, args.lookback,
                                         args.pt_mult, args.sl_mult, args.max_holding,
                                         args.offset_mult, args.queue_timeout)
        if market.empty:
            print(f"  {data_file}: no candidates, skipping.")
            continue
        for frame in (market, fills, maker):
            frame['asset'] = data_file
        market_all.append(market)
        fills_all.append(fills)
        maker_all.append(maker)

        fill_rate = float(fills['filled'].mean()) if len(fills) else float('nan')
        longs, shorts = int((market['side'] == 1).sum()), int((market['side'] == -1).sum())
        print(f"  {data_file}: {len(market)} candidates ({longs}L/{shorts}S), "
              f"{len(maker)} filled ({fill_rate:.1%})")

    if not market_all:
        print("\nNo results.")
        return

    market_df = pd.concat(market_all)
    fills_df = pd.concat(fills_all)
    maker_df = pd.concat(maker_all)

    n_candidates = len(market_df)
    win_rate = float((market_df['label'] == 1).mean())
    taker_pf, taker_exp = net_pf_expectancy(market_df['ret'].to_numpy(), args.taker_cost)
    taker_p = bootstrap_mean_pvalue(market_df['ret'].to_numpy() - args.taker_cost)

    fill_rate = float(fills_df['filled'].mean()) if len(fills_df) else float('nan')
    if len(maker_df):
        maker_wr = float((maker_df['label'] == 1).mean())
        maker_pf, maker_exp = net_pf_expectancy(maker_df['ret'].to_numpy(), args.maker_cost)
        maker_p = bootstrap_mean_pvalue(maker_df['ret'].to_numpy() - args.maker_cost)
    else:
        maker_wr, maker_pf, maker_exp, maker_p = (float('nan'),) * 4

    print(f"\n{'=' * 78}")
    print(f"  Funding-extreme-reversion primary signal -- pooled across {len(market_all)} asset(s)")
    print(f"{'=' * 78}")
    print(f"  candidates: {n_candidates}   win rate: {win_rate:.1%}")
    print(f"  TAKER (instant fill, cost {args.taker_cost:.2%}) -- "
          f"PF {taker_pf:.2f}  exp/trade {taker_exp:+.4%}  p={taker_p:.4f}")
    print(f"  MAKER (simulated fill within {args.queue_timeout} bars, cost {args.maker_cost:.2%}) "
          f"-- n={len(maker_df)}  fill rate {fill_rate:.1%}  win rate {maker_wr:.1%}  "
          f"PF {maker_pf:.2f}  exp/trade {maker_exp:+.4%}  p={maker_p:.4f}")

    out_path = os.path.join(OUTPUT_DIR, args.out)
    maker_df.drop(columns=[c for c in ['asset'] if c in maker_df.columns], errors='ignore').to_csv(out_path)
    print(f"\n  Filled-trade detail saved to {out_path}")
    print("\n  Reminder: the MAKER figure is an OPTIMISTIC upper bound (OHLC bars carry no "
          "real queue-position data -- a touched price is not a guaranteed fill in a real "
          "book). If the edge does not survive even this, it will not survive live maker orders.")


if __name__ == "__main__":
    main()

"""Does the ~0.08% maker-cost edge (README section 8) survive actually
trying to get filled, or is it an artifact of assuming every candidate
trade fills instantly?

Every prior cost figure in this repo (including the controlled experiment
that isolated cost as the sole variable) assumed the ENTIRE candidate pool
gets filled and only varied the cost subtracted from each trade's return.
That is the right assumption for a taker/market order. It is not the right
assumption for a maker/limit order, which is the only way to actually reach
the sub-0.15% round-trip cost this signal's raw edge needs to survive.

This script replaces that assumption with `src.execution.simulate_maker_fills`
(see that module for the exact rule and its OPTIMISTIC-upper-bound caveat:
OHLC bars carry no true book/queue-position data) and reports three things
side by side:

  1. taker economics on ALL candidates (the number this repo has quoted so far)
  2. maker economics on the FILLED subset only, at the lower maker cost
  3. the fill rate, and whether the trades that failed to fill were
     disproportionately the ones that would have won (adverse selection) —
     the specific failure mode the README's execution caveat warns about.

Answering (3) is the actual test: if unfilled trades' would-be (taker-basis)
win rate is markedly higher than filled trades', the maker-cost edge is
partly or wholly an illusion — you'd be filling the losers and missing the
winners. If it is not, the maker-fill assumption behind the 0.08% figure
holds up under this (still optimistic) simulation.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from scripts.train_meta_ml import SIGNAL_BUILDERS
from src.labeling import triple_barrier_labels
from src.execution import simulate_maker_fills, triple_barrier_from_fill
from src.significance import bootstrap_mean_pvalue


def economics(rets: np.ndarray, cost: float) -> tuple:
    """Profit factor and mean net return per trade, after `cost` per trade."""
    if len(rets) == 0:
        return float('nan'), float('nan')
    net = rets - cost
    wins, losses = net[net > 0], net[net <= 0]
    pf = (wins.sum() / abs(losses.sum())
          if len(losses) and losses.sum() != 0
          else (float('inf') if len(wins) else 0.0))
    return float(pf), float(net.mean())


def run_asset(path: str, signal: str, lookback: int, pt_mult: float, sl_mult: float,
              max_holding: int, offset_mult: float, queue_timeout: int):
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df_features = create_features(df)
    raw_atr = df_features['ATR_14'] * df_features['close']
    entries = SIGNAL_BUILDERS[signal](df_features, lookback)

    market = triple_barrier_labels(df_features, entries, raw_atr,
                                    pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding)
    fills = simulate_maker_fills(df_features, entries, raw_atr,
                                  offset_mult=offset_mult, queue_timeout=queue_timeout)
    maker = triple_barrier_from_fill(df_features, fills, raw_atr,
                                      pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding)
    return market, fills, maker


def main():
    parser = argparse.ArgumentParser(
        description="Simulate maker (limit-order) fills for the primary signal and check "
                    "whether the maker-cost edge survives, or is eaten by adverse selection.")
    parser.add_argument("--data", type=str, nargs='+', required=True)
    parser.add_argument("--signal", type=str, default="vol_breakout",
                         choices=list(SIGNAL_BUILDERS.keys()))
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--pt-mult", type=float, default=2.0)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--max-holding", type=int, default=18)
    parser.add_argument("--offset-mult", type=float, default=0.15,
                         help="How much better than the signal price the resting limit is "
                              "priced, in ATR multiples.")
    parser.add_argument("--queue-timeout", type=int, default=3,
                         help="Bars the limit order stays resting before being cancelled.")
    parser.add_argument("--taker-fee-pct", type=float, default=0.001)
    parser.add_argument("--taker-slippage-pct", type=float, default=0.001)
    parser.add_argument("--maker-fee-pct", type=float, default=0.0002,
                         help="Binance BNB-discounted maker fee is ~0.02%% per side.")
    parser.add_argument("--maker-slippage-pct", type=float, default=0.0002)
    parser.add_argument("--out", type=str, default="maker_fill_sim.csv")
    args = parser.parse_args()

    taker_cost = 2 * (args.taker_fee_pct + args.taker_slippage_pct)
    maker_cost = 2 * (args.maker_fee_pct + args.maker_slippage_pct)

    print(f"Maker-fill queue simulation — signal={args.signal}, pt/sl={args.pt_mult}/{args.sl_mult}, "
          f"limit offset={args.offset_mult} ATR, timeout={args.queue_timeout} bars")
    print(f"taker round-trip cost: {taker_cost:.3%}   maker round-trip cost: {maker_cost:.3%}\n")

    market_all, fills_all, maker_all = [], [], []
    for data_file in args.data:
        path = os.path.join(OUTPUT_DIR, data_file)
        if not os.path.exists(path):
            print(f"  {data_file}: not found, skipping.")
            continue
        market, fills, maker = run_asset(path, args.signal, args.lookback, args.pt_mult,
                                          args.sl_mult, args.max_holding, args.offset_mult,
                                          args.queue_timeout)
        if market.empty:
            print(f"  {data_file}: no candidate trades, skipping.")
            continue
        for frame in (market, fills, maker):
            frame['asset'] = data_file
        market_all.append(market)
        fills_all.append(fills)
        maker_all.append(maker)
        fill_rate = fills['filled'].mean() if len(fills) else float('nan')
        print(f"  {data_file}: {len(market)} candidates, {len(maker)} filled ({fill_rate:.1%})")

    if not market_all:
        print("No results.")
        return

    market_df = pd.concat(market_all)
    fills_df = pd.concat(fills_all)
    maker_df = pd.concat(maker_all)

    n_candidates = len(market_df)
    fill_rate = float(fills_df['filled'].mean())

    taker_pf, taker_exp = economics(market_df['ret'].to_numpy(), taker_cost)
    taker_wr = float((market_df['label'] == 1).mean())
    taker_p = bootstrap_mean_pvalue(market_df['ret'].to_numpy() - taker_cost, n_iter=2000)

    maker_pf, maker_exp = economics(maker_df['ret'].to_numpy(), maker_cost)
    maker_wr = float((maker_df['label'] == 1).mean()) if len(maker_df) else float('nan')
    maker_p = (bootstrap_mean_pvalue(maker_df['ret'].to_numpy() - maker_cost, n_iter=2000)
               if len(maker_df) else float('nan'))

    # Adverse selection check: among candidates that never filled, what
    # would their TAKER-basis outcome have been? Join on (asset, timestamp)
    # since timestamps repeat across pooled assets.
    market_df['_key'] = list(zip(market_df['asset'], market_df.index))
    fills_df['_key'] = list(zip(fills_df['asset'], fills_df.index))
    unfilled_keys = set(fills_df.loc[~fills_df['filled'], '_key'])
    filled_keys = set(fills_df.loc[fills_df['filled'], '_key'])
    unfilled_wouldbe = market_df[market_df['_key'].isin(unfilled_keys)]
    filled_wouldbe = market_df[market_df['_key'].isin(filled_keys)]

    unfilled_wr = float((unfilled_wouldbe['label'] == 1).mean()) if len(unfilled_wouldbe) else float('nan')
    filled_wr_takerbasis = float((filled_wouldbe['label'] == 1).mean()) if len(filled_wouldbe) else float('nan')

    print(f"\n{'=' * 78}")
    print("  Taker (assumed instant fill, this repo's prior figures) vs "
          "maker (simulated fill)")
    print(f"{'=' * 78}")
    print(f"  candidates: {n_candidates}")
    print(f"  TAKER  — n={n_candidates:5d}  win rate {taker_wr:.1%}  PF {taker_pf:.2f}  "
          f"exp/trade {taker_exp:+.3%}  (cost {taker_cost:.2%})  p={taker_p:.4f}")
    print(f"  MAKER  — n={len(maker_df):5d}  win rate {maker_wr:.1%}  PF {maker_pf:.2f}  "
          f"exp/trade {maker_exp:+.3%}  (cost {maker_cost:.2%})  p={maker_p:.4f}")
    print(f"  fill rate: {fill_rate:.1%}  ({len(maker_df)}/{n_candidates} candidates filled "
          f"within {args.queue_timeout} bars)")

    print(f"\n{'=' * 78}")
    print("  Adverse selection check (taker-basis win rate of each group)")
    print(f"{'=' * 78}")
    print(f"  filled candidates   (n={len(filled_wouldbe):5d}): would-be win rate {filled_wr_takerbasis:.1%}")
    print(f"  unfilled candidates (n={len(unfilled_wouldbe):5d}): would-be win rate {unfilled_wr:.1%}")
    if not np.isnan(unfilled_wr) and not np.isnan(filled_wr_takerbasis):
        gap = unfilled_wr - filled_wr_takerbasis
        print(f"  gap (unfilled - filled): {gap:+.1%}")
        if gap > 0.02:
            print("  -> ADVERSE SELECTION: trades that failed to fill would have won more "
                  "often than trades that did. The maker-cost edge is partly an illusion.")
        else:
            print("  -> No material adverse selection detected in this simulation: fill "
                  "status is roughly independent of trade quality.")

    out_path = os.path.join(OUTPUT_DIR, args.out)
    maker_df.drop(columns=[c for c in ['asset'] if c in maker_df.columns], errors='ignore')
    maker_df.to_csv(out_path)
    print(f"\n  Filled-trade detail saved to {out_path}")
    print("\n  Reminder: this simulation is an OPTIMISTIC upper bound (OHLC bars carry no "
          "real queue-position data — a touched price is not a guaranteed fill in a real "
          "book). If the edge does not survive even this, it will not survive live maker orders.")


if __name__ == "__main__":
    main()

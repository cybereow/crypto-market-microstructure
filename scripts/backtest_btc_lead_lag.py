"""Does BTC's own recent momentum, traded on a DIFFERENT asset (an
altcoin), carry a real edge -- a cross-asset lead-lag bet, structurally
unlike every other signal in this repo (all of which condition only on
the traded asset's own price/volume/positioning history)?

`src.labeling.btc_lead_lag_entries` fires a long/short on an altcoin the
bar BTC's own trailing 5-bar return crosses a threshold, betting the alt
hasn't fully "caught up" to BTC's move yet. This script joins BTC's
regime (`src.regime.build_btc_regime`, already built and tested for the
meta-labeling gate's `btc_alignment` feature) onto each altcoin's own
bars, then holds the result to the same bar as every other primary
signal here: taker/maker economics, the maker-order queue fill
simulation from README section 9, and a bootstrap significance test.

Unlike `--funding-data` on backtest_funding_reversion.py (one file per
asset), BTC's own data is a SINGLE file shared across every altcoin
being tested -- pass it once via `--btc-data`, and `--data` should list
altcoins only (an asset can't lead-lag against itself).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from src.labeling import btc_lead_lag_entries, triple_barrier_labels
from src.execution import simulate_maker_fills, triple_barrier_from_fill
from src.regime import build_btc_regime
from src.significance import bootstrap_mean_pvalue
from src.metrics import net_pf_expectancy


def load_asset(data_path: str, btc_regime: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    df = df.join(btc_regime[['btc_ret_5']], how='left')
    df['btc_ret_5'] = df['btc_ret_5'].ffill()
    return df


def run_asset(data_path: str, btc_regime: pd.DataFrame, threshold: float,
              pt_mult: float, sl_mult: float, max_holding: int,
              offset_mult: float, queue_timeout: int) -> tuple:
    df = load_asset(data_path, btc_regime)
    df_features = create_features(df)
    raw_atr = df_features['ATR_14'] * df_features['close']
    entries = btc_lead_lag_entries(df_features, threshold=threshold)

    market = triple_barrier_labels(df_features, entries, raw_atr, pt_mult=pt_mult,
                                    sl_mult=sl_mult, max_holding=max_holding)
    fills = simulate_maker_fills(df_features, entries, raw_atr, offset_mult=offset_mult,
                                 queue_timeout=queue_timeout)
    maker = triple_barrier_from_fill(df_features, fills, raw_atr, pt_mult=pt_mult,
                                     sl_mult=sl_mult, max_holding=max_holding)
    return market, fills, maker


def main():
    parser = argparse.ArgumentParser(
        description="Backtest the BTC-lead-lag cross-asset primary signal on altcoins.")
    parser.add_argument("--data", type=str, nargs='+', required=True,
                        help="Altcoin OHLCV files (an asset can't lead-lag against itself, "
                             "so don't include BTC's own file here).")
    parser.add_argument("--btc-data", type=str, required=True,
                        help="BTC's own OHLCV file, shared across every altcoin above.")
    parser.add_argument("--threshold", type=float, default=0.03,
                        help="BTC's trailing 5-bar return must cross +/- this to fire.")
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
    parser.add_argument("--out", type=str, default="btc_lead_lag_backtest.csv")
    args = parser.parse_args()

    btc_path = os.path.join(OUTPUT_DIR, args.btc_data)
    if not os.path.exists(btc_path):
        raise SystemExit(f"--btc-data file not found: {btc_path}")
    btc_df = pd.read_csv(btc_path, index_col='timestamp', parse_dates=True)
    btc_regime = build_btc_regime(btc_df)

    print(f"BTC-lead-lag backtest -- threshold={args.threshold:+.1%}, "
          f"pt/sl={args.pt_mult}/{args.sl_mult}\n")

    market_all, fills_all, maker_all = [], [], []
    for data_file in args.data:
        data_path = os.path.join(OUTPUT_DIR, data_file)
        if not os.path.exists(data_path):
            print(f"  {data_file}: not found, skipping.")
            continue

        market, fills, maker = run_asset(data_path, btc_regime, args.threshold, args.pt_mult,
                                         args.sl_mult, args.max_holding, args.offset_mult,
                                         args.queue_timeout)
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
    print(f"  BTC-lead-lag primary signal -- pooled across {len(market_all)} altcoin(s)")
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

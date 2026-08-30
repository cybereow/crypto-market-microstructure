"""Cross-sectional strategy: instead of asking "will BTC go up?" (absolute,
single-asset direction — the ~50% coin-flip this repo already measured),
ask "which of these coins looks best RIGHT NOW, relative to the others?"
Market-wide noise that drowns out single-asset signals often cancels out
across a relative ranking, which is why cross-sectional momentum is one of
the more robust, well-documented effects in multi-asset markets.

Reuses the pooled meta-labeling model (train_meta_ml.py) as the ranking
score: at each rebalance checkpoint, every asset's current feature vector
is scored for P(a long looks good), assets are ranked, and the top-K are
held (equal-weighted) until the next checkpoint. Note this evaluates the
model on bars where its trained trigger (the primary breakout/reversion
signal) may not have actually fired — a mild distribution shift from
training, since the model learned "will THIS flagged setup work" rather
than "how good is a long in general" — treat this as an exploratory
extension of the meta-label score, not an equally rigorous claim as the
walk-forward validation in backtest_meta_ml_walkforward.py.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features


def main():
    parser = argparse.ArgumentParser(description="Cross-sectional ranking strategy across pooled assets.")
    parser.add_argument("--data", type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, default="meta_ml_model.json")
    parser.add_argument("--top-k", type=int, default=1, help="How many top-ranked assets to hold each period")
    parser.add_argument("--long-short", action="store_true",
                         help="Also short the bottom-K ranked assets (dollar-neutral) instead of going long-only")
    parser.add_argument("--rebalance-bars", type=int, default=12, help="Bars between rebalances (12 bars = 2 days at 4h)")
    parser.add_argument("--split-pct", type=float, default=0.8)
    parser.add_argument("--fee-pct", type=float, default=0.001)
    parser.add_argument("--slippage-pct", type=float, default=0.001)
    args = parser.parse_args()

    with open(os.path.join(OUTPUT_DIR, "meta_ml_features.txt")) as f:
        feature_cols = [c.strip() for c in f.read().split(',') if c.strip()]

    model = XGBClassifier()
    model.load_model(os.path.join(OUTPUT_DIR, args.model))

    asset_features = {}
    for data_file in args.data:
        data_path = os.path.join(OUTPUT_DIR, data_file)
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping.")
            continue
        df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        feats = create_features(df)
        feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
        feats['p_win'] = model.predict_proba(feats[feature_cols])[:, 1]
        split_ts = feats.index[int(len(feats) * args.split_pct)]
        asset_features[data_file] = feats[feats.index >= split_ts][['close', 'p_win']]

    if len(asset_features) < 2:
        print("Need at least 2 assets for cross-sectional ranking.")
        sys.exit(1)

    # Common OOS timeline: intersect indices so every checkpoint has a score
    # for every asset.
    common_idx = None
    for feats in asset_features.values():
        common_idx = feats.index if common_idx is None else common_idx.intersection(feats.index)
    common_idx = common_idx.sort_values()

    checkpoints = common_idx[::args.rebalance_bars]
    cost_per_trade = 2 * (args.fee_pct + args.slippage_pct)

    period_rets = []
    for i in range(len(checkpoints) - 1):
        t0, t1 = checkpoints[i], checkpoints[i + 1]
        scores = {name: feats.loc[t0, 'p_win'] for name, feats in asset_features.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        longs = ranked[:args.top_k]

        long_rets = []
        for name, score in longs:
            p0 = asset_features[name].loc[t0, 'close']
            p1 = asset_features[name].loc[t1, 'close']
            long_rets.append(p1 / p0 - 1 - cost_per_trade)

        if args.long_short:
            shorts = ranked[-args.top_k:]
            short_rets = []
            for name, score in shorts:
                p0 = asset_features[name].loc[t0, 'close']
                p1 = asset_features[name].loc[t1, 'close']
                short_rets.append(-(p1 / p0 - 1) - cost_per_trade)
            period_ret = 0.5 * np.mean(long_rets) + 0.5 * np.mean(short_rets)
        else:
            period_ret = np.mean(long_rets)

        period_rets.append(period_ret)

    rets = pd.Series(period_rets, index=checkpoints[:-1])
    # Each rebalance period is already a discrete, closed "trade" (a fresh
    # pick made and settled every checkpoint) — not a continuously-held
    # position, so win rate is just the fraction of periods with positive
    # return; trade_level_stats (built for a position series that stays
    # open across bars) doesn't apply here.
    win_rate = (rets > 0).mean()

    total_return = (1 + rets).prod() - 1
    periods_per_year = 365 / (args.rebalance_bars * 4 / 24)
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(periods_per_year)

    # Baseline: equal-weight all assets every period (no ranking at all).
    ew_rets = []
    for i in range(len(checkpoints) - 1):
        t0, t1 = checkpoints[i], checkpoints[i + 1]
        rs = [asset_features[name].loc[t1, 'close'] / asset_features[name].loc[t0, 'close'] - 1 - cost_per_trade
              for name in asset_features]
        ew_rets.append(np.mean(rs))
    ew_total_return = (1 + pd.Series(ew_rets)).prod() - 1

    print("=" * 55)
    print("  Cross-Sectional Ranking Strategy")
    print("=" * 55)
    print(f"  Assets:                     {len(asset_features)}")
    print(f"  Rebalance periods (OOS):    {len(rets)}")
    print(f"  Top-K held per period:      {args.top_k}")
    print(f"  Win Rate (per period):      {win_rate:>9.2%}")
    print(f"  Total Return:               {total_return:>9.2%}")
    print(f"  Equal-Weight Baseline:      {ew_total_return:>9.2%}")
    print(f"  Sharpe (period-annualized): {sharpe:>9.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()

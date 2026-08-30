"""Backtest the pooled meta-labeled model (see train_meta_ml.py) on one
asset's held-out out-of-sample candidate trades.

Sizing uses Kelly with the ACTUAL known payoff ratio b = pt_mult/sl_mult
(from the triple-barrier setup), rather than the heuristic edge/2 sizing
used elsewhere in this repo for raw direction prediction — here we really
do know the reward:risk ratio a taken trade is exposed to, so full/half
Kelly is the correct sizing, not just a proxy for "how confident are we".

Overlapping candidate trades on the same asset are conservatively skipped
(a real single-position bot can't open a second position while the first
is still live), so the reported equity curve is a plausible sequential
compounding of realized trades for a bot trading this one asset — it is
NOT a diversified multi-asset portfolio backtest.

NOTE: a single asset's held-out 20% is a small sample once you also filter
by confidence (often well under 50 trades), too little to trust on its own
— that's why the numbers here can swing wildly between assets. Treat this
script as a quick single-asset sanity check; scripts/backtest_meta_ml_walkforward.py
(pooled across assets, evaluated across multiple retrains) is the one with
enough sample size to actually validate whether the edge is real.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_meta_ml import build_asset_dataset, load_btc_regime
from src.novelty import load_novelty
from src.gating import GateConfig, apply_gate, select_non_overlapping


def kelly_fraction(p: np.ndarray, b: float, kelly_scale: float = 0.5, max_position: float = 1.0) -> np.ndarray:
    f = p - (1 - p) / b
    return np.clip(f * kelly_scale, 0.0, max_position)


def main():
    parser = argparse.ArgumentParser(description="Backtest the meta-labeled ML strategy on one asset.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta_ml_model.json")
    parser.add_argument("--confidence", type=float, default=None,
                         help="Minimum predicted P(win) to take a candidate trade. "
                              "Default: the threshold calibrated at training time "
                              "(data/meta_ml_threshold.json), which is the one that "
                              "was actually validated — override only for exploration.")
    parser.add_argument("--btc-regime-file", type=str, default=None,
                         help="BTC OHLCV CSV for cross-asset regime features. Must match "
                              "what training used, or the feature list will not line up.")
    parser.add_argument("--no-dynamic-threshold", action="store_true")
    parser.add_argument("--no-novelty", action="store_true")
    parser.add_argument("--kelly-scale", type=float, default=0.5, help="Fraction of full Kelly to size with")
    parser.add_argument("--fee-pct", type=float, default=0.001)
    parser.add_argument("--slippage-pct", type=float, default=0.001)
    parser.add_argument("--use-funding", action="store_true")
    args = parser.parse_args()

    params_path = os.path.join(OUTPUT_DIR, "meta_ml_params.txt")
    params = {}
    with open(params_path) as f:
        for line in f:
            k, v = line.strip().split('=')
            params[k] = float(v)
    pt_mult, sl_mult, max_holding, lookback = (
        params['pt_mult'], params['sl_mult'], int(params['max_holding']), int(params['lookback'])
    )
    payoff_ratio = pt_mult / sl_mult

    with open(os.path.join(OUTPUT_DIR, "meta_ml_features.txt")) as f:
        feature_cols = [c.strip() for c in f.read().split(',') if c.strip()]
    signal = 'breakout'
    signal_path = os.path.join(OUTPUT_DIR, "meta_ml_signal.txt")
    if os.path.exists(signal_path):
        with open(signal_path) as f:
            signal = f.read().strip()

    # The threshold calibrated during training is the default, so this
    # script evaluates the strategy that was actually validated rather than
    # an arbitrary constant.
    threshold_path = os.path.join(OUTPUT_DIR, "meta_ml_threshold.json")
    calibrated_threshold = None
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            calibrated_threshold = json.load(f).get('threshold')
    base_threshold = (args.confidence if args.confidence is not None
                      else (calibrated_threshold if calibrated_threshold is not None else 0.55))

    data_path = os.path.join(OUTPUT_DIR, args.data)
    funding_path = None
    if args.use_funding:
        base = args.data.rsplit('.', 1)[0]
        candidate = os.path.join(OUTPUT_DIR, f"{base}_funding.csv")
        if os.path.exists(candidate):
            funding_path = candidate

    btc_regime = load_btc_regime(args.btc_regime_file)
    _, test_df, _ = build_asset_dataset(data_path, lookback, pt_mult, sl_mult, max_holding,
                                         split_pct=0.8, funding_path=funding_path,
                                         signal=signal, btc_regime=btc_regime)
    if test_df is None or test_df.empty:
        print("No out-of-sample candidate trades for this asset.")
        return

    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        print(f"Error: the saved model expects features this run did not build: {missing}")
        print("       If the model was trained with --btc-regime-file, pass the same "
              "file here.")
        sys.exit(1)

    model = XGBClassifier()
    model.load_model(os.path.join(OUTPUT_DIR, args.model))
    test_df = test_df.copy()
    test_df['p_win'] = model.predict_proba(test_df[feature_cols])[:, 1]

    # OOD guard, if the training run saved one.
    detector = None if args.no_novelty else load_novelty(
        os.path.join(OUTPUT_DIR, "meta_ml_novelty.json"))
    is_novel = (detector.is_novel(model, test_df[feature_cols])
                if detector is not None else None)

    align = (test_df['btc_alignment'].to_numpy()
             if ('btc_alignment' in test_df.columns and not args.no_dynamic_threshold)
             else None)

    gate = GateConfig(base_threshold=base_threshold,
                      use_alignment=align is not None,
                      use_novelty=is_novel is not None)
    take, thresholds = apply_gate(test_df['p_win'].to_numpy(), gate,
                                  btc_alignment=align, is_novel=is_novel)

    base_rate = test_df['label'].mean()
    total_candidates = len(test_df)

    print("=" * 55)
    print(f"  Meta-Label Strategy — {args.data}")
    print("=" * 55)
    print(f"  Candidate trades (OOS):     {total_candidates:>6d}")
    print(f"  Base win rate (unfiltered): {base_rate:>10.2%}")
    print(f"  Base confidence threshold:  {base_threshold:>10.3f}"
          f"{'  (calibrated)' if args.confidence is None and calibrated_threshold else ''}")
    print(f"  Dynamic threshold range:    {thresholds.min():>10.3f} - {thresholds.max():.3f}")
    print(f"  BTC alignment gating:       {'on' if align is not None else 'off':>10}")
    print(f"  Leaf-novelty gating:        {'on' if is_novel is not None else 'off':>10}")
    print(f"  Payoff ratio (pt:sl):       {payoff_ratio:>10.2f}")

    # Enforce the single-position constraint a real bot has.
    taken_df = select_non_overlapping(test_df, take)
    if taken_df.empty:
        print("\n  No trades cleared the confidence threshold.")
        return
    cost_per_trade = 2 * (args.fee_pct + args.slippage_pct)  # entry + exit
    size = kelly_fraction(taken_df['p_win'].to_numpy(), payoff_ratio, args.kelly_scale)
    net_ret = taken_df['ret'].to_numpy() * size - size * cost_per_trade

    win_rate = (taken_df['label'] == 1).mean()
    wins = net_ret[net_ret > 0]
    losses = net_ret[net_ret <= 0]
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float('inf')
    total_return = np.prod(1 + net_ret) - 1

    avg_hold_days = taken_df['hold'].mean() * 4 / 24  # bars are 4h
    trades_per_year = 365 / max(avg_hold_days, 0.1)
    sharpe = (np.mean(net_ret) / (np.std(net_ret) + 1e-9)) * np.sqrt(trades_per_year) if len(net_ret) > 1 else 0.0

    print(f"\n  Trades taken:               {len(taken_df):>6d}")
    print(f"  Win Rate (per closed trade): {win_rate:>9.2%}")
    print(f"  Avg size (Kelly frac):      {size.mean():>10.2f}")
    print(f"  Profit Factor:              {profit_factor:>10.2f}")
    print(f"  Total Return (sequential):  {total_return:>10.2%}")
    print(f"  Sharpe (trade-annualized):  {sharpe:>10.2f}")
    print("=" * 55)
    if len(taken_df) < 50:
        print(f"  WARNING: {len(taken_df)} trades is too small a sample to conclude")
        print("  anything from. Use backtest_meta_ml_walkforward.py for the")
        print("  number that actually validates the edge.")


if __name__ == "__main__":
    main()

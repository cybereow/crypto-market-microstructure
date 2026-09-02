"""Meta-labeling training: pool multiple assets, label a simple rule-based
(Donchian breakout) primary signal with the triple-barrier method, and train
one XGBoost classifier to answer "will THIS candidate trade actually hit its
profit target before its stop-loss?" instead of "which way will price move?".

Why: scripts/train_ml.py's next-4h-candle direction model measured ~50%
out-of-sample directional accuracy on all six assets tested (BTC/ETH/SOL/
LINK/AVAX/DOT) — no exploitable edge at that granularity. Meta-labeling asks
a narrower, structured question (does a specific, already-defined trade
setup with a known risk/reward work out) which is the standard way to make
ML pull its weight in a trading system per Lopez de Prado's "Advances in
Financial Machine Learning".

Pooling across assets (instead of training one tiny model per asset) also
multiplies the sample count and pushes the model toward patterns that
generalize across coins rather than one coin's idiosyncratic noise.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from src.labeling import (donchian_breakout_entries, rsi_reversion_entries,
                          volatility_breakout_entries, trend_pullback_entries,
                          range_fade_entries, obi_momentum_entries,
                          funding_extreme_reversion_entries, funding_reversion_confirmed_entries,
                          funding_reversion_regime_filtered_entries,
                          obv_divergence_entries, triple_barrier_labels)
from src.regime import build_btc_regime, add_alignment_features, REGIME_FEATURE_COLS
from src.calibration import (precision_at_threshold_scorer, calibrate_threshold_for_precision,
                             precision_threshold_table)
from src.novelty import LeafNoveltyDetector, save_novelty

EXCLUDE_COLS = {'open', 'high', 'low', 'close', 'volume', 'target', 'ret_1d'}


def load_btc_regime(btc_file: str):
    """Build the market-regime frame once from a BTC OHLCV CSV, to be shared
    by every asset. Returns None when the file is missing so the caller can
    proceed without regime features rather than crashing.
    """
    if not btc_file:
        return None
    btc_path = os.path.join(OUTPUT_DIR, btc_file)
    if not os.path.exists(btc_path):
        print(f"Warning: BTC regime file {btc_path} not found — "
              f"proceeding WITHOUT cross-asset regime features.")
        return None
    btc_df = pd.read_csv(btc_path, index_col='timestamp', parse_dates=True)
    return build_btc_regime(btc_df)

SIGNAL_BUILDERS = {
    'breakout': lambda df, lookback: donchian_breakout_entries(df, lookback=lookback),
    'reversion': lambda df, lookback: rsi_reversion_entries(df),
    # Regime-conditional variants: each restricts an existing edge to the
    # market state where its economic rationale actually applies, instead
    # of firing in every state and diluting the average.
    'vol_breakout': lambda df, lookback: volatility_breakout_entries(df, lookback=lookback),
    'trend_pullback': lambda df, lookback: trend_pullback_entries(df, lookback=lookback),
    'range_fade': lambda df, lookback: range_fade_entries(df, lookback=lookback),
    # Order-flow, not price action -- requires an 'obi' column (see
    # scripts/download_l2_obi.py and --obi-data on backtest_maker_fill.py).
    'obi_momentum': lambda df, lookback: obi_momentum_entries(df, lookback=lookback),
    # Alt-data, not price/order-book action -- requires a 'funding_rate'
    # column (see scripts/download_funding_vision.py and
    # scripts/backtest_funding_reversion.py).
    'funding_reversion': lambda df, lookback: funding_extreme_reversion_entries(df, lookback=lookback),
    # funding_reversion, restricted to bars where price independently
    # confirms the crowding thesis too (see docstring, section 18).
    'funding_reversion_confirmed': lambda df, lookback: funding_reversion_confirmed_entries(df, lookback=lookback),
    # funding_reversion, gated off during volatility expansion (the SAME
    # ATR_ratio<1.05 guard range_fade_entries already uses) -- see
    # docstring, section 19-20.
    'funding_reversion_regime_filtered': lambda df, lookback: funding_reversion_regime_filtered_entries(df, lookback=lookback),
    # This asset's own volume flow vs. its own price -- needs only the
    # standard 'close'/'volume' OHLCV columns, no extra data source.
    'obv_divergence': lambda df, lookback: obv_divergence_entries(df, lookback=lookback),
}


def build_asset_labels(data_path: str, lookback: int, pt_mult: float, sl_mult: float,
                        max_holding: int, funding_path: str = None, signal: str = 'breakout',
                        btc_regime: pd.DataFrame = None):
    """Returns (labeled_df, feature_cols) for one asset: every candidate
    trade (a bar where the primary rule-based signal fired), across the
    ENTIRE series, joined with its feature snapshot at entry time and its
    triple-barrier outcome (label/ret/hold/entry_pos/exit_pos). No
    train/test split — callers slice by whatever boundary they need (a
    single chronological split, or repeated walk-forward boundaries).

    When `btc_regime` is supplied (see src/regime.py), each candidate trade
    additionally gets cross-asset market-context features — most importantly
    `btc_alignment`, whether the trade runs with or against BTC's trend.
    These are computed as an explicit interaction with the trade's own side
    rather than left for the trees to discover from a raw BTC column.
    """
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

    if funding_path and os.path.exists(funding_path):
        funding_df = pd.read_csv(funding_path, index_col='timestamp', parse_dates=True)
        df = df.join(funding_df, how='left')
        for col in funding_df.columns:
            df[col] = df[col].ffill()

    df_features = create_features(df)
    raw_atr = df_features['ATR_14'] * df_features['close']

    entries = SIGNAL_BUILDERS[signal](df_features, lookback)
    labels = triple_barrier_labels(df_features, entries, raw_atr,
                                    pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding)
    if labels.empty:
        return None, []

    feature_cols = [c for c in df_features.columns if c not in EXCLUDE_COLS]
    joined = labels.join(df_features[feature_cols], how='left')

    if btc_regime is not None:
        joined = add_alignment_features(joined, btc_regime)
        feature_cols = feature_cols + REGIME_FEATURE_COLS

    joined[feature_cols] = joined[feature_cols].replace([np.inf, -np.inf], np.nan)
    joined = joined.dropna(subset=feature_cols)
    return joined, feature_cols


def build_asset_dataset(data_path: str, lookback: int, pt_mult: float, sl_mult: float,
                         max_holding: int, split_pct: float, funding_path: str = None,
                         signal: str = 'breakout', btc_regime: pd.DataFrame = None):
    """Returns (train_df, test_df, feature_cols): build_asset_labels sliced
    at one chronological split point (as a fraction of the asset's bar
    count, not of its trade count, so it lines up with how every other
    script in this repo defines its 80/20 split).
    """
    df_full = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    joined, feature_cols = build_asset_labels(data_path, lookback, pt_mult, sl_mult, max_holding,
                                               funding_path=funding_path, signal=signal,
                                               btc_regime=btc_regime)
    if joined is None:
        return None, None, []

    split_ts = df_full.index[int(len(df_full) * split_pct)]
    train_df = joined[joined.index < split_ts]
    test_df = joined[joined.index >= split_ts]
    return train_df, test_df, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train a pooled, meta-labeled XGBoost classifier.")
    parser.add_argument("--data", type=str, nargs='+', required=True,
                         help="One or more CSV filenames (in the data dir) to pool for training.")
    parser.add_argument("--signal", type=str, default="breakout", choices=list(SIGNAL_BUILDERS.keys()),
                         help="Primary rule-based entry signal: 'breakout' (Donchian, trend-following) or 'reversion' (RSI 30/70, mean-reversion)")
    parser.add_argument("--lookback", type=int, default=20, help="Donchian breakout lookback (bars); unused for --signal reversion")
    parser.add_argument("--pt-mult", type=float, default=2.0, help="Profit-take distance, in ATR multiples")
    parser.add_argument("--sl-mult", type=float, default=2.0, help="Stop-loss distance, in ATR multiples")
    parser.add_argument("--max-holding", type=int, default=18, help="Max bars held before the vertical barrier")
    parser.add_argument("--split-pct", type=float, default=0.8, help="Chronological train/test split fraction, per asset")
    parser.add_argument("--model-out", type=str, default="meta_ml_model.json")
    parser.add_argument("--use-funding", action="store_true",
                         help="Merge <exchange>_funding_<SYMBOL>.csv (from download_funding_vision.py) per asset if present")
    parser.add_argument("--btc-regime-file", type=str, default=None,
                         help="BTC OHLCV CSV used to build cross-asset market-regime features "
                              "(btc_alignment etc). Strongly recommended — see src/regime.py.")
    parser.add_argument("--target-precision", type=float, default=0.65,
                         help="Win rate the calibrated confidence threshold aims for. "
                              "Replaces optimizing F1, which rewards taking trades.")
    parser.add_argument("--min-trades", type=int, default=30,
                         help="Minimum trades a calibrated threshold must still admit, so a "
                              "high win rate cannot be 'achieved' on a meaningless sample.")
    parser.add_argument("--scoring-quantile", type=float, default=0.8,
                         help="Model selection measures precision on scores above this "
                              "quantile — the region a selective strategy actually trades.")
    args = parser.parse_args()

    btc_regime = load_btc_regime(args.btc_regime_file)
    if btc_regime is not None:
        print(f"Loaded BTC market-regime context from {args.btc_regime_file} "
              f"({len(btc_regime)} bars).")

    all_train = []
    per_asset_test = {}
    feature_cols = None

    for data_file in args.data:
        data_path = os.path.join(OUTPUT_DIR, data_file)
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping.")
            continue

        funding_path = None
        if args.use_funding:
            base = data_file.rsplit('.', 1)[0]
            candidate = os.path.join(OUTPUT_DIR, f"{base}_funding.csv")
            if os.path.exists(candidate):
                funding_path = candidate

        train_df, test_df, feats = build_asset_dataset(
            data_path, args.lookback, args.pt_mult, args.sl_mult, args.max_holding,
            args.split_pct, funding_path=funding_path, signal=args.signal,
            btc_regime=btc_regime,
        )
        if train_df is None or train_df.empty:
            print(f"Warning: no candidate trades for {data_file}, skipping.")
            continue

        feature_cols = feats  # same feature set for every asset (create_features is deterministic)
        all_train.append(train_df)
        per_asset_test[data_file] = test_df
        print(f"{data_file}: {len(train_df)} train trades, {len(test_df)} test trades "
              f"(train win rate {train_df['label'].mean():.1%})")

    if not all_train:
        print("Error: no usable training data.")
        sys.exit(1)

    pooled_train = pd.concat(all_train).sort_index()
    X_train = pooled_train[feature_cols]
    y_train = pooled_train['label']

    print(f"\nPooled training set: {len(X_train)} trades across {len(all_train)} assets.")
    print(f"Pooled win rate (base rate, pre-model): {y_train.mean():.1%}")

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    # NOTE on validation: unlike train_ml.py's single-series purged
    # walk-forward split, this pools independent, interleaved per-asset
    # series, so a single chronological purge boundary doesn't map onto one
    # global time axis. For hyperparameter search we fall back to a plain
    # shuffled K-fold here (acceptable for a research prototype: the
    # leakage risk is bounded to trades whose barrier windows overlap,
    # which is a small fraction of the ~max_holding-bar window). The
    # train/test split used for the headline numbers below IS strictly
    # chronological per asset — that's the leakage-critical part.
    param_distributions = {
        'max_depth': [2, 3, 4],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.8],
        'min_child_weight': [5, 10, 20],
        'reg_alpha': [0.1, 1.0, 5.0],
        'reg_lambda': [3.0, 5.0, 10.0],
    }
    # Model selection targets PRECISION ON THE CONFIDENT SLICE, not F1.
    # F1's recall term rewards a model for firing often; a selective
    # strategy only ever trades its top-confidence candidates, so that is
    # the only region where being right has any value. See src/calibration.py
    # for why this is preferred over a custom asymmetric objective (which
    # would also decalibrate the probabilities the Kelly sizing depends on).
    #
    # NOTE: scale_pos_weight is deliberately NOT set here. Up-weighting the
    # positive class pushes the model toward predicting wins — precisely the
    # wrong bias when the goal is precision. It is kept out so probabilities
    # stay calibrated for Kelly sizing.
    base_model = XGBClassifier(random_state=42, eval_metric='logloss',
                                objective='binary:logistic')
    scorer = precision_at_threshold_scorer(quantile=args.scoring_quantile, min_support=10)
    search = RandomizedSearchCV(base_model, param_distributions=param_distributions,
                                 n_iter=15, scoring=scorer, cv=5, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV top-{(1 - args.scoring_quantile) * 100:.0f}% precision: {search.best_score_:.4f}")

    model = search.best_estimator_
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 features:")
    print(importances.head(10).to_string())

    # --- Confidence threshold calibration -------------------------------
    # Calibrated on the cross-validated out-of-fold predictions, NOT on
    # in-sample fits: a threshold tuned on data the model memorized would
    # read far too optimistically and collapse in live use.
    print("\n" + "=" * 60)
    print("  Confidence threshold calibration (target-precision search)")
    print("=" * 60)
    oof_probs = cross_val_predict(model, X_train, y_train, cv=5,
                                  method='predict_proba', n_jobs=-1)[:, 1]
    calib = calibrate_threshold_for_precision(
        y_train, oof_probs, target_precision=args.target_precision,
        min_trades=args.min_trades,
    )
    print(f"  Target precision:      {args.target_precision:.1%}")
    print(f"  Chosen threshold:      {calib['threshold']:.4f}")
    print(f"  Precision achieved:    {calib['precision']:.1%} (out-of-fold)")
    print(f"  Trades admitted:       {calib['n_trades']} of {len(y_train)} candidates")
    print(f"  Target met:            {calib['target_met']}  ({calib['reason']})")
    print("\n  Precision/volume tradeoff across thresholds (out-of-fold):")
    print(precision_threshold_table(y_train, oof_probs).to_string(index=False))

    # --- Leaf-novelty (OOD) detector -------------------------------------
    # Reuses this very model's leaf assignments as the novelty fingerprint,
    # so there is no second model to train or keep in sync. Persisted as a
    # small JSON of leaf occupancy counts.
    detector = LeafNoveltyDetector().fit(model, X_train)
    print(f"\nLeaf-novelty detector fitted on {len(X_train)} training trades "
          f"(rare-path cutoff at the {detector.rare_percentile:.0f}th percentile).")

    print("\n" + "=" * 60)
    print("  Per-asset out-of-sample evaluation")
    print("=" * 60)
    for data_file, test_df in per_asset_test.items():
        if test_df.empty:
            print(f"{data_file}: no OOS trades.")
            continue
        X_test = test_df[feature_cols]
        y_test = test_df['label']
        base_rate = y_test.mean()
        # Report at the CALIBRATED threshold, not the default 0.5 that
        # .predict() implies — 0.5 is not the threshold this strategy trades.
        p_test = model.predict_proba(X_test)[:, 1]
        y_pred = (p_test >= calib['threshold']).astype(int)
        print(f"\n{data_file} ({len(test_df)} OOS candidate trades, base win rate {base_rate:.1%}):")
        print(classification_report(y_test, y_pred, target_names=['Loss', 'Win'], zero_division=0))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, args.model_out)
    model.save_model(model_path)

    with open(os.path.join(OUTPUT_DIR, "meta_ml_features.txt"), 'w') as f:
        f.write(','.join(feature_cols))
    with open(os.path.join(OUTPUT_DIR, "meta_ml_params.txt"), 'w') as f:
        f.write(f"pt_mult={args.pt_mult}\nsl_mult={args.sl_mult}\n"
                f"max_holding={args.max_holding}\nlookback={args.lookback}\n")
    with open(os.path.join(OUTPUT_DIR, "meta_ml_signal.txt"), 'w') as f:
        f.write(args.signal)

    # The calibrated threshold and the novelty detector's leaf counts are
    # part of the trained artifact: a model shipped without them would be
    # evaluated at 0.5 with no OOD guard, which is not the strategy that was
    # validated.
    save_novelty(detector, os.path.join(OUTPUT_DIR, "meta_ml_novelty.json"))
    with open(os.path.join(OUTPUT_DIR, "meta_ml_threshold.json"), 'w') as f:
        json.dump({
            'threshold': calib['threshold'],
            'target_precision': args.target_precision,
            'oof_precision': calib['precision'],
            'oof_n_trades': calib['n_trades'],
            'target_met': calib['target_met'],
            'used_btc_regime': btc_regime is not None,
        }, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Calibrated threshold saved to {os.path.join(OUTPUT_DIR, 'meta_ml_threshold.json')}")
    print(f"Novelty leaf counts saved to {os.path.join(OUTPUT_DIR, 'meta_ml_novelty.json')}")


if __name__ == "__main__":
    main()

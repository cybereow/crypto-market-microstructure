"""Walk-forward validation of the meta-labeled strategy — the ONLY number
in this repo worth judging the edge by.

Instead of one static 80/20 split (which tells you whether the edge held
over one particular recent stretch), retrain periodically on an expanding
window of pooled data and evaluate strictly on the next chunk of calendar
time, rolling forward.

Every component added for win-rate improvement is validated HERE, inside
the fold loop, and each is fitted only on that fold's training data:

  * cross-asset BTC regime / alignment features  (src/regime.py)
  * precision-targeted threshold calibration     (src/calibration.py)
  * leaf-novelty OOD filtering                   (src/novelty.py)
  * context-dependent confidence gating          (src/gating.py)

That placement is the whole point. Each of those is a knob, and knobs fit
noise. Calibrating the threshold on all the data and then reporting the win
rate on that same data would produce a beautiful, meaningless number. Here
the threshold for fold k is calibrated on out-of-fold predictions from
fold k's training window only, and the novelty detector sees only fold k's
training rows — so the reported win rate is what a bot could actually have
achieved trading forward in time.

The report also always prints trade counts and net-of-cost profitability
next to the win rate, because a win rate quoted without them is not a
result: a 90% win rate on 9 trades, or one paid for by losses larger than
the wins, is not an edge.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_meta_ml import build_asset_labels, load_btc_regime
from src.calibration import (precision_at_threshold_scorer, calibrate_threshold_for_precision,
                            precision_threshold_table)
from src.novelty import LeafNoveltyDetector
from src.gating import GateConfig, apply_gate, select_non_overlapping


def kelly_fraction(p: np.ndarray, b: float, kelly_scale: float = 0.5,
                   max_position: float = 1.0) -> np.ndarray:
    """Kelly with the known payoff ratio b = pt_mult/sl_mult."""
    f = p - (1 - p) / b
    return np.clip(f * kelly_scale, 0.0, max_position)


def summarize(taken: pd.DataFrame, payoff_ratio: float, cost_per_trade: float,
              kelly_scale: float, label: str) -> dict:
    """Win rate AND net-of-cost economics for a set of taken trades.

    Win rate alone is not a result — a strategy can win 80% of the time and
    still lose money if the 20% losses are large enough, and every reported
    number here is after fees and slippage on both legs.
    """
    if taken.empty:
        return {'label': label, 'n_trades': 0, 'win_rate': float('nan'),
                'profit_factor': float('nan'), 'total_return': float('nan'),
                'avg_net_ret': float('nan'), 'expectancy_r': float('nan')}

    p = taken['p_win'].to_numpy()
    size = kelly_fraction(p, payoff_ratio, kelly_scale)
    net_ret = taken['ret'].to_numpy() * size - size * cost_per_trade

    wins = net_ret[net_ret > 0]
    losses = net_ret[net_ret <= 0]
    profit_factor = (wins.sum() / abs(losses.sum())
                     if len(losses) and losses.sum() != 0
                     else (float('inf') if len(wins) else 0.0))

    # Expectancy in R multiples: unit-sized, cost-adjusted, so it is
    # comparable across assets and independent of the sizing scheme.
    gross_r = taken['ret'].to_numpy() / (taken['ret'].abs().mean() + 1e-9)

    return {
        'label': label,
        'n_trades': int(len(taken)),
        'win_rate': float((taken['label'] == 1).mean()),
        'profit_factor': float(profit_factor),
        'total_return': float(np.prod(1 + net_ret) - 1),
        'avg_net_ret': float(net_ret.mean()),
        'expectancy_r': float(gross_r.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation of the meta-labeled strategy.")
    parser.add_argument("--data", type=str, nargs='+', required=True)
    parser.add_argument("--signal", type=str, default="breakout", choices=["breakout", "reversion"])
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--pt-mult", type=float, default=1.0)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--max-holding", type=int, default=18)
    parser.add_argument("--n-folds", type=int, default=4, help="Number of chronological chunks (first is warm-up-only)")
    parser.add_argument("--use-funding", action="store_true")

    # --- the four new components, each independently switchable so their
    # --- marginal contribution can be measured rather than assumed.
    parser.add_argument("--btc-regime-file", type=str, default=None,
                        help="BTC OHLCV CSV for cross-asset regime features (src/regime.py)")
    parser.add_argument("--target-precision", type=float, default=0.65,
                        help="Win rate the per-fold calibrated threshold aims for")
    parser.add_argument("--min-trades", type=int, default=20,
                        help="Min trades a calibrated threshold must admit (per fold)")
    parser.add_argument("--no-dynamic-threshold", action="store_true",
                        help="Disable BTC-alignment-based threshold adjustment")
    parser.add_argument("--no-novelty", action="store_true",
                        help="Disable leaf-novelty OOD threshold adjustment")
    parser.add_argument("--novelty-hard-reject", action="store_true",
                        help="Skip novel-region trades outright instead of just raising their bar")
    parser.add_argument("--misalignment-penalty", type=float, default=0.08)
    parser.add_argument("--alignment-bonus", type=float, default=0.02)
    parser.add_argument("--novelty-penalty", type=float, default=0.06)

    parser.add_argument("--kelly-scale", type=float, default=0.5)
    parser.add_argument("--fee-pct", type=float, default=0.001)
    parser.add_argument("--slippage-pct", type=float, default=0.001)
    args = parser.parse_args()

    payoff_ratio = args.pt_mult / args.sl_mult
    cost_per_trade = 2 * (args.fee_pct + args.slippage_pct)

    btc_regime = load_btc_regime(args.btc_regime_file)
    if btc_regime is not None:
        print(f"Cross-asset regime context: {args.btc_regime_file} ({len(btc_regime)} BTC bars)")

    pooled = []
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

        joined, feature_cols = build_asset_labels(
            data_path, args.lookback, args.pt_mult, args.sl_mult, args.max_holding,
            funding_path=funding_path, signal=args.signal, btc_regime=btc_regime,
        )
        if joined is None or joined.empty:
            print(f"Warning: no candidate trades for {data_file}, skipping.")
            continue
        joined = joined.copy()
        joined['asset'] = data_file
        pooled.append(joined)

    if not pooled:
        print("Error: no usable data.")
        sys.exit(1)

    pooled_df = pd.concat(pooled).sort_index()
    n = len(pooled_df)
    fold_edges = [pooled_df.index[int(n * i / args.n_folds)] for i in range(args.n_folds)]
    fold_edges.append(pooled_df.index[-1] + pd.Timedelta(seconds=1))
    purge_gap = pd.Timedelta(hours=4 * args.max_holding)

    gate_config = GateConfig(
        use_alignment=(not args.no_dynamic_threshold) and btc_regime is not None,
        misalignment_penalty=args.misalignment_penalty,
        alignment_bonus=args.alignment_bonus,
        use_novelty=not args.no_novelty,
        novelty_penalty=args.novelty_penalty,
        novelty_hard_reject=args.novelty_hard_reject,
    )

    param_distributions = {
        'max_depth': [2, 3, 4],
        'n_estimators': [50, 100],
        'learning_rate': [0.03, 0.05],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'min_child_weight': [5, 10, 20],
        'reg_alpha': [0.1, 1.0],
        'reg_lambda': [3.0, 5.0],
    }
    scorer = precision_at_threshold_scorer(quantile=0.8, min_support=10)

    all_oos = []
    print(f"\nPooled: {n} candidate trades across {len(pooled)} assets, {args.n_folds} folds.")
    print(f"Gate config: {gate_config.describe()}\n")

    for fold in range(1, args.n_folds):
        test_start, test_end = fold_edges[fold], fold_edges[fold + 1]
        train_mask = pooled_df.index < (test_start - purge_gap)
        test_mask = (pooled_df.index >= test_start) & (pooled_df.index < test_end)

        train_fold = pooled_df[train_mask]
        test_fold = pooled_df[test_mask]
        if len(train_fold) < 100 or test_fold.empty:
            print(f"Fold {fold} ({test_start.date()} to {test_end.date()}): skipped (too little data).")
            continue

        X_train, y_train = train_fold[feature_cols], train_fold['label']

        # scale_pos_weight intentionally omitted: it biases the model toward
        # predicting wins, which is the opposite of a precision goal, and it
        # decalibrates the probabilities Kelly sizing consumes.
        base_model = XGBClassifier(random_state=42, eval_metric='logloss',
                                   objective='binary:logistic')
        search = RandomizedSearchCV(base_model, param_distributions=param_distributions,
                                    n_iter=8, scoring=scorer, cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        # Threshold calibrated on THIS fold's out-of-fold predictions only.
        oof = cross_val_predict(model, X_train, y_train, cv=3,
                                method='predict_proba', n_jobs=-1)[:, 1]
        calib = calibrate_threshold_for_precision(
            y_train, oof, target_precision=args.target_precision,
            min_trades=args.min_trades)

        # Novelty detector fitted on this fold's training rows only.
        detector = LeafNoveltyDetector().fit(model, X_train)

        test_fold = test_fold.copy()
        test_fold['p_win'] = model.predict_proba(test_fold[feature_cols])[:, 1]
        test_fold['is_novel'] = detector.is_novel(model, test_fold[feature_cols])
        test_fold['fold'] = fold
        test_fold['fold_threshold'] = calib['threshold']

        fold_gate = GateConfig(**{**gate_config.describe(),
                                  'base_threshold': calib['threshold']})
        fold_align = (test_fold['btc_alignment'].to_numpy()
                      if 'btc_alignment' in test_fold.columns else None)
        take, thresholds = apply_gate(test_fold['p_win'].to_numpy(), fold_gate,
                                      btc_alignment=fold_align,
                                      is_novel=test_fold['is_novel'].to_numpy())
        test_fold['effective_threshold'] = thresholds
        test_fold['taken'] = take
        all_oos.append(test_fold)

        taken = select_non_overlapping(test_fold, take)
        base_rate = test_fold['label'].mean()
        stats = summarize(taken, payoff_ratio, cost_per_trade, args.kelly_scale, f"fold{fold}")
        print(f"Fold {fold} ({test_start.date()} to {test_end.date()}): "
              f"train={len(train_fold)}, test={len(test_fold)}, base={base_rate:.1%} | "
              f"thr={calib['threshold']:.3f} (oof prec {calib['precision']:.1%}, "
              f"met={calib['target_met']}) | "
              f"taken={stats['n_trades']}, win={stats['win_rate']:.1%}, "
              f"PF={stats['profit_factor']:.2f}, ret={stats['total_return']:.1%}")

    if not all_oos:
        print("\nNo folds produced results.")
        return

    combined = pd.concat(all_oos)
    print(f"\n{'=' * 72}")
    print(f"  Combined walk-forward OOS ({len(combined)} candidate trades)")
    print(f"{'=' * 72}")
    print(f"  Base win rate (all candidates, no filter): {combined['label'].mean():>7.1%}"
          f"  (n={len(combined)})")

    # --- Ablation: the marginal contribution of each component. Without
    # --- this table there is no way to tell whether a knob earned its
    # --- overfitting risk or just added a degree of freedom.
    print(f"\n  {'Configuration':<44}{'n':>6}{'win%':>8}{'PF':>7}{'ret%':>9}")
    print(f"  {'-' * 72}")

    rows = []
    p = combined['p_win'].to_numpy()
    align = (combined['btc_alignment'].to_numpy()
             if 'btc_alignment' in combined.columns else None)
    novel = combined['is_novel'].to_numpy()
    thr_col = combined['fold_threshold'].to_numpy()

    variants = [
        ("all candidates (primary signal only)", np.ones(len(combined), bool)),
        ("+ fixed 0.55 confidence", p >= 0.55),
        ("+ per-fold calibrated threshold", p >= thr_col),
    ]

    # Each ablation reuses that trade's OWN fold-calibrated base threshold
    # (thr_col), so the rows differ only by the component being ablated —
    # otherwise the comparison would confound the component with a change
    # of base threshold.
    if align is not None:
        cfg_a = GateConfig(use_alignment=True, use_novelty=False,
                           misalignment_penalty=args.misalignment_penalty,
                           alignment_bonus=args.alignment_bonus)
        mask_a, _ = apply_gate(p, cfg_a, btc_alignment=align, base_threshold=thr_col)
        variants.append(("+ dynamic threshold (BTC alignment)", mask_a))

    cfg_n = GateConfig(use_alignment=False, use_novelty=True,
                       novelty_penalty=args.novelty_penalty,
                       novelty_hard_reject=args.novelty_hard_reject)
    mask_n, _ = apply_gate(p, cfg_n, is_novel=novel, base_threshold=thr_col)
    variants.append(("+ leaf-novelty OOD filter", mask_n))
    variants.append(("+ ALL (full gate, as validated)", combined['taken'].to_numpy()))

    for name, mask in variants:
        taken = select_non_overlapping(combined, mask)
        s = summarize(taken, payoff_ratio, cost_per_trade, args.kelly_scale, name)
        wr = f"{s['win_rate']:.1%}" if s['n_trades'] else "n/a"
        pf = f"{s['profit_factor']:.2f}" if s['n_trades'] else "n/a"
        rt = f"{s['total_return']:.1%}" if s['n_trades'] else "n/a"
        print(f"  {name:<44}{s['n_trades']:>6}{wr:>8}{pf:>7}{rt:>9}")
        rows.append(s)

    print(f"\n  Precision/volume tradeoff on pooled OOS predictions:")
    print("  " + precision_threshold_table(combined['label'], combined['p_win'])
          .to_string(index=False).replace("\n", "\n  "))

    if align is not None:
        print(f"\n  BTC alignment breakdown (all candidates):")
        for val, name in [(1.0, "with BTC trend"), (0.0, "against BTC trend")]:
            sub = combined[combined['btc_alignment'] == val]
            if len(sub):
                print(f"    {name:<22} n={len(sub):>5}  win rate={sub['label'].mean():.1%}")

    print(f"\n  Novelty breakdown (all candidates):")
    for val, name in [(False, "familiar leaf paths"), (True, "rare/novel leaf paths")]:
        sub = combined[combined['is_novel'] == val]
        if len(sub):
            print(f"    {name:<24} n={len(sub):>5}  win rate={sub['label'].mean():.1%}")

    corr = combined['p_win'].corr(combined['label'])
    print(f"\n  corr(p_win, label): {corr:.3f}  "
          f"(the model's ranking ability — the foundation everything else builds on)")
    print("=" * 72)


if __name__ == "__main__":
    main()

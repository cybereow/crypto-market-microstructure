"""Sweep the triple-barrier payoff geometry and the primary signal, and
report the resulting base win rate, expectancy and the model's ranking
ability for each combination.

WHY THIS SCRIPT EXISTS (the most important finding in this repo):

Win rate is set primarily by the *barrier geometry*, not by the model.
A trade labeled with a profit target and a stop the same distance away
(pt_mult == sl_mult) is close to a coin flip by construction, so its base
win rate sits near 50% no matter how good the classifier is. Filtering such
trades by model confidence moved the measured walk-forward win rate from
50.3% to roughly 54% — and no amount of feature engineering, threshold
calibration or OOD filtering closes the gap to a 90% target, because the
target is not reachable from that geometry.

The identity that governs this:

    breakeven_win_rate = 1 / (1 + pt/sl)

    pt/sl = 2.0  ->  a 33% win rate breaks even; ~50% is achievable and good
    pt/sl = 1.0  ->  50% breaks even
    pt/sl = 0.33 ->  75% breaks even  (so 90% is a real, sizable edge)
    pt/sl = 0.2  ->  83% breaks even  (90% is a thin edge; costs may eat it)

So a high win rate is *purchasable* by taking small profits against wide
stops. What it costs is that each loss is several times the size of each
win, which means a high win rate on its own says nothing about
profitability. This is exactly why the ONLY defensible way to state a
target is jointly: "win rate X at profit factor > 1 after costs, over N
walk-forward trades".

The model's job changes accordingly, and becomes much more tractable: at
low pt/sl the classifier no longer has to predict direction, it has to
predict which of the many small-target trades will hit the rare, large
stop. That is a rarer, more structured event than "which way next" — and
it is where ML can genuinely add value.

This script produces the evidence table for choosing that geometry, using
the same purged, pooled walk-forward protocol as the main validator, so the
numbers are comparable and not in-sample.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_meta_ml import build_asset_labels, load_btc_regime
from src.calibration import precision_at_threshold_scorer
from src.gating import select_non_overlapping

PARAM_DIST = {
    'max_depth': [2, 3, 4],
    'n_estimators': [50, 100],
    'learning_rate': [0.03, 0.05],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [5, 10, 20],
    'reg_alpha': [0.1, 1.0],
    'reg_lambda': [3.0, 5.0],
}


def evaluate_geometry(data_files, signal, lookback, pt_mult, sl_mult, max_holding,
                      n_folds, btc_regime, cost_per_trade, top_quantile=0.9):
    """Pooled purged walk-forward for one (signal, pt, sl) combination.

    Returns base/filtered win rates plus profit factor, so win rate is never
    reported without the economics that make it meaningful.
    """
    pooled = []
    feature_cols = None
    for data_file in data_files:
        path = os.path.join(OUTPUT_DIR, data_file)
        if not os.path.exists(path):
            continue
        joined, feature_cols = build_asset_labels(
            path, lookback, pt_mult, sl_mult, max_holding,
            signal=signal, btc_regime=btc_regime)
        if joined is None or joined.empty:
            continue
        joined = joined.copy()
        joined['asset'] = data_file
        pooled.append(joined)

    if not pooled:
        return None

    pooled_df = pd.concat(pooled).sort_index()
    n = len(pooled_df)
    edges = [pooled_df.index[int(n * i / n_folds)] for i in range(n_folds)]
    edges.append(pooled_df.index[-1] + pd.Timedelta(seconds=1))
    purge = pd.Timedelta(hours=4 * max_holding)
    scorer = precision_at_threshold_scorer(quantile=0.8, min_support=10)

    oos = []
    for fold in range(1, n_folds):
        test_start, test_end = edges[fold], edges[fold + 1]
        train_fold = pooled_df[pooled_df.index < (test_start - purge)]
        test_fold = pooled_df[(pooled_df.index >= test_start) & (pooled_df.index < test_end)]
        if len(train_fold) < 150 or test_fold.empty:
            continue
        if train_fold['label'].nunique() < 2:
            continue

        search = RandomizedSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss',
                          objective='binary:logistic'),
            param_distributions=PARAM_DIST, n_iter=6, scoring=scorer,
            cv=3, random_state=42, n_jobs=-1)
        search.fit(train_fold[feature_cols], train_fold['label'])
        model = search.best_estimator_

        test_fold = test_fold.copy()
        test_fold['p_win'] = model.predict_proba(test_fold[feature_cols])[:, 1]
        oos.append(test_fold)

    if not oos:
        return None

    combined = pd.concat(oos)
    payoff = pt_mult / sl_mult
    breakeven = 1.0 / (1.0 + payoff)

    def econ(df):
        """Unit-sized net expectancy: no Kelly, no compounding, so
        geometries are compared on the raw edge rather than on sizing."""
        if df.empty:
            return float('nan'), float('nan')
        net = df['ret'].to_numpy() - cost_per_trade
        wins, losses = net[net > 0], net[net <= 0]
        pf = (wins.sum() / abs(losses.sum())
              if len(losses) and losses.sum() != 0
              else (float('inf') if len(wins) else 0.0))
        return float(pf), float(net.mean())

    base_all = select_non_overlapping(combined, np.ones(len(combined), bool))
    thr = combined['p_win'].quantile(top_quantile)
    top = select_non_overlapping(combined, (combined['p_win'] >= thr).to_numpy())

    pf_base, exp_base = econ(base_all)
    pf_top, exp_top = econ(top)

    return {
        'signal': signal, 'pt': pt_mult, 'sl': sl_mult, 'payoff': payoff,
        'breakeven_wr': breakeven,
        'n_candidates': len(combined),
        'base_n': len(base_all),
        'base_wr': float((base_all['label'] == 1).mean()) if len(base_all) else float('nan'),
        'base_pf': pf_base, 'base_exp': exp_base,
        'top_n': len(top),
        'top_wr': float((top['label'] == 1).mean()) if len(top) else float('nan'),
        'top_pf': pf_top, 'top_exp': exp_top,
        'corr': float(combined['p_win'].corr(combined['label'])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep barrier geometry to find where a high win rate is actually reachable.")
    parser.add_argument("--data", type=str, nargs='+', required=True)
    parser.add_argument("--btc-regime-file", type=str, default=None)
    parser.add_argument("--signals", type=str, nargs='+', default=["breakout", "reversion"])
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--max-holding", type=int, default=18)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--top-quantile", type=float, default=0.9,
                        help="Model-filtered slice to report (0.9 = top 10% by confidence)")
    parser.add_argument("--fee-pct", type=float, default=0.001)
    parser.add_argument("--slippage-pct", type=float, default=0.001)
    parser.add_argument("--geometries", type=str, nargs='+',
                        default=["2.0:2.0", "1.0:1.0", "1.0:2.0", "0.5:2.0",
                                 "0.5:3.0", "0.33:3.0", "0.25:3.0"],
                        help="pt:sl multiples to test")
    parser.add_argument("--out", type=str, default="barrier_sweep.csv")
    args = parser.parse_args()

    cost = 2 * (args.fee_pct + args.slippage_pct)
    btc_regime = load_btc_regime(args.btc_regime_file)

    print("Barrier-geometry sweep — purged pooled walk-forward per combination.")
    print(f"Assets: {len(args.data)}, folds: {args.n_folds}, "
          f"round-trip cost: {cost:.2%}\n")

    results = []
    for signal in args.signals:
        for geom in args.geometries:
            pt, sl = (float(x) for x in geom.split(':'))
            res = evaluate_geometry(
                args.data, signal, args.lookback, pt, sl, args.max_holding,
                args.n_folds, btc_regime, cost, args.top_quantile)
            if res is None:
                print(f"  {signal:<10} pt/sl={pt}/{sl}: insufficient data")
                continue
            results.append(res)
            print(f"  {signal:<10} pt/sl={pt}/{sl} (payoff {res['payoff']:.2f}, "
                  f"breakeven WR {res['breakeven_wr']:.1%}): "
                  f"base WR {res['base_wr']:.1%} (n={res['base_n']}, PF {res['base_pf']:.2f}) | "
                  f"top-{(1 - args.top_quantile) * 100:.0f}% WR {res['top_wr']:.1%} "
                  f"(n={res['top_n']}, PF {res['top_pf']:.2f}) | corr {res['corr']:+.3f}")

    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, args.out)
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 78}")
    print("  Ranked by top-slice profit factor (profitability first, win rate second)")
    print(f"{'=' * 78}")
    ranked = df.sort_values('top_pf', ascending=False)
    cols = ['signal', 'pt', 'sl', 'breakeven_wr', 'base_wr', 'top_wr', 'top_n',
            'top_pf', 'top_exp', 'corr']
    print(ranked[cols].to_string(index=False))

    print(f"\n  Saved to {out_path}")
    print("\n  Reading this table: a high `top_wr` is only meaningful when")
    print("  `top_wr` > `breakeven_wr` AND `top_pf` > 1 AND `top_n` is large")
    print("  enough to be more than an anecdote. A geometry with a 90% win")
    print("  rate and an 83% breakeven is a much thinner edge than one with a")
    print("  55% win rate and a 33% breakeven.")


if __name__ == "__main__":
    main()

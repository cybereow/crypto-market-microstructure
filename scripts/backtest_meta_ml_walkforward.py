"""Walk-forward validation of the meta-labeled strategy: instead of one
static 80/20 train/test split (which only tells you whether the edge held
up over one particular recent stretch), retrain periodically on an
expanding window of pooled data and evaluate strictly on the next chunk of
calendar time, rolling forward. This is what actually matters for whether
an edge is real and durable versus a fluke of which period happened to
land in the test set.

Each retrain is purged: any candidate trade whose entry falls within
`max_holding` bars of the test window's start is dropped from that
retrain's training data, since its triple-barrier outcome could otherwise
peek past the boundary.
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_meta_ml import build_asset_labels


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
    args = parser.parse_args()

    pooled = []
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
            funding_path=funding_path, signal=args.signal,
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

    all_oos = []
    print(f"Pooled: {n} candidate trades across {len(pooled)} assets, {args.n_folds} folds.\n")

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
        n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

        base_model = XGBClassifier(random_state=42, eval_metric='logloss',
                                    objective='binary:logistic', scale_pos_weight=scale_pos_weight)
        search = RandomizedSearchCV(base_model, param_distributions=param_distributions,
                                     n_iter=8, scoring='f1', cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        test_fold = test_fold.copy()
        test_fold['p_win'] = model.predict_proba(test_fold[feature_cols])[:, 1]
        all_oos.append(test_fold)

        base_rate = test_fold['label'].mean()
        top20 = test_fold[test_fold['p_win'] >= test_fold['p_win'].quantile(0.8)]
        print(f"Fold {fold} ({test_start.date()} to {test_end.date()}): "
              f"train={len(train_fold)}, test={len(test_fold)}, "
              f"base win rate={base_rate:.1%}, top-20% win rate={top20['label'].mean():.1%} (n={len(top20)})")

    if not all_oos:
        print("\nNo folds produced results.")
        return

    combined = pd.concat(all_oos)
    print(f"\n{'=' * 55}\n  Combined walk-forward OOS ({len(combined)} trades)\n{'=' * 55}")
    print(f"  Overall win rate:            {combined['label'].mean():>8.1%}")
    for q in [0.9, 0.8, 0.7]:
        thr = combined['p_win'].quantile(q)
        top = combined[combined['p_win'] >= thr]
        print(f"  Top {(1 - q) * 100:>2.0f}% by confidence:      {top['label'].mean():>8.1%}  (n={len(top)})")
    corr = combined['p_win'].corr(combined['label'])
    print(f"  corr(p_win, label):          {corr:>8.3f}")
    print("=" * 55)


if __name__ == "__main__":
    main()

import argparse
import os
import sys

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data['ret_1d'] = data['close'].pct_change()
    data['ret_2d'] = data['close'].pct_change(2)
    data['ret_3d'] = data['close'].pct_change(3)
    data['ret_5d'] = data['close'].pct_change(5)

    # Trend: MACD (normalized)
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD_12_26_9'] = (ema_12 - ema_26) / (data['close'] + 1e-9)
    data['MACDs_12_26_9'] = data['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    data['MACDh_12_26_9'] = data['MACD_12_26_9'] - data['MACDs_12_26_9']

    # Momentum: RSI
    delta = data['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    data['RSI_14'] = 100 - (100 / (1 + rs))

    ema_up_w = up.ewm(com=69, adjust=False).mean()
    ema_down_w = down.ewm(com=69, adjust=False).mean()
    rs_w = ema_up_w / (ema_down_w + 1e-9)
    data['RSI_70'] = 100 - (100 / (1 + rs_w))

    # Volatility: ATR (normalized) and Bollinger Bands
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR_14'] = true_range.rolling(14).mean() / (data['close'] + 1e-9)

    sma_20 = data['close'].rolling(window=20).mean()
    std_20 = data['close'].rolling(window=20).std()
    bbu_20 = sma_20 + (std_20 * 2)
    bbl_20 = sma_20 - (std_20 * 2)
    data['bb_width'] = (bbu_20 - bbl_20) / (sma_20 + 1e-9)
    data['bb_pct'] = (data['close'] - bbl_20) / (bbu_20 - bbl_20 + 1e-9)
    # bb_position normalizes the close price against the Bollinger Bands range [-1, 1]
    data['bb_position'] = (data['close'] - sma_20) / (std_20 * 2 + 1e-9)

    # Moving Average Distance
    sma_10 = data['close'].rolling(window=10).mean()
    sma_50 = data['close'].rolling(window=50).mean()
    data['sma_dist'] = sma_10 / sma_50 - 1

    # Volatility Ratio
    data['ATR_50'] = true_range.rolling(50).mean() / (data['close'] + 1e-9)
    data['ATR_ratio'] = data['ATR_14'] / (data['ATR_50'] + 1e-9)

    # Realized volatility (log returns)
    log_ret = np.log(data['close'] / data['close'].shift(1))
    data['realized_vol_10'] = log_ret.rolling(10).std() * np.sqrt(365)
    data['realized_vol_30'] = log_ret.rolling(30).std() * np.sqrt(365)
    data['vol_regime'] = data['realized_vol_10'] / (data['realized_vol_30'] + 1e-9)

    # Mean reversion signals
    data['close_to_sma20'] = data['close'] / (sma_20 + 1e-9) - 1
    data['close_to_sma50'] = data['close'] / (sma_50 + 1e-9) - 1

    # Momentum features
    data['roc_10'] = data['close'].pct_change(10)
    data['roc_20'] = data['close'].pct_change(20)

    if 'volume' in data.columns:
        data['vol_change_1d'] = data['volume'].pct_change()
        data['vol_change_3d'] = data['volume'].pct_change(3)
        data['vol_sma_20'] = data['volume'].rolling(window=20).mean()
        data['vol_ratio'] = data['volume'] / (data['vol_sma_20'] + 1e-9)

        obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        data['OBV_sma_20'] = obv.rolling(window=20).mean()
        data['OBV_dist'] = (obv - data['OBV_sma_20']) / (data['OBV_sma_20'].replace(0, 1e-9))
        data['OBV_sma_70'] = obv.rolling(window=70).mean()
        data['OBV_trend'] = (obv - data['OBV_sma_70']) / (data['OBV_sma_70'].replace(0, 1e-9))

    if 'funding_rate' in data.columns:
        data['funding_rate_diff'] = data['funding_rate'].diff()
        data['funding_sma_5'] = data['funding_rate'].rolling(window=5).mean()

    if 'obi' in data.columns:
        data['obi_raw'] = data['obi']
        data['obi_imbalance'] = data['obi'] - 0.5
        data['obi_diff'] = data['obi'].diff()
        data['obi_sma_5'] = data['obi'].rolling(window=5).mean()

    # Day of week and hour
    data['dow'] = data.index.dayofweek
    data['hour'] = data.index.hour

    # Garman-Klass volatility
    log_hl = np.log(data['high'] / data['low'])
    log_co = np.log(data['close'] / data['open'])
    rs = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    data['garman_klass_vol'] = np.sqrt(rs)

    data.dropna(inplace=True)
    return data


def create_target(df: pd.DataFrame, threshold_pct: float = 0.005) -> pd.Series:
    """3-class target: 2=strong up, 1=flat/noise, 0=strong down."""
    future_ret = df['ret_1d'].shift(-1)
    target = pd.Series(1, index=df.index, name='target')  # default flat
    target[future_ret > threshold_pct] = 2
    target[future_ret < -threshold_pct] = 0
    return target


def purged_walk_forward_split(n_samples, n_splits=5, purge_gap=5, embargo_pct=0.01):
    """Walk-forward splits with purge gap AND embargo zone to kill leakage."""
    test_size = n_samples // (n_splits + 1)
    embargo = max(1, int(test_size * embargo_pct))
    splits = []
    for i in range(n_splits):
        train_end = test_size * (i + 1)
        test_start = train_end + purge_gap
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_end - embargo)
        test_idx = np.arange(test_start, min(test_end, n_samples))
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model for price direction prediction.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model-out", type=str, default="ml_model.json")
    parser.add_argument("--l2-obi-data", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.005, help="Return threshold for strong up/down (default 0.5%%)")
    args = parser.parse_args()

    data_path = os.path.join(OUTPUT_DIR, args.data)
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)

    print(f"Loading data from {args.data}.")
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

    if args.l2_obi_data:
        obi_path = os.path.join(OUTPUT_DIR, args.l2_obi_data)
        if os.path.exists(obi_path):
            print(f"Merging L2 OBI data from {args.l2_obi_data}.")
            obi_df = pd.read_csv(obi_path, index_col='timestamp', parse_dates=True)
            df = df.join(obi_df, how='left')
            df['obi'] = df['obi'].ffill().fillna(0.5)
        else:
            print(f"Warning: L2 OBI data file {obi_path} does not exist. Skipping.")

    print("Creating features.")
    df_features = create_features(df)

    # Auto-scale threshold for timeframe
    import re
    threshold = args.threshold
    tf_match = re.search(r'(\d+)(h|m)', args.data)
    if tf_match:
        val = int(tf_match.group(1))
        unit = tf_match.group(2)
        if unit == 'h':
            threshold = threshold * np.sqrt(val / 24)
        elif unit == 'm':
            threshold = threshold * np.sqrt(val / (24 * 60))

    # 3-class target
    df_features['target'] = create_target(df_features, threshold_pct=threshold)
    df_features.dropna(subset=['target'], inplace=True)

    # Drop rows where future return is NaN (last row)
    future_ret = df_features['ret_1d'].shift(-1)
    df_features = df_features[future_ret.notna()]

    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'ret_1d']
    features = [col for col in df_features.columns if col not in exclude_cols]

    X = df_features[features]
    y = df_features['target']

    train_size = int(len(X) * 0.8)
    X_train_full = X.iloc[:train_size]
    y_train_full = y.iloc[:train_size]

    # Class distribution
    class_counts = y_train_full.value_counts().sort_index()
    print(f"\nTarget distribution (in-sample):")
    print(f"  Down (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y_train_full):.1%})")
    print(f"  Flat (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y_train_full):.1%})")
    print(f"  Up   (2): {class_counts.get(2, 0)} ({class_counts.get(2, 0)/len(y_train_full):.1%})")

    # Feature selection on isolated early slice
    sel_train_size = int(len(X_train_full) * 0.5)
    X_sel_train = X_train_full.iloc[:sel_train_size]
    y_sel_train = y_train_full.iloc[:sel_train_size]

    print("\nRunning feature selection on isolated slice.")
    sel_model = XGBClassifier(
        random_state=42, eval_metric='mlogloss',
        num_class=3, objective='multi:softprob'
    )
    sel_model.fit(X_sel_train, y_sel_train)
    importance = sel_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

    top_features = feat_imp.head(10)['Feature'].tolist()
    print(f"Selected top 10 features: {top_features}")
    X_train_full = X_train_full[top_features]

    # Purged walk-forward CV
    splits = purged_walk_forward_split(len(X_train_full), n_splits=5, purge_gap=5)
    print(f"\nPurged walk-forward: {len(splits)} splits, {len(X_train_full)} samples.")

    # Class weights to handle imbalanced classes
    total = len(y_train_full)
    n_classes = 3
    class_weight_map = {}
    for c in range(n_classes):
        count = (y_train_full == c).sum()
        if count > 0:
            class_weight_map[c] = total / (n_classes * count)
        else:
            class_weight_map[c] = 1.0
    sample_weights = y_train_full.map(class_weight_map).values

    param_distributions = {
        'max_depth': [2, 3, 4],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.8],
        'min_child_weight': [5, 10, 20],
        'reg_alpha': [0.1, 1.0, 5.0],
        'reg_lambda': [3.0, 5.0, 10.0],
        'gamma': [0, 0.1, 0.5],
    }

    base_model = XGBClassifier(
        random_state=42, eval_metric='mlogloss',
        num_class=3, objective='multi:softprob'
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=15,
        scoring='f1_macro',
        cv=splits,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train_full, y_train_full, sample_weight=sample_weights)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV F1-macro: {search.best_score_:.4f}")

    model = search.best_estimator_

    # Evaluate on last CV fold
    if splits:
        last_train, last_test = splits[-1]
        y_pred = model.predict(X_train_full.iloc[last_test])
        print(f"\nLast fold classification report:")
        print(classification_report(y_train_full.iloc[last_test], y_pred, target_names=['Down', 'Flat', 'Up']))

    features_path = os.path.join(OUTPUT_DIR, "ml_features.txt")
    with open(features_path, 'w') as f:
        f.write(','.join(top_features))

    # Save threshold so backtest uses the same one
    threshold_path = os.path.join(OUTPUT_DIR, "ml_threshold.txt")
    with open(threshold_path, 'w') as f:
        f.write(str(threshold))

    model_path = os.path.join(OUTPUT_DIR, args.model_out)
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates technical indicators as features for ML using pandas-ta."""
    data = df.copy()

    # Core Returns
    data['ret_1d'] = data['close'].pct_change()
    data['ret_3d'] = data['close'].pct_change(3)

    # Advanced Indicators implemented with pure pandas
    # 1. Trend: MACD
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD_12_26_9'] = ema_12 - ema_26
    data['MACDs_12_26_9'] = data['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    data['MACDh_12_26_9'] = data['MACD_12_26_9'] - data['MACDs_12_26_9']

    # 2. Momentum: RSI
    delta = data['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # 3. Volatility: ATR and Bollinger Bands
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR_14'] = true_range.rolling(14).mean()

    sma_20 = data['close'].rolling(window=20).mean()
    std_20 = data['close'].rolling(window=20).std()
    data['BBU_20_2.0'] = sma_20 + (std_20 * 2)
    data['BBL_20_2.0'] = sma_20 - (std_20 * 2)
    data['BBM_20_2.0'] = sma_20

    # Moving Average Distance
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_dist'] = data['sma_10'] / data['sma_50'] - 1

    # Target: 1 if next day's return is positive, 0 otherwise
    data['target'] = (data['ret_1d'].shift(-1) > 0).astype(int)

    # Drop rows with NaN due to rolling windows and shift
    data.dropna(inplace=True)

    # Remove 'ret_1d' from features to prevent noise/overfitting (today's return rarely predicts tomorrow's direction)
    # We keep it in the dataframe to construct targets and backtest, but exclude it later
    return data

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model for price direction prediction.")
    parser.add_argument("--data", type=str, required=True, help="Filename of the asset CSV (e.g. kraken_BTC_USDT_1d.csv)")
    parser.add_argument("--model-out", type=str, default="ml_model.json", help="Filename to save the trained model")
    args = parser.parse_args()

    data_path = os.path.join(OUTPUT_DIR, args.data)
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

    print("Creating advanced features...")
    df_features = create_features(df)

    # Extract all created features (excluding price data, current bar's return, and target)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'ret_1d']
    features = [col for col in df_features.columns if col not in exclude_cols]

    X = df_features[features]
    y = df_features['target']

    # Walk-Forward Validation using TimeSeriesSplit on the first 80% of data (In-Sample)
    # The last 20% is strictly held out for backtesting (Out-of-Sample)
    train_size = int(len(X) * 0.8)
    X_train_full = X.iloc[:train_size]
    y_train_full = y.iloc[:train_size]

    tscv = TimeSeriesSplit(n_splits=5)

    base_model = XGBClassifier(random_state=42, eval_metric='logloss')

    param_distributions = {
        'max_depth': [3, 4, 5, 7],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    print(f"Running RandomizedSearchCV with TimeSeriesSplit on {len(X_train_full)} samples...")
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=15,
        scoring='accuracy',
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train_full, y_train_full)

    print(f"\nBest params found: {search.best_params_}")
    print(f"Best CV Accuracy: {search.best_score_:.2%}")

    model = search.best_estimator_

    # Feature Importance Logging
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    print("\nTop 5 Feature Importances:")
    print(feat_imp.head(5).to_string(index=False))

    model_path = os.path.join(OUTPUT_DIR, args.model_out)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()

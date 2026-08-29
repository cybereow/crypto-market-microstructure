import argparse
import os
import sys
import pickle

import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates technical indicators as features for ML using pandas-ta."""
    data = df.copy()

    # Core Returns
    data['ret_1d'] = data['close'].pct_change()
    data['ret_3d'] = data['close'].pct_change(3)

    # Advanced Indicators via pandas-ta
    # 1. Trend: MACD
    macd = data.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None:
        data = pd.concat([data, macd], axis=1)

    # 2. Momentum: RSI
    data['rsi'] = data.ta.rsi(length=14)

    # 3. Volatility: ATR and Bollinger Bands
    data['atr'] = data.ta.atr(length=14)
    bbands = data.ta.bbands(length=20, std=2)
    if bbands is not None:
        data = pd.concat([data, bbands], axis=1)

    # Moving Average Distance
    data['sma_10'] = data.ta.sma(length=10)
    data['sma_50'] = data.ta.sma(length=50)
    data['sma_dist'] = data['sma_10'] / data['sma_50'] - 1

    # Target: 1 if next day's return is positive, 0 otherwise
    data['target'] = (data['ret_1d'].shift(-1) > 0).astype(int)

    # Drop rows with NaN due to rolling windows and shift
    data.dropna(inplace=True)

    return data

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model for price direction prediction.")
    parser.add_argument("--data", type=str, required=True, help="Filename of the asset CSV (e.g. kraken_BTC_USDT_1d.csv)")
    parser.add_argument("--model-out", type=str, default="ml_model.pkl", help="Filename to save the trained model")
    args = parser.parse_args()

    data_path = os.path.join(OUTPUT_DIR, args.data)
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

    print("Creating advanced features with pandas-ta...")
    df_features = create_features(df)

    # Extract all created features (excluding price data and target)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
    features = [col for col in df_features.columns if col not in exclude_cols]

    X = df_features[features]
    y = df_features['target']

    # Time-series split (no shuffling to prevent data leakage)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training XGBoost Classifier on {len(X_train)} samples with {len(features)} features...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(OUTPUT_DIR, args.model_out)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()

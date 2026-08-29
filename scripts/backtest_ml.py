import argparse
import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from src.strategies.ml_strategy import MLTradingStrategy

def main():
    parser = argparse.ArgumentParser(description="Backtest ML Trading Strategy.")
    parser.add_argument("--data", type=str, required=True, help="Filename of the asset CSV (e.g. kraken_BTC_USDT_1d.csv)")
    parser.add_argument("--model", type=str, default="ml_model.json", help="Filename of the trained model")
    args = parser.parse_args()

    data_path = os.path.join(OUTPUT_DIR, args.data)
    model_path = os.path.join(OUTPUT_DIR, args.model)

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist. Train the model first using scripts/train_ml.py")
        sys.exit(1)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

    print("Creating advanced features...")
    df_features = create_features(df)

    # Slice the out-of-sample data FIRST to prevent any data leakage
    split_idx = int(len(df_features) * 0.8)
    oos_df = df_features.iloc[split_idx:].copy()

    # Ensure same feature extraction logic as training
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'ret_1d']
    features = [col for col in oos_df.columns if col not in exclude_cols]

    X_oos = oos_df[features]

    print(f"Loading XGBoost model from {args.model}...")
    model = XGBClassifier()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    strategy = MLTradingStrategy(model)
    signals = strategy.generate_signals(X_oos)
    oos_results = strategy.calculate_returns(oos_df, signals)

    # Recalculate cumulative return for out-of-sample period
    oos_results['cum_ret'] = (1 + oos_results['strat_ret']).cumprod()

    total_return = oos_results['cum_ret'].iloc[-1] - 1
    buy_hold_return = (1 + oos_results['ret_1d']).cumprod().iloc[-1] - 1

    win_rate = (oos_results['strat_ret'] > 0).sum() / (oos_results['strat_ret'] != 0).sum() if (oos_results['strat_ret'] != 0).sum() > 0 else 0

    # Calculate Risk Metrics
    daily_rf = 0.0
    sharpe = np.sqrt(365) * (oos_results['strat_ret'].mean() - daily_rf) / (oos_results['strat_ret'].std() + 1e-9)
    roll_max = oos_results['cum_ret'].cummax()
    drawdown = oos_results['cum_ret'] / roll_max - 1.0
    max_dd = drawdown.min()

    print("-" * 40)
    print("Out-of-Sample Results (with 0.1% fees):")
    print(f"Strategy Return: {total_return:.2%}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"Win Rate (on active trades): {win_rate:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()

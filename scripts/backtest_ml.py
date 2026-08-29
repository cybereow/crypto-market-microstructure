import argparse
import os
import sys
import pickle

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from src.strategies.ml_strategy import MLTradingStrategy

def main():
    parser = argparse.ArgumentParser(description="Backtest ML Trading Strategy.")
    parser.add_argument("--data", type=str, required=True, help="Filename of the asset CSV (e.g. kraken_BTC_USDT_1d.csv)")
    parser.add_argument("--model", type=str, default="ml_model.pkl", help="Filename of the trained model")
    args = parser.parse_args()

    data_path = os.path.join(OUTPUT_DIR, args.data)
    model_path = os.path.join(OUTPUT_DIR, args.model)

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist. Train the model first.")
        sys.exit(1)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

    print("Creating advanced features with pandas-ta...")
    df_features = create_features(df)

    # Ensure same feature extraction logic as training
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
    features = [col for col in df_features.columns if col not in exclude_cols]

    X = df_features[features]

    print(f"Loading XGBoost model from {args.model}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    strategy = MLTradingStrategy(model)
    signals = strategy.generate_signals(X)
    results = strategy.calculate_returns(df_features, signals)

    # We evaluate on the out-of-sample data (last 20%) to be realistic
    split_idx = int(len(results) * 0.8)
    oos_results = results.iloc[split_idx:].copy()

    # Recalculate cumulative return for out-of-sample period
    oos_results['cum_ret'] = (1 + oos_results['strat_ret']).cumprod()

    total_return = oos_results['cum_ret'].iloc[-1] - 1
    buy_hold_return = (1 + oos_results['ret_1d']).cumprod().iloc[-1] - 1

    win_rate = (oos_results['strat_ret'] > 0).sum() / (oos_results['strat_ret'] != 0).sum() if (oos_results['strat_ret'] != 0).sum() > 0 else 0

    print("-" * 40)
    print("Out-of-Sample Results:")
    print(f"Strategy Return: {total_return:.2%}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"Win Rate (on active trades): {win_rate:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()

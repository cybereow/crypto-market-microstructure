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
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="ml_model.json")
    parser.add_argument("--confidence", type=float, default=0.40, help="Confidence threshold for taking positions")
    args = parser.parse_args()

    data_path = os.path.join(OUTPUT_DIR, args.data)
    model_path = os.path.join(OUTPUT_DIR, args.model)

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist. Train first.")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    df_features = create_features(df)

    split_idx = int(len(df_features) * 0.8)
    oos_df = df_features.iloc[split_idx:].copy()

    features_path = os.path.join(OUTPUT_DIR, "ml_features.txt")
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            top_features = [feat.strip() for feat in f.read().split(',') if feat.strip()]
        print(f"Loaded {len(top_features)} selected features.")
    else:
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'ret_1d']
        top_features = [col for col in oos_df.columns if col not in exclude_cols]

    X_oos = oos_df[top_features]

    model = XGBClassifier()
    model.load_model(model_path)

    strategy = MLTradingStrategy(model)
    signals = strategy.generate_signals(X_oos, confidence_threshold=args.confidence, close_series=oos_df['close'])
    oos_results = strategy.calculate_returns(oos_df, signals)
    oos_results['cum_ret'] = (1 + oos_results['strat_ret']).cumprod()

    total_return = oos_results['cum_ret'].iloc[-1] - 1
    buy_hold_return = (1 + oos_results['ret_1d']).cumprod().iloc[-1] - 1

    active = oos_results['strat_ret'] != 0
    win_rate = (oos_results.loc[active, 'strat_ret'] > 0).mean() if active.sum() > 0 else 0

    daily_rf = 0.0
    diffs = oos_results.index.to_series().diff().dropna()
    if len(diffs) > 0:
        median_diff = diffs.median()
        periods_per_year = int(pd.Timedelta(days=365) / median_diff)
    else:
        periods_per_year = 365

    sharpe = np.sqrt(periods_per_year) * (oos_results['strat_ret'].mean() - daily_rf) / (oos_results['strat_ret'].std() + 1e-9)

    roll_max = oos_results['cum_ret'].cummax()
    drawdown = oos_results['cum_ret'] / roll_max - 1.0
    max_dd = drawdown.min()

    # Calmar ratio
    n_days = len(oos_results)
    annual_ret = (1 + total_return) ** (365 / max(n_days, 1)) - 1
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

    # Profit factor
    gross_profit = oos_results.loc[oos_results['strat_ret'] > 0, 'strat_ret'].sum()
    gross_loss = abs(oos_results.loc[oos_results['strat_ret'] < 0, 'strat_ret'].sum())
    profit_factor = gross_profit / (gross_loss + 1e-9)

    # Position stats
    long_bars = (oos_results['position'] > 0).sum()
    short_bars = (oos_results['position'] < 0).sum()
    flat_bars = (oos_results['position'] == 0).sum()
    total_bars = len(oos_results)

    # Trade count (position changes)
    trades = (oos_results['position'].diff().fillna(0) != 0).sum()

    print("=" * 50)
    print("  ML Strategy — Out-of-Sample Results")
    print("=" * 50)
    print(f"  Strategy Return:    {total_return:>10.2%}")
    print(f"  Buy & Hold Return:  {buy_hold_return:>10.2%}")
    print(f"  Alpha:              {total_return - buy_hold_return:>10.2%}")
    print("-" * 50)
    print(f"  Sharpe Ratio:       {sharpe:>10.2f}")
    print(f"  Calmar Ratio:       {calmar:>10.2f}")
    print(f"  Max Drawdown:       {max_dd:>10.2%}")
    print(f"  Profit Factor:      {profit_factor:>10.2f}")
    print(f"  Win Rate:           {win_rate:>10.2%}")
    print("-" * 50)
    print(f"  Total Trades:       {trades:>10d}")
    print(f"  Long Bars:          {long_bars:>10d} ({long_bars/total_bars:.0%})")
    print(f"  Short Bars:         {short_bars:>10d} ({short_bars/total_bars:.0%})")
    print(f"  Flat Bars:          {flat_bars:>10d} ({flat_bars/total_bars:.0%})")
    print("=" * 50)


if __name__ == "__main__":
    main()

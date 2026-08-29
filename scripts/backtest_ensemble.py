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
from src.strategies.grid_trading import GridTradingStrategy

def main():
    parser = argparse.ArgumentParser(description="Backtest Regime-Aware Ensemble Strategy (ML + Grid).")
    parser.add_argument("--data", type=str, required=True, help="Filename of the asset CSV (e.g. kraken_BTC_USDT_1d.csv)")
    parser.add_argument("--model", type=str, default="ml_model.json", help="Filename of the trained ML model")
    parser.add_argument("--adx-threshold", type=float, default=25.0, help="ADX threshold to determine trend vs range regime")
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

    # 1. Regime Classifier: ADX (Average Directional Index)
    # Using pure Pandas implementation of ADX
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    minus_dm = pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()

    plus_di = 100 * (plus_dm / atr)
    minus_di = 100 * (minus_dm / atr)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    # Define Regime: 1 = Trending (ML), 0 = Ranging (Grid)
    regime = (adx > args.adx_threshold).astype(int)

    # Generate ML Signals
    print("Creating advanced features for ML strategy...")
    df_features = create_features(df)

    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'ret_1d']
    features = [col for col in df_features.columns if col not in exclude_cols]

    X = df_features[features]

    print(f"Loading XGBoost model from {args.model}...")
    model = XGBClassifier()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    ml_strategy = MLTradingStrategy(model)
    ml_signals = ml_strategy.generate_signals(X)

    # 3. Simulate Grid Strategy Equity Curve
    print("Running Grid Backtest for regime mapping...")
    grid_strategy = GridTradingStrategy(num_grids=10, grid_range_pct=0.2, adaptive_atr_period=14, initial_capital=10000)
    # We only care about returns for blending
    grid_results = grid_strategy.backtest(df)

    # Align Data Indices
    common_idx = df.index.intersection(X.index).intersection(grid_results['equity_curve'].index)
    df_combined = df.loc[common_idx].copy()
    regime_aligned = regime.loc[common_idx]

    # Align Returns
    df_combined['ret_1d'] = df_combined['close'].pct_change()
    ml_returns_df = ml_strategy.calculate_returns(df_combined, ml_signals.loc[common_idx])
    ml_strat_returns = ml_returns_df['strat_ret']

    grid_strat_returns = grid_results['equity_curve'].loc[common_idx]['return'].fillna(0)

    # Ensemble Allocation:
    # If Regime == 1 (Trend) -> Allocation = 100% ML
    # If Regime == 0 (Range) -> Allocation = 100% Grid

    # Ensure signal doesn't look ahead: shift regime by 1 day
    regime_signal = regime_aligned.shift(1).fillna(0)

    ensemble_returns = np.where(regime_signal == 1, ml_strat_returns, grid_strat_returns)
    df_combined['ensemble_ret'] = ensemble_returns
    df_combined['cum_ret'] = (1 + df_combined['ensemble_ret']).cumprod()

    total_return = df_combined['cum_ret'].iloc[-1] - 1
    buy_hold_return = (df_combined['close'].iloc[-1] / df_combined['close'].iloc[0]) - 1

    daily_rf = 0.0
    sharpe = np.sqrt(365) * (df_combined['ensemble_ret'].mean() - daily_rf) / (df_combined['ensemble_ret'].std() + 1e-9)
    roll_max = df_combined['cum_ret'].cummax()
    drawdown = df_combined['cum_ret'] / roll_max - 1.0
    max_dd = drawdown.min()

    print("-" * 40)
    print("Regime-Aware Ensemble Results:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()

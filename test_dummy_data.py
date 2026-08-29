import pandas as pd
import numpy as np
import os
sys = __import__('sys')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

# Create a dummy CSV for backtest_ensemble to run and crash on ret_1d KeyError
os.makedirs(OUTPUT_DIR, exist_ok=True)
dates = pd.date_range("2024-01-01", periods=100, freq='D')
df = pd.DataFrame({
    'timestamp': dates,
    'open': np.random.uniform(100, 110, 100),
    'high': np.random.uniform(105, 115, 100),
    'low': np.random.uniform(95, 105, 100),
    'close': np.random.uniform(100, 110, 100),
    'volume': np.random.uniform(1000, 5000, 100)
})
df.to_csv(os.path.join(OUTPUT_DIR, "dummy_BTC_USDT_1d.csv"), index=False)

# Make a dummy XGBoost model using the train_ml script

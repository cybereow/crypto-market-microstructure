import pandas as pd
import numpy as np

# Create dummy flat market data
df = pd.DataFrame({
    'high': [100, 100, 100, 100, 100],
    'low': [100, 100, 100, 100, 100],
    'close': [100, 100, 100, 100, 100]
}, index=pd.date_range("2024-01-01", periods=5))

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

# THIS will cause NaN division
dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
print(dx)

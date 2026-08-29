import pandas as pd
import numpy as np

# Create dummy trend up market
df = pd.DataFrame({
    'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
})

delta = df['close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)

ema_up = up.ewm(com=13, adjust=False).mean()
ema_down = down.ewm(com=13, adjust=False).mean()
rs = ema_up / ema_down
rsi = 100 - (100 / (1 + rs))

print("EMA Down values:")
print(ema_down.tail())
print("RS values:")
print(rs.tail())
print("RSI values:")
print(rsi.tail())

import pandas as pd

# Simulating two chunks of data with very different sizes
# Chunk 1: 2 trades, highly bid-heavy (OBI = 0.9)
chunk1 = pd.DataFrame({'timestamp': [pd.Timestamp('2024-01-01 10:00:00')]*2, 'bid_qty': [90, 90], 'ask_qty': [10, 10]})
chunk1['obi'] = chunk1['bid_qty'] / (chunk1['bid_qty'] + chunk1['ask_qty'])

# Chunk 2: 1000 trades, slightly ask-heavy (OBI = 0.4)
chunk2 = pd.DataFrame({'timestamp': [pd.Timestamp('2024-01-01 10:00:00')]*1000, 'bid_qty': [40]*1000, 'ask_qty': [60]*1000})
chunk2['obi'] = chunk2['bid_qty'] / (chunk2['bid_qty'] + chunk2['ask_qty'])

print("Chunk 1 Mean OBI:", chunk1['obi'].mean())
print("Chunk 2 Mean OBI:", chunk2['obi'].mean())

# The current scripts/download_l2_obi.py logic:
resampled_c1 = chunk1.set_index('timestamp').resample('1h').mean()
resampled_c2 = chunk2.set_index('timestamp').resample('1h').mean()
combined = pd.concat([resampled_c1, resampled_c2])
final = combined.groupby(combined.index).mean()
print("Final OBI using script logic (mean of means):", final['obi'].iloc[0])

# The mathematically correct logic:
all_raw = pd.concat([chunk1, chunk2])
all_resampled_correct = all_raw.set_index('timestamp').resample('1h').mean()
print("True Mathematical Mean OBI:", all_resampled_correct['obi'].iloc[0])

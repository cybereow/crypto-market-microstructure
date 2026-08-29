import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.strategies.pairs_trading import PairsTradingStrategy

# Create dummy spread data
# We simulate z-scores to trigger a stop-loss, then revert, then re-enter
# T0: Z=1 (Normal)
# T1: Z=5 (Stop-loss triggers!)
# T2: Z=0 (Reverts to mean, stopped_out should clear AND it should not trade)
# T3: Z=3 (Entry condition met again! It SHOULD enter, but let's see if it does)
df = pd.DataFrame({
    'S1': [100, 105, 100, 103],
    'S2': [100, 100, 100, 100],
    'spread': [0, 5, 0, 3],
    'zscore': [1, 5, 0, 3] # We hack the zscore later
})

class HackedPairsTradingStrategy(PairsTradingStrategy):
    def generate_signals(self, s1, s2, mock_zscores):
        df = pd.DataFrame({'S1': s1, 'S2': s2})
        zscores = mock_zscores

        pos_s1 = np.zeros(len(zscores))
        pos_s2 = np.zeros(len(zscores))
        current_pos = 0
        stopped_out = False

        for i in range(len(zscores)):
            z = zscores[i]

            if stopped_out:
                if abs(z) < self.z_exit_threshold:
                    stopped_out = False
                current_pos = 0
            else:
                if abs(z) > self.z_stop_loss:
                    current_pos = 0
                    stopped_out = True
                elif z > self.z_entry_threshold:
                    current_pos = -1
                elif z < -self.z_entry_threshold:
                    current_pos = 1
                elif abs(z) < self.z_exit_threshold:
                    current_pos = 0

            if current_pos == 1:
                pos_s1[i] = 1
                pos_s2[i] = -1
            elif current_pos == -1:
                pos_s1[i] = -1
                pos_s2[i] = 1

        df['position_s1'] = pos_s1
        df['position_s2'] = pos_s2
        df['position_s1'] = df['position_s1'].shift(1).fillna(0)
        return df

strategy = HackedPairsTradingStrategy(z_entry_threshold=2.0, z_exit_threshold=0.5, z_stop_loss=4.0)
signals = strategy.generate_signals(df['S1'], df['S2'], df['zscore'].values)
print("Input Z-Scores:")
print(df['zscore'].values)
print("Generated Positions (Shifted):")
print(signals['position_s1'].values)

# What happened at T3? Z=3 is entry condition! But let's check unshifted positions
unshifted_pos = np.zeros(4)
current_pos = 0
stopped_out = False
for i, z in enumerate(df['zscore'].values):
    if stopped_out:
        if abs(z) < 0.5:
            stopped_out = False
        current_pos = 0
    else:
        if abs(z) > 4.0:
            current_pos = 0
            stopped_out = True
        elif z > 2.0:
            current_pos = -1
        elif z < -2.0:
            current_pos = 1
        elif abs(z) < 0.5:
            current_pos = 0
    unshifted_pos[i] = current_pos
print("Unshifted Positions:")
print(unshifted_pos)

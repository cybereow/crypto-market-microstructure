import pandas as pd
import numpy as np
import statsmodels.api as sm

class PairsTradingStrategy:
    def __init__(self, z_entry_threshold=2.0, z_exit_threshold=0.5, window=30):
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        self.window = window

    def generate_signals(self, s1: pd.Series, s2: pd.Series) -> pd.DataFrame:
        """
        Generates trading signals for a cointegrated pair (S1, S2) using a rolling z-score of the spread.
        Returns a DataFrame with 'zscore', 'position_s1', 'position_s2'
        """
        df = pd.concat([s1, s2], axis=1).dropna()
        df.columns = ['S1', 'S2']

        # Calculate hedge ratio using rolling OLS
        # In practice, rolling OLS can be slow, for simplicity we use a rolling ratio or a static OLS over the window
        # For this implementation, we calculate rolling spread = S1 - (beta * S2) where beta is rolling mean of S1/S2
        # A more rigorous approach uses rolling OLS, but rolling ratio is faster for simple backtests.
        df['ratio'] = df['S1'] / df['S2']
        rolling_mean = df['ratio'].rolling(window=self.window).mean()
        rolling_std = df['ratio'].rolling(window=self.window).std()

        # Z-score of the ratio
        df['zscore'] = (df['ratio'] - rolling_mean) / rolling_std

        # Signals
        df['position_s1'] = 0
        df['position_s2'] = 0

        # We hold positions
        current_pos = 0 # 1 means long spread (Long S2, Short S1), -1 means short spread (Long S1, Short S2)

        pos_s1 = []
        pos_s2 = []

        for z in df['zscore']:
            if np.isnan(z):
                pos_s1.append(0)
                pos_s2.append(0)
                continue

            if z > self.z_entry_threshold:
                # Ratio is too high -> S1 is overpriced relative to S2
                # Short S1, Long S2
                current_pos = -1
            elif z < -self.z_entry_threshold:
                # Ratio is too low -> S1 is underpriced relative to S2
                # Long S1, Short S2
                current_pos = 1
            elif abs(z) < self.z_exit_threshold:
                # Reverted to mean
                current_pos = 0

            if current_pos == 1:
                pos_s1.append(1)
                pos_s2.append(-1)
            elif current_pos == -1:
                pos_s1.append(-1)
                pos_s2.append(1)
            else:
                pos_s1.append(0)
                pos_s2.append(0)

        df['position_s1'] = pos_s1
        df['position_s2'] = pos_s2

        # Shift positions by 1 so we trade on the NEXT bar after signal
        df['position_s1'] = df['position_s1'].shift(1).fillna(0)
        df['position_s2'] = df['position_s2'].shift(1).fillna(0)

        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates strategy returns given the price and position dataframe.
        """
        df['ret_s1'] = df['S1'].pct_change()
        df['ret_s2'] = df['S2'].pct_change()

        df['strat_ret'] = (df['position_s1'] * df['ret_s1']) + (df['position_s2'] * df['ret_s2'])
        df['cum_ret'] = (1 + df['strat_ret']).cumprod()

        return df

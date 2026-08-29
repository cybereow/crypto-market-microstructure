import pandas as pd
import numpy as np

class PairsTradingStrategy:
    def __init__(self, z_entry_threshold=2.0, z_exit_threshold=0.5, window=30, fee_pct=0.001, slippage_pct=0.001):
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        self.window = window
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def generate_signals(self, s1: pd.Series, s2: pd.Series) -> pd.DataFrame:
        """
        Generates trading signals for a cointegrated pair (S1, S2) using a rolling z-score of the spread.
        Returns a DataFrame with 'zscore', 'position_s1', 'position_s2'
        """
        df = pd.concat([s1, s2], axis=1).dropna()
        df.columns = ['S1', 'S2']

        # Correct Spread Calculation: S1 - (beta * S2)
        # We calculate the rolling hedge ratio (beta) using rolling covariance / rolling variance
        rolling_cov = df['S1'].rolling(window=self.window).cov(df['S2'])
        rolling_var = df['S2'].rolling(window=self.window).var()

        df['beta'] = rolling_cov / rolling_var
        df['spread'] = df['S1'] - (df['beta'] * df['S2'])

        rolling_spread_mean = df['spread'].rolling(window=self.window).mean()
        rolling_spread_std = df['spread'].rolling(window=self.window).std()

        # Z-score of the spread
        df['zscore'] = (df['spread'] - rolling_spread_mean) / rolling_spread_std

        # Vectorized Signals
        zscores = df['zscore'].values
        pos_s1 = np.zeros(len(zscores))
        pos_s2 = np.zeros(len(zscores))

        current_pos = 0

        for i in range(len(zscores)):
            z = zscores[i]
            if np.isnan(z):
                pos_s1[i] = 0
                pos_s2[i] = 0
                continue

            if z > self.z_entry_threshold:
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

        # Shift positions by 1 so we trade on the NEXT bar after signal
        df['position_s1'] = df['position_s1'].shift(1).fillna(0)
        df['position_s2'] = df['position_s2'].shift(1).fillna(0)

        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates strategy returns given the price and position dataframe.
        """
        df = df.copy()
        df['ret_s1'] = df['S1'].pct_change()
        df['ret_s2'] = df['S2'].pct_change()

        # Calculate when trades happen
        df['trade_s1'] = df['position_s1'].diff().fillna(0).abs()
        df['trade_s2'] = df['position_s2'].diff().fillna(0).abs()

        # Apply fees and slippage on trades
        # Both fee and slippage reduce our net return whenever we flip positions
        cost_per_trade = self.fee_pct + self.slippage_pct

        gross_ret = (df['position_s1'] * df['ret_s1']) + (df['position_s2'] * df['ret_s2'])
        costs = (df['trade_s1'] * cost_per_trade) + (df['trade_s2'] * cost_per_trade)

        df['strat_ret'] = gross_ret - costs
        df['cum_ret'] = (1 + df['strat_ret']).cumprod()

        return df

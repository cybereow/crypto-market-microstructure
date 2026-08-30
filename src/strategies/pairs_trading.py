import pandas as pd
import numpy as np

class PairsTradingStrategy:
    def __init__(self, z_entry_threshold=2.0, z_exit_threshold=0.5, z_stop_loss=4.0, window=30, fee_pct=0.001, slippage_pct=0.001, delta=1e-5, vt=1e-3):
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        self.z_stop_loss = z_stop_loss
        self.window = window
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.delta = delta
        self.vt = vt

    def generate_signals(self, s1: pd.Series, s2: pd.Series) -> pd.DataFrame:
        """
        Generates trading signals for a cointegrated pair (S1, S2) using a rolling z-score of the spread.
        Returns a DataFrame with 'zscore', 'position_s1', 'position_s2'
        """
        df = pd.concat([s1, s2], axis=1).dropna()
        df.columns = ['S1', 'S2']

        # State-Space Kalman Filter for dynamic Hedge Ratio (beta)
        # We assume S1 = beta * S2 + error
        # Beta is a random walk state
        wt = self.delta / (1 - self.delta) * np.eye(2) # Process noise covariance
        theta = np.zeros((2, 1)) # State vector [alpha, beta]
        P = np.eye(2) * 1000 # State covariance matrix (High initial uncertainty for fast convergence)
        R = None # Expected variance of measurement

        betas = np.zeros(len(df))

        S1_vals = df['S1'].values
        S2_vals = df['S2'].values

        for i in range(len(df)):
            x = np.array([[1], [S2_vals[i]]])
            y = S1_vals[i]

            # Predict
            # theta(t|t-1) = theta(t-1|t-1)
            R = P + wt

            # Measurement prediction
            yhat = x.T.dot(theta)[0, 0]

            # Measurement variance
            Q = x.T.dot(R).dot(x)[0, 0] + self.vt

            # Error
            e = y - yhat

            # Kalman gain
            K = R.dot(x) / Q

            # Update state
            theta = theta + K * e

            # Update covariance
            P = R - K.dot(x.T).dot(R)

            betas[i] = theta[1, 0]

        df['beta'] = betas

        # We use the raw instantaneous Kalman beta to eliminate lag completely.
        # This allows the strategy to react immediately to structural breaks.
        df['spread'] = df['S1'] - (df['beta'] * df['S2'])

        # Adaptive z-score window based on Ornstein-Uhlenbeck half-life
        # OLS regression: Δspread ~ spread_lag
        spread_lag = df['spread'].shift(1)
        spread_diff = df['spread'].diff()

        # Calculate half life over a rolling window or expanding window.
        # For performance, we'll calculate static half-life over the whole series and use it as the window
        # (Alternatively, could do a rolling half-life, but static is standard for adaptive window sizing)

        # Dropna for regression
        reg_df = pd.DataFrame({'y': spread_diff, 'x': spread_lag}).dropna()
        if len(reg_df) > 10:
            x = reg_df['x'].values
            y = reg_df['y'].values

            # Simple linear regression slope
            cov = np.cov(x, y)[0, 1]
            var = np.var(x)
            beta_hl = cov / (var + 1e-9)

            if beta_hl < 0:
                half_life = -np.log(2) / beta_hl
                window = max(10, int(half_life * 2))
            else:
                window = self.window
        else:
            window = self.window

        print(f"Adaptive Window calculated as: {window} (fallback: {self.window})")

        rolling_spread_mean = df['spread'].rolling(window=window).mean()
        rolling_spread_std = df['spread'].rolling(window=window).std()

        # Z-score of the spread
        df['zscore'] = (df['spread'] - rolling_spread_mean) / rolling_spread_std

        # Vectorized Signals
        zscores = df['zscore'].values
        pos_s1 = np.zeros(len(zscores))
        pos_s2 = np.zeros(len(zscores))

        current_pos = 0
        stopped_out = False

        for i in range(len(zscores)):
            z = zscores[i]
            if np.isnan(z):
                pos_s1[i] = 0
                pos_s2[i] = 0
                continue

            # State Machine for Stop-Loss
            # If we are stopped out, we wait until z-score reverts to the mean (inside exit threshold)
            if stopped_out:
                if abs(z) < self.z_exit_threshold:
                    stopped_out = False  # Reset state, ready to trade again
                else:
                    # Still stopped out, do not take positions
                    current_pos = 0

            # Use 'if not stopped_out:' instead of 'else:' so we can evaluate entry conditions
            # on the exact same bar that stopped_out becomes False
            if not stopped_out:
                # Normal trading logic
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

import pandas as pd
import numpy as np
from scipy.stats import norm


class MLTradingStrategy:
    def __init__(self, model, fee_pct=0.001, slippage_pct=0.001,
                 max_position=1.0, stop_loss_pct=0.03, max_daily_dd=0.05,
                 zscore_window=500):
        self.model = model
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.max_daily_dd = max_daily_dd
        # Trailing window (in bars) used to standardize predicted
        # probabilities before thresholding — see note in the 2-class
        # branch of generate_signals for why this is necessary.
        self.zscore_window = zscore_window

    def generate_signals(self, X: pd.DataFrame, confidence_threshold=0.55, close_series=None) -> pd.DataFrame:
        """
        Returns DataFrame with 'position' (direction) and 'size' (Kelly fraction).
        3-class model: 0=down, 1=flat, 2=up.
        """
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
            n_classes = probs.shape[1]

            position = pd.Series(0.0, index=X.index, name='position')
            size = pd.Series(0.0, index=X.index, name='size')

            if close_series is not None:
                sma_50 = close_series.rolling(50).mean()

            if n_classes == 3:
                prob_up = probs[:, 2]
                prob_down = probs[:, 0]

                for i in range(len(X)):
                    pu, pd_val = prob_up[i], prob_down[i]
                    if pu >= confidence_threshold and pu > pd_val:
                        position.iloc[i] = 1
                        # Half-Kelly: (p * b - q) / b, capped, then halved
                        edge = pu - (1 - pu)
                        size.iloc[i] = min(np.clip(edge / 2, 0.0, 1.0), self.max_position)
                    elif pd_val >= confidence_threshold and pd_val > pu:
                        position.iloc[i] = -1
                        edge = pd_val - (1 - pd_val)
                        size.iloc[i] = min(np.clip(edge / 2, 0.0, 1.0), self.max_position)

                        # Trend Filter for Shorts
                        if close_series is not None and close_series.iloc[i] >= sma_50.iloc[i]:
                            position.iloc[i] = 0
                            size.iloc[i] = 0.0
            else:
                # 2-class fallback.
                #
                # A regularized XGBoost trained on noisy 4h crypto returns
                # produces well-*ranked* but tightly compressed probabilities
                # (e.g. spanning ~0.47-0.53 instead of the full 0-1 range) —
                # that compression is a symptom of a real, weak edge, not a
                # bug, and forcing the model to spread out (less
                # regularization) would just mean overfitting noise instead.
                # A fixed threshold like 0.55 on raw probability is then
                # either always-on or (as here) never reached, so we instead
                # standardize prob_up against its own trailing distribution
                # and threshold the resulting z-score. confidence_threshold
                # keeps its 0.5-1.0 "probability-like" meaning via the
                # inverse-normal-CDF map to a z cutoff.
                min_size = 0.20
                prob_up = pd.Series(probs[:, 1], index=X.index)
                min_periods = max(30, self.zscore_window // 5)
                roll_mean = prob_up.rolling(self.zscore_window, min_periods=min_periods).mean()
                roll_std = prob_up.rolling(self.zscore_window, min_periods=min_periods).std()
                z = (prob_up - roll_mean) / roll_std.replace(0, np.nan)
                z_threshold = norm.ppf(np.clip(confidence_threshold, 0.5001, 0.9999))

                for i in range(len(X)):
                    zi = z.iloc[i]
                    if pd.isna(zi):
                        continue
                    if zi >= z_threshold:
                        position.iloc[i] = 1
                        size.iloc[i] = min(max(np.clip(zi / (2 * z_threshold), 0.0, 1.0), min_size), self.max_position)
                    elif zi <= -z_threshold:
                        position.iloc[i] = -1
                        size.iloc[i] = min(max(np.clip(-zi / (2 * z_threshold), 0.0, 1.0), min_size), self.max_position)

                        # Trend Filter for Shorts
                        if close_series is not None and close_series.iloc[i] >= sma_50.iloc[i]:
                            position.iloc[i] = 0
                            size.iloc[i] = 0.0
        else:
            predictions = self.model.predict(X)
            position = pd.Series(predictions, index=X.index, name='position')
            position = position.replace(0, -1)
            size = pd.Series(self.max_position, index=X.index, name='size')

        # Shift by 1 — trade on next bar
        position = position.shift(1).fillna(0)
        size = size.shift(1).fillna(0)

        signals = pd.DataFrame({'position': position, 'size': size}, index=X.index)
        return signals

    def calculate_returns(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data['position'] = signals['position']
        data['size'] = signals['size']
        data['weighted_pos'] = data['position'] * data['size']

        # Trade sizing: cost based on change in weighted position
        data['trade'] = data['weighted_pos'].diff().fillna(0).abs()
        cost_per_trade = self.fee_pct + self.slippage_pct
        data['strat_ret'] = (data['weighted_pos'] * data['ret_1d']) - (data['trade'] * cost_per_trade)

        # Apply trailing stop and daily drawdown limits iteratively
        stopped = False
        adjusted_rets = data['strat_ret'].copy()

        current_equity = 1.0
        peak_equity = 1.0

        # Track daily drawdown
        dates = data.index.date
        current_day = None
        day_start_equity = 1.0
        daily_stopped = False

        for i in range(len(data)):
            # Daily reset
            if dates[i] != current_day:
                current_day = dates[i]
                day_start_equity = current_equity
                daily_stopped = False

            # If stopped by trailing stop, wait for position change
            if stopped:
                if i > 0 and data['position'].iloc[i] != data['position'].iloc[i-1] and data['position'].iloc[i] != 0:
                    stopped = False
                    peak_equity = current_equity # Reset peak on new position
                else:
                    adjusted_rets.iloc[i] = 0

            # If daily stopped, stay flat for the rest of the day
            if daily_stopped:
                adjusted_rets.iloc[i] = 0

            # Update equity with (potentially zeroed) return
            ret = adjusted_rets.iloc[i]
            current_equity *= (1 + ret)

            # Check trailing stop (only if we have a position)
            if data['position'].iloc[i] != 0 and not stopped and not daily_stopped:
                if current_equity > peak_equity:
                    peak_equity = current_equity

                drawdown = (current_equity / peak_equity) - 1

                if drawdown < -self.stop_loss_pct:
                    stopped = True
                    # Stop fires: zero out current return, but add transaction cost
                    adjusted_rets.iloc[i] = 0
                    # Charge fee for exiting position
                    adjusted_rets.iloc[i] -= abs(data['weighted_pos'].iloc[i]) * cost_per_trade
                    # Recalculate equity for this step after fee
                    # Back out the previous current_equity calculation
                    current_equity = current_equity / (1 + ret)
                    current_equity *= (1 + adjusted_rets.iloc[i])

            # Check daily drawdown
            if not daily_stopped:
                daily_dd = (current_equity / day_start_equity) - 1
                if daily_dd < -self.max_daily_dd:
                    daily_stopped = True
                    # Exit immediately
                    if data['position'].iloc[i] != 0 and not stopped:
                        # If not already stopped out, charge fee for flattening here
                        adjusted_rets.iloc[i] -= abs(data['weighted_pos'].iloc[i]) * cost_per_trade
                        # Recalculate equity
                        current_equity = current_equity / (1 + ret)
                        current_equity *= (1 + adjusted_rets.iloc[i])

        data['strat_ret'] = adjusted_rets
        data['cum_ret'] = (1 + data['strat_ret']).cumprod()

        return data

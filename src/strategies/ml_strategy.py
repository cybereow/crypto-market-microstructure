import pandas as pd
import numpy as np


class MLTradingStrategy:
    def __init__(self, model, fee_pct=0.001, slippage_pct=0.001,
                 max_position=1.0, stop_loss_pct=0.03, max_daily_dd=0.05):
        self.model = model
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.max_daily_dd = max_daily_dd

    def generate_signals(self, X: pd.DataFrame, confidence_threshold=0.55) -> pd.DataFrame:
        """
        Returns DataFrame with 'position' (direction) and 'size' (Kelly fraction).
        3-class model: 0=down, 1=flat, 2=up.
        """
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
            n_classes = probs.shape[1]

            position = pd.Series(0.0, index=X.index, name='position')
            size = pd.Series(0.0, index=X.index, name='size')

            if n_classes == 3:
                prob_up = probs[:, 2]
                prob_down = probs[:, 0]
                prob_flat = probs[:, 1]

                for i in range(len(X)):
                    pu, pd_val, pf = prob_up[i], prob_down[i], prob_flat[i]
                    if pu >= confidence_threshold and pu > pd_val and pu > pf:
                        position.iloc[i] = 1
                        # Half-Kelly: (p * b - q) / b, capped, then halved
                        edge = pu - (1 - pu)
                        size.iloc[i] = min(np.clip(edge / 2, 0.1, 1.0), self.max_position)
                    elif pd_val >= confidence_threshold and pd_val > pu and pd_val > pf:
                        position.iloc[i] = -1
                        edge = pd_val - (1 - pd_val)
                        size.iloc[i] = min(np.clip(edge / 2, 0.1, 1.0), self.max_position)
            else:
                # 2-class fallback
                prob_up = probs[:, 1]
                for i in range(len(X)):
                    if prob_up[i] >= confidence_threshold:
                        position.iloc[i] = 1
                        edge = prob_up[i] - (1 - prob_up[i])
                        size.iloc[i] = min(np.clip(edge / 2, 0.1, 1.0), self.max_position)
                    elif prob_up[i] <= (1 - confidence_threshold):
                        position.iloc[i] = -1
                        edge = (1 - prob_up[i]) - prob_up[i]
                        size.iloc[i] = min(np.clip(edge / 2, 0.1, 1.0), self.max_position)
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

        # Stop loss: if cumulative return from entry drops below threshold, flatten
        data['cum_ret'] = (1 + data['strat_ret']).cumprod()
        peak = data['cum_ret'].expanding().max()
        drawdown = data['cum_ret'] / peak - 1

        # Apply trailing stop: zero out returns after stop triggered until position changes
        stopped = False
        stop_price_level = 0.0
        adjusted_rets = data['strat_ret'].copy()

        for i in range(1, len(data)):
            if stopped:
                if data['position'].iloc[i] != data['position'].iloc[i-1]:
                    stopped = False
                else:
                    adjusted_rets.iloc[i] = 0
                    continue

            if drawdown.iloc[i] < -self.stop_loss_pct:
                stopped = True
                adjusted_rets.iloc[i] = 0

        data['strat_ret'] = adjusted_rets
        data['cum_ret'] = (1 + data['strat_ret']).cumprod()

        return data

import pandas as pd

class MLTradingStrategy:
    def __init__(self, model, fee_pct=0.001, slippage_pct=0.001):
        self.model = model
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals using the provided ML model.
        Returns a Series where 1 means Long and 0 means no position.
        """
        # Predict returns 1 (up) or 0 (down)
        predictions = self.model.predict(X)

        # Position is 1 if prediction is up, -1 if down
        signals = pd.Series(predictions, index=X.index, name='position')
        signals = signals.replace(0, -1)

        # Shift signals by 1 so we trade on the NEXT bar after the prediction
        # (Since prediction uses data up to close of current bar, execution happens next open/close)
        signals = signals.shift(1).fillna(0)

        return signals

    def calculate_returns(self, df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Calculates strategy returns given the price dataframe and signals.
        """
        data = df.copy()
        data['position'] = signals
        # Trade occurs when position changes
        data['trade'] = data['position'].diff().fillna(0).abs()
        # Strategy return = position * asset return - (fee + slippage) * trade
        cost_per_trade = self.fee_pct + self.slippage_pct
        data['strat_ret'] = (data['position'] * data['ret_1d']) - (data['trade'] * cost_per_trade)
        data['cum_ret'] = (1 + data['strat_ret']).cumprod()

        return data

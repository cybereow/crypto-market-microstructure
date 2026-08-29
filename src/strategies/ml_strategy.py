import pandas as pd

class MLTradingStrategy:
    def __init__(self, model):
        self.model = model

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals using the provided ML model.
        Returns a Series where 1 means Long and 0 means no position.
        """
        # Predict returns 1 (up) or 0 (down)
        predictions = self.model.predict(X)

        # Position is 1 if prediction is up, else 0
        signals = pd.Series(predictions, index=X.index, name='position')

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
        data['strat_ret'] = data['position'] * data['ret_1d']
        data['cum_ret'] = (1 + data['strat_ret']).cumprod()

        return data

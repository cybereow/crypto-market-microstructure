from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, data):
        """data = dict of inputs (e.g. funding/candle history) -> signal list"""
        pass

    @abstractmethod
    def backtest(self, data, capital, fees):
        """data = dict of inputs -> backtest result dict"""
        pass

    def forward_test(self, data_new_period, capital=None, fees=None):
        """Out-of-sample check: same backtest, later/unseen data."""
        return self.backtest(data_new_period, capital, fees)

    # v0.3 -- not yet implemented
    def paper_trade(self, exchange):
        pass

    # v1 -- not yet implemented
    def execute(self, exchange, signal):
        pass

import pytest
import pandas as pd
import numpy as np

from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from src.strategies.ml_strategy import MLTradingStrategy

@pytest.fixture
def dummy_price_data():
    """Generates simple dummy price data."""
    dates = pd.date_range("2023-01-01", periods=10)
    # Price drops then goes up to test grid
    prices = [100.0, 90.0, 80.0, 70.0, 80.0, 90.0, 100.0, 110.0, 100.0, 105.0]
    return pd.DataFrame({'close': prices}, index=dates)

@pytest.fixture
def dummy_pair_data():
    """Generates dummy data for two highly correlated assets."""
    dates = pd.date_range("2023-01-01", periods=50)
    s1 = pd.Series(np.linspace(100, 150, 50) + np.random.normal(0, 1, 50), index=dates)
    s2 = pd.Series(np.linspace(100, 150, 50) + np.random.normal(0, 1, 50), index=dates)
    return s1, s2

class DummyModel:
    def predict(self, X):
        # Predicts 1 (up) for even rows, 0 (down) for odd rows
        return np.array([i % 2 for i in range(len(X))])

def test_grid_trading_execution(dummy_price_data):
    """Test if Grid Trading generates trades and calculates equity properly."""
    strategy = GridTradingStrategy(num_grids=5, grid_range_pct=0.4, initial_capital=1000, fee_pct=0.0)
    results = strategy.backtest(dummy_price_data)

    assert 'final_equity' in results
    assert results['num_trades'] > 0
    assert len(results['equity_curve']) == len(dummy_price_data)

def test_pairs_trading_signals(dummy_pair_data):
    """Test if Pairs Trading can handle data and generate positions."""
    s1, s2 = dummy_pair_data
    strategy = PairsTradingStrategy(z_entry_threshold=1.0, z_exit_threshold=0.1, window=10, fee_pct=0.0)
    signals = strategy.generate_signals(s1, s2)

    assert not signals.empty
    assert 'position_s1' in signals.columns
    assert 'position_s2' in signals.columns

    returns = strategy.calculate_returns(signals)
    assert 'strat_ret' in returns.columns
    assert 'cum_ret' in returns.columns

def test_ml_strategy_execution():
    """Test if ML strategy calculates returns correctly given a model prediction."""
    dates = pd.date_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        'ret_1d': [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.04, -0.02, 0.01]
    }, index=dates)

    model = DummyModel()
    strategy = MLTradingStrategy(model, fee_pct=0.0)
    signals = strategy.generate_signals(df)

    assert len(signals) == len(df)

    returns = strategy.calculate_returns(df, signals)
    assert 'strat_ret' in returns.columns
    assert 'trade' in returns.columns

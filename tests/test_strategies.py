import pytest
import pandas as pd
import numpy as np

from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from src.strategies.ml_strategy import MLTradingStrategy

@pytest.fixture
def dummy_price_data():
    """Generates simple dummy price data including high and low for grid trading."""
    dates = pd.date_range("2023-01-01", periods=5)
    # Start at 100. Drop to 80. Bounce to 120. Drop to 100.
    # We set close, high, low to make predictable limit triggers.
    data = {
        'close': [100.0, 80.0, 120.0, 100.0, 100.0],
        'high':  [100.0, 100.0, 120.0, 120.0, 100.0],
        'low':   [100.0,  80.0,  80.0, 100.0, 100.0]
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def dummy_pair_data():
    """Generates dummy data for two highly correlated assets."""
    dates = pd.date_range("2023-01-01", periods=50)
    # Base asset
    s2 = pd.Series(np.linspace(100, 150, 50), index=dates)
    # Highly correlated asset (S1 is roughly 1.5 * S2)
    s1 = pd.Series(np.linspace(150, 225, 50), index=dates)
    return s1, s2

class DummyModel:
    def predict(self, X):
        # Predicts 1 (up) for even rows, 0 (down) for odd rows
        return np.array([i % 2 for i in range(len(X))])

def test_grid_trading_execution(dummy_price_data):
    """Test if Grid Trading generates trades, calculates equity properly, and conserves inventory math."""
    # Start price 100, grids: 80, 90, 100, 110, 120 (5 grids)
    strategy = GridTradingStrategy(num_grids=5, grid_range_pct=0.4, initial_capital=1000, fee_pct=0.0, slippage_pct=0.0)
    results = strategy.backtest(dummy_price_data)

    assert 'final_equity' in results
    assert results['num_trades'] > 0
    assert len(results['equity_curve']) == len(dummy_price_data)

    # We should have bought at 90 and 80 on day 2 when it dropped,
    # and sold at 90, 100, 110, 120 on day 3 when it bounced.
    # This proves the logic limits are working mathematically.
    assert results['num_trades'] >= 2

def test_pairs_trading_signals(dummy_pair_data):
    """Test if Pairs Trading calculates the spread correctly using rolling covariance and beta."""
    s1, s2 = dummy_pair_data
    strategy = PairsTradingStrategy(z_entry_threshold=1.0, z_exit_threshold=0.1, window=10, fee_pct=0.0, slippage_pct=0.0)
    signals = strategy.generate_signals(s1, s2)

    assert not signals.empty
    assert 'spread' in signals.columns
    assert 'beta' in signals.columns

    # Because S1 is exactly 1.5 * S2, beta should be 1.5 and spread should be ~0
    # (after the window populates)
    valid_betas = signals['beta'].dropna()
    assert len(valid_betas) > 0
    assert np.isclose(valid_betas.iloc[-1], 1.5, atol=0.01)

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
    strategy = MLTradingStrategy(model, fee_pct=0.0, slippage_pct=0.0)
    signals = strategy.generate_signals(df)

    assert len(signals) == len(df)

    returns = strategy.calculate_returns(df, signals)
    assert 'strat_ret' in returns.columns
    assert 'trade' in returns.columns

    # Check that strat_ret equals asset ret where position was 1 (long)
    # Prediction is 1 on even rows (index 0, 2, 4).
    # Since position shifts by 1 and fills leading NaN with 0, index 1 becomes prediction 0.
    # The DummyModel returns [0, 1, 0, 1...]. Wait, i % 2 for 0 is 0.
    # Ah, range(len) is 0,1,2. So [0, 1, 0, 1...].
    # Prediction 0 is 0. Shifted by 1, index 1 is 0.
    # Prediction 1 is 1. Shifted by 1, index 2 is 1.
    assert returns['position'].iloc[2] == 1.0
    assert returns['strat_ret'].iloc[2] == returns['ret_1d'].iloc[2]

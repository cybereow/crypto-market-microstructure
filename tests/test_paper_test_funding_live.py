import pandas as pd
import pytest

from scripts.paper_test_funding_live import load_state, DATETIME_COLS, fetch_funding_series
import scripts.paper_test_funding_live as paper_test_funding_live


@pytest.fixture
def state_csv(tmp_path, monkeypatch):
    path = tmp_path / "paper_trades_funding.csv"
    monkeypatch.setattr(paper_test_funding_live, "STATE_PATH", str(path))
    return path


def test_load_state_datetime_columns_accept_microsecond_precision_assignment(state_csv):
    """Same regression this repo already hit for scripts/paper_test_live.py's
    load_state (see its own test) -- whole-second timestamps round-trip
    through CSV as datetime64[s], and assigning a microsecond-precision
    `now` into that column raises unless forced to a fixed (ns) resolution.
    """
    row = {
        'id': 'BTC/USD:USD_2026-08-31T00:00:00', 'asset': 'BTC/USD:USD', 'side': 1,
        'signal_time': '2026-08-31 00:00:00', 'signal_price': 100.0, 'atr': 1.0,
        'limit_price': 99.85, 'pt_price': 101.85, 'sl_price': 98.85,
        'deadline': '2026-08-31 12:00:00', 'status': 'pending_fill',
        'fill_time': '', 'fill_price': '', 'vertical_deadline': '',
        'close_time': '', 'close_price': '', 'ret': '',
    }
    pd.DataFrame([row]).to_csv(state_csv, index=False)

    state = load_state()
    for col in DATETIME_COLS:
        assert state[col].dtype == 'datetime64[ns]'

    microsecond_now = pd.Timestamp('2026-08-31 06:00:00.123456')
    state.loc[0, 'fill_time'] = microsecond_now
    assert state.loc[0, 'fill_time'] == microsecond_now


def test_load_state_missing_file_returns_empty_frame(state_csv):
    state = load_state()
    assert state.empty
    assert list(state.columns) == list(load_state().columns)


class _FakeExchange:
    def __init__(self, history):
        self._history = history

    def fetch_funding_rate_history(self, asset, limit=1000):
        return self._history


def test_fetch_funding_series_resamples_to_gapless_hourly():
    """Kraken's history can have gaps or sub-hourly noise in principle;
    the series handed to the signal must be a clean, forward-filled
    hourly index so it joins onto 4h OHLCV bars by exact timestamp the
    same way scripts/download_funding_vision.py's historical files do.
    """
    history = [
        {'timestamp': pd.Timestamp('2026-01-01 00:00:00').value // 10**6, 'fundingRate': 1e-5},
        {'timestamp': pd.Timestamp('2026-01-01 02:00:00').value // 10**6, 'fundingRate': 3e-5},
    ]
    exchange = _FakeExchange(history)
    series = fetch_funding_series(exchange, 'BTC/USD:USD')

    assert list(series.index) == [pd.Timestamp('2026-01-01 00:00:00'),
                                   pd.Timestamp('2026-01-01 01:00:00'),
                                   pd.Timestamp('2026-01-01 02:00:00')]
    assert series.iloc[1] == 1e-5  # the gap hour forward-filled from the prior reading


def test_fetch_funding_series_empty_history_returns_empty_series():
    exchange = _FakeExchange([])
    series = fetch_funding_series(exchange, 'BTC/USD:USD')
    assert series.empty

import pandas as pd
import pytest

from scripts.paper_test_live import load_state, DATETIME_COLS
import scripts.paper_test_live as paper_test_live


@pytest.fixture
def state_csv(tmp_path, monkeypatch):
    path = tmp_path / "paper_trades.csv"
    monkeypatch.setattr(paper_test_live, "STATE_PATH", str(path))
    return path


def test_load_state_datetime_columns_accept_microsecond_precision_assignment(state_csv):
    """Regression test: a real run hit `TypeError: Invalid value '...' for
    dtype 'datetime64[s]'` because whole-second signal/deadline timestamps
    round-tripped through CSV as datetime64[s], and then assigning a
    microsecond-precision `now` (from datetime.now()) into that column
    raised. load_state must force a fixed (ns) resolution so later
    assignments of finer-grained timestamps always succeed.
    """
    row = {
        'id': 'BTC/USDT_2026-08-31T00:00:00', 'asset': 'BTC/USDT', 'side': 1,
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

    # This is exactly the assignment that previously raised.
    microsecond_now = pd.Timestamp('2026-08-31 06:00:00.123456')
    state.loc[0, 'fill_time'] = microsecond_now
    assert state.loc[0, 'fill_time'] == microsecond_now


def test_load_state_missing_file_returns_empty_frame(state_csv):
    state = load_state()
    assert state.empty
    assert list(state.columns) == list(load_state().columns)

import sys

import pandas as pd
import pytest

import scripts.paper_test_llm as paper_test_llm
from scripts.paper_test_llm import load_state, DATETIME_COLS, STATE_COLUMNS


@pytest.fixture
def state_csv(tmp_path, monkeypatch):
    path = tmp_path / "paper_trades_llm.csv"
    monkeypatch.setattr(paper_test_llm, "STATE_PATH", str(path))
    return path


def test_load_state_datetime_columns_accept_microsecond_precision_assignment(state_csv):
    row = {
        'id': 'BTC/USDT_2026-08-31T00:00:00', 'asset': 'BTC/USDT', 'side': 1,
        'signal_time': '2026-08-31 00:00:00', 'signal_price': 100.0, 'atr': 1.0,
        'limit_price': 99.85, 'pt_price': 101.85, 'sl_price': 98.85,
        'deadline': '2026-08-31 12:00:00', 'status': 'pending_fill',
        'fill_time': '', 'fill_price': '', 'vertical_deadline': '',
        'close_time': '', 'close_price': '', 'ret': '',
        'llm_decision': 'approve', 'llm_confidence': 0.8, 'llm_reason': 'ok setup',
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
    assert list(state.columns) == STATE_COLUMNS


def test_state_columns_extend_base_with_llm_fields():
    assert STATE_COLUMNS[-3:] == ['llm_decision', 'llm_confidence', 'llm_reason']


class _FakeExchange:
    def __init__(self, last_price):
        self._last_price = last_price

    def fetch_ticker(self, asset):
        return {'last': self._last_price}


def _run_main(monkeypatch, assets, candidates, decisions, state_csv):
    """Drive main() with detect_new_signal/get_llm_decision/make_exchange/
    make_llm_client all faked out, so the gating logic (approve -> pending
    order, reject -> llm_rejected row, never a real network or API call)
    is exercised end to end.
    """
    monkeypatch.setattr(sys, "argv", ["paper_test_llm.py"])
    monkeypatch.setattr(paper_test_llm, "make_llm_client", lambda base_url=None: object())
    monkeypatch.setattr(paper_test_llm, "make_exchange", lambda: _FakeExchange(100.0))

    def fake_detect(exchange, asset, now):
        return candidates.get(asset, (None, None))
    monkeypatch.setattr(paper_test_llm, "detect_new_signal", fake_detect)

    def fake_get_llm_decision(client, model, asset, side, signal_price, atr, features):
        return decisions[asset]
    monkeypatch.setattr(paper_test_llm, "get_llm_decision", fake_get_llm_decision)
    monkeypatch.setattr(paper_test_llm, "ASSETS", assets)

    paper_test_llm.main()
    return pd.read_csv(state_csv)


def test_main_approved_candidate_becomes_pending_paper_trade(monkeypatch, state_csv):
    t = pd.Timestamp("2026-08-31 00:00:00")
    order = {
        'id': 'BTC/USDT_x', 'asset': 'BTC/USDT', 'side': 1, 'signal_time': t,
        'signal_price': 100.0, 'atr': 5.0, 'limit_price': 99.25, 'pt_price': 110.0,
        'sl_price': 95.0, 'deadline': t + pd.Timedelta(hours=12), 'status': 'pending_fill',
        'fill_time': pd.NaT, 'fill_price': float('nan'), 'vertical_deadline': pd.NaT,
        'close_time': pd.NaT, 'close_price': float('nan'), 'ret': float('nan'),
    }
    candidates = {'BTC/USDT': (order, {'RSI_14': 60.0})}
    decisions = {'BTC/USDT': {'decision': 'approve', 'confidence': 0.9, 'reason': 'clean breakout'}}

    state = _run_main(monkeypatch, ['BTC/USDT'], candidates, decisions, state_csv)

    assert len(state) == 1
    assert state.loc[0, 'status'] == 'pending_fill'
    assert state.loc[0, 'llm_decision'] == 'approve'


def test_main_rejected_candidate_never_becomes_a_paper_trade(monkeypatch, state_csv):
    t = pd.Timestamp("2026-08-31 00:00:00")
    order = {
        'id': 'ETH/USDT_x', 'asset': 'ETH/USDT', 'side': -1, 'signal_time': t,
        'signal_price': 100.0, 'atr': 5.0, 'limit_price': 100.75, 'pt_price': 90.0,
        'sl_price': 105.0, 'deadline': t + pd.Timedelta(hours=12), 'status': 'pending_fill',
        'fill_time': pd.NaT, 'fill_price': float('nan'), 'vertical_deadline': pd.NaT,
        'close_time': pd.NaT, 'close_price': float('nan'), 'ret': float('nan'),
    }
    candidates = {'ETH/USDT': (order, {'RSI_14': 45.0})}
    decisions = {'ETH/USDT': {'decision': 'reject', 'confidence': 0.7, 'reason': 'conflicting trend'}}

    state = _run_main(monkeypatch, ['ETH/USDT'], candidates, decisions, state_csv)

    assert len(state) == 1
    assert state.loc[0, 'status'] == 'llm_rejected'
    assert state.loc[0, 'llm_decision'] == 'reject'


def test_main_idempotent_does_not_re_ask_llm_for_already_logged_signal(monkeypatch, state_csv):
    t = pd.Timestamp("2026-08-31 00:00:00")
    order = {
        'id': 'SOL/USDT_x', 'asset': 'SOL/USDT', 'side': 1, 'signal_time': t,
        'signal_price': 100.0, 'atr': 5.0, 'limit_price': 99.25, 'pt_price': 110.0,
        'sl_price': 95.0, 'deadline': t + pd.Timedelta(hours=12), 'status': 'pending_fill',
        'fill_time': pd.NaT, 'fill_price': float('nan'), 'vertical_deadline': pd.NaT,
        'close_time': pd.NaT, 'close_price': float('nan'), 'ret': float('nan'),
    }
    candidates = {'SOL/USDT': (order, {'RSI_14': 60.0})}
    call_count = {'n': 0}

    monkeypatch.setattr(sys, "argv", ["paper_test_llm.py"])
    monkeypatch.setattr(paper_test_llm, "make_llm_client", lambda base_url=None: object())
    monkeypatch.setattr(paper_test_llm, "make_exchange", lambda: _FakeExchange(100.0))
    monkeypatch.setattr(paper_test_llm, "detect_new_signal",
                         lambda exchange, asset, now: candidates.get(asset, (None, None)))

    def counting_decision(client, model, asset, side, signal_price, atr, features):
        call_count['n'] += 1
        return {'decision': 'approve', 'confidence': 0.9, 'reason': 'ok'}
    monkeypatch.setattr(paper_test_llm, "get_llm_decision", counting_decision)
    monkeypatch.setattr(paper_test_llm, "ASSETS", ['SOL/USDT'])

    paper_test_llm.main()
    paper_test_llm.main()

    state = pd.read_csv(state_csv)
    assert len(state) == 1
    assert call_count['n'] == 1

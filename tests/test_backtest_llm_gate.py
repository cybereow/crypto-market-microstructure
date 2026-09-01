import sys

import pandas as pd
import pytest

import scripts.backtest_llm_gate as backtest_llm_gate
from scripts.backtest_llm_gate import load_cache, save_cache, CACHE_COLUMNS


def test_load_cache_missing_file_returns_empty_frame(tmp_path):
    cache = load_cache(str(tmp_path / "nope.csv"))
    assert cache.empty
    assert list(cache.columns) == CACHE_COLUMNS


def test_save_and_load_cache_round_trip(tmp_path):
    path = str(tmp_path / "cache.csv")
    df = pd.DataFrame([
        {'asset': 'a.csv', 'signal_time': pd.Timestamp('2024-01-01'), 'decision': 'approve',
         'confidence': 0.8, 'reason': 'ok'},
    ])
    save_cache(df, path)
    loaded = load_cache(path)
    assert loaded.loc[0, 'decision'] == 'approve'
    assert loaded.loc[0, 'signal_time'] == pd.Timestamp('2024-01-01')


def _fake_market():
    idx = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
    market = pd.DataFrame({
        'side': [1, -1, 1, -1],
        'label': [1, 0, 1, 0],
        'ret': [0.02, -0.01, 0.015, -0.008],
        'hold': [3, 3, 3, 3],
        'entry_pos': [0, 1, 2, 3],
        'exit_pos': [3, 4, 5, 6],
        'asset': ['x.csv'] * 4,
        'signal_price': [100.0, 101.0, 102.0, 103.0],
        'atr': [2.0, 2.0, 2.0, 2.0],
    }, index=idx)
    features = {('x.csv', ts): {'RSI_14': 50.0} for ts in idx}
    return market, features


def _patch_common(monkeypatch, tmp_path, argv):
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(backtest_llm_gate, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(backtest_llm_gate.anthropic, "Anthropic", lambda api_key: object())
    monkeypatch.setattr(backtest_llm_gate, "gather_candidates", lambda data_files: _fake_market())


def test_main_caches_decisions_and_writes_results(monkeypatch, tmp_path, capsys):
    _patch_common(monkeypatch, tmp_path, ["backtest_llm_gate.py", "--data", "x.csv"])

    def fake_get_llm_decision(client, model, asset, side, signal_price, atr, features):
        # Approve longs, reject shorts -- a clean, testable split.
        decision = 'approve' if side > 0 else 'reject'
        return {'decision': decision, 'confidence': 0.75, 'reason': f'side={side}'}
    monkeypatch.setattr(backtest_llm_gate, "get_llm_decision", fake_get_llm_decision)

    backtest_llm_gate.main()

    cache = pd.read_csv(tmp_path / "llm_gate_backtest_cache.csv")
    assert len(cache) == 4
    assert set(cache['decision']) == {'approve', 'reject'}

    results = pd.read_csv(tmp_path / "llm_gate_backtest_results.csv")
    assert len(results) == 4
    assert (results.loc[results['side'] == 1, 'llm_decision'] == 'approve').all()
    assert (results.loc[results['side'] == -1, 'llm_decision'] == 'reject').all()

    out = capsys.readouterr().out
    assert "Permutation test" in out


def test_main_idempotent_does_not_recall_llm_for_cached_candidates(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, ["backtest_llm_gate.py", "--data", "x.csv"])

    call_count = {'n': 0}

    def counting_decision(client, model, asset, side, signal_price, atr, features):
        call_count['n'] += 1
        return {'decision': 'approve', 'confidence': 0.9, 'reason': 'ok'}
    monkeypatch.setattr(backtest_llm_gate, "get_llm_decision", counting_decision)

    backtest_llm_gate.main()
    backtest_llm_gate.main()

    assert call_count['n'] == 4


def test_main_respects_limit(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, ["backtest_llm_gate.py", "--data", "x.csv", "--limit", "2"])

    call_count = {'n': 0}

    def counting_decision(client, model, asset, side, signal_price, atr, features):
        call_count['n'] += 1
        return {'decision': 'approve', 'confidence': 0.9, 'reason': 'ok'}
    monkeypatch.setattr(backtest_llm_gate, "get_llm_decision", counting_decision)

    backtest_llm_gate.main()
    assert call_count['n'] == 2


def test_main_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["backtest_llm_gate.py", "--data", "x.csv"])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(SystemExit):
        backtest_llm_gate.main()

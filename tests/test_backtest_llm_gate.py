import sys

import numpy as np
import pandas as pd
import pytest

import scripts.backtest_llm_gate as backtest_llm_gate
from scripts.backtest_llm_gate import (load_cache, save_cache, CACHE_COLUMNS,
                                       maker_fill_economics, gather_candidates)


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


def _fake_per_asset():
    """A small, self-consistent OHLC series (10 daily bars) with vol_breakout-shaped
    entries only at the four signal dates _fake_market() also labels, so
    maker_fill_economics has something real (if synthetic) to simulate fills against.
    """
    idx = pd.date_range('2024-01-01', periods=10, freq='D')
    close = pd.Series([100, 101, 100.5, 103, 104, 103.5, 106, 105, 108, 107],
                      index=idx, dtype=float)
    df_features = pd.DataFrame({'close': close, 'high': close + 1.0, 'low': close - 1.0})
    raw_atr = pd.Series(2.0, index=idx)
    entries = pd.Series(0, index=idx)
    entries.loc[idx[0]] = 1
    entries.loc[idx[1]] = -1
    entries.loc[idx[2]] = 1
    entries.loc[idx[3]] = -1
    return {'df_features': df_features, 'raw_atr': raw_atr, 'entries': entries}


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
        'signal_price': [100.0, 101.0, 100.5, 103.0],
        'atr': [2.0, 2.0, 2.0, 2.0],
    }, index=idx)
    features = {('x.csv', ts): {'RSI_14': 50.0} for ts in idx}
    per_asset = {'x.csv': _fake_per_asset()}
    return market, features, per_asset


def _patch_common(monkeypatch, tmp_path, argv):
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(backtest_llm_gate, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(backtest_llm_gate.anthropic, "Anthropic",
                        lambda api_key, base_url=None: object())
    monkeypatch.setattr(backtest_llm_gate, "gather_candidates",
                        lambda data_files, **kwargs: _fake_market())


def test_main_caches_decisions_and_writes_results(monkeypatch, tmp_path, capsys):
    _patch_common(monkeypatch, tmp_path, ["backtest_llm_gate.py", "--data", "x.csv"])

    def fake_get_llm_decision(client, model, asset, side, signal_price, atr, features, max_tokens=1024, signal_name=None):
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
    assert "Realistic maker-fill execution" in out


def test_main_idempotent_does_not_recall_llm_for_cached_candidates(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, ["backtest_llm_gate.py", "--data", "x.csv"])

    call_count = {'n': 0}

    def counting_decision(client, model, asset, side, signal_price, atr, features, max_tokens=1024, signal_name=None):
        call_count['n'] += 1
        return {'decision': 'approve', 'confidence': 0.9, 'reason': 'ok'}
    monkeypatch.setattr(backtest_llm_gate, "get_llm_decision", counting_decision)

    backtest_llm_gate.main()
    backtest_llm_gate.main()

    assert call_count['n'] == 4


def test_main_respects_limit(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, ["backtest_llm_gate.py", "--data", "x.csv", "--limit", "2"])

    call_count = {'n': 0}

    def counting_decision(client, model, asset, side, signal_price, atr, features, max_tokens=1024, signal_name=None):
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


def test_main_min_confidence_filters_low_confidence_approvals(monkeypatch, tmp_path, capsys):
    """Two of the four candidates are 'approve' but below --min-confidence; only the
    high-confidence approval should count toward n_approved in the printed summary.
    """
    _patch_common(monkeypatch, tmp_path,
                  ["backtest_llm_gate.py", "--data", "x.csv", "--min-confidence", "0.7"])

    # side=1 candidates get high confidence, side=-1 get low confidence -- both 'approve'.
    def fake_get_llm_decision(client, model, asset, side, signal_price, atr, features, max_tokens=1024, signal_name=None):
        confidence = 0.9 if side > 0 else 0.3
        return {'decision': 'approve', 'confidence': confidence, 'reason': 'x'}
    monkeypatch.setattr(backtest_llm_gate, "get_llm_decision", fake_get_llm_decision)

    backtest_llm_gate.main()

    out = capsys.readouterr().out
    assert "2 approved at confidence >= 0.70" in out


def test_maker_fill_economics_no_approved_candidates_returns_zero_filled():
    approved = pd.DataFrame(columns=['asset', 'side'])
    result = maker_fill_economics(approved, {}, maker_cost=0.0008)
    assert result['n_filled'] == 0
    assert result['n_candidates'] == 0


def test_maker_fill_economics_runs_against_synthetic_history():
    _, _, per_asset = _fake_market()
    idx = pd.to_datetime(['2024-01-01', '2024-01-03'])
    approved = pd.DataFrame({'asset': ['x.csv', 'x.csv'], 'side': [1, 1]}, index=idx)

    result = maker_fill_economics(approved, per_asset, maker_cost=0.0008)

    assert result['n_candidates'] == 2
    assert 0.0 <= result['fill_rate'] <= 1.0


def test_gather_candidates_raises_on_mismatched_funding_data_length():
    with pytest.raises(ValueError):
        gather_candidates(['a.csv', 'b.csv'], signal='funding_reversion',
                          funding_data_files=['only_one.csv'])


def _write_funding_reversion_fixture(tmp_path):
    """80 bars of near-flat OHLCV with one funding-rate spike at bar 75 --
    enough history for create_features' longest rolling window (50-bar
    SMA) to warm up before the spike, and enough for a short --lookback
    quantile window to treat the spike as extreme.
    """
    n = 80
    idx = pd.date_range('2024-01-01', periods=n, freq='4h')
    close = pd.Series(100.0 + np.linspace(0, 1, n), index=idx)
    ohlcv = pd.DataFrame({
        'open': close, 'high': close + 0.5, 'low': close - 0.5, 'close': close,
        'volume': 100.0,
    }, index=idx)
    funding = pd.Series(0.0001, index=idx)
    funding.iloc[75] = 0.05  # extreme positive -> fade with a short
    funding_df = pd.DataFrame({'funding_rate': funding}, index=idx)

    data_path = tmp_path / "asset.csv"
    funding_path = tmp_path / "funding.csv"
    ohlcv.to_csv(data_path, index_label='timestamp')
    funding_df.to_csv(funding_path, index_label='timestamp')
    return data_path.name, funding_path.name


def test_gather_candidates_joins_funding_data_and_finds_a_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest_llm_gate, "OUTPUT_DIR", str(tmp_path))
    data_file, funding_file = _write_funding_reversion_fixture(tmp_path)

    market, features_by_key, per_asset = gather_candidates(
        [data_file], signal='funding_reversion', lookback=5,
        funding_data_files=[funding_file])

    assert not market.empty
    assert (market['side'] == -1).any()  # extreme positive funding -> fade short
    key = (data_file, market.index[0])
    assert key in features_by_key
    assert features_by_key[key]['funding_rate'] is not None

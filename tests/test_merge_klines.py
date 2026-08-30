"""Tests for scripts/merge_klines.py.

The safety value of this script is entirely in its refusals, so those are
tested at least as carefully as the happy path. A merge tool that silently
splices a mismatched series is worse than no merge tool at all.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Loaded by path because scripts/ is not an importable package.
_spec = importlib.util.spec_from_file_location(
    "merge_klines", os.path.join(ROOT, "scripts", "merge_klines.py"))
merge_klines = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_klines)


def _frame(start, periods, base=100.0, freq='4h'):
    idx = pd.date_range(start, periods=periods, freq=freq)
    close = base + np.arange(periods, dtype=float)
    return pd.DataFrame({
        'timestamp': idx,
        'open': close, 'high': close + 1.0, 'low': close - 1.0,
        'close': close, 'volume': np.full(periods, 10.0),
    })


def _write(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_merges_only_bars_newer_than_existing(tmp_path):
    """The overlapping region must be preserved, not duplicated or replaced."""
    old = _frame('2026-01-01', 100)
    # New file overlaps the last 20 bars and extends 30 beyond.
    new = _frame('2026-01-01', 130).iloc[80:]

    old_p = _write(tmp_path, 'old.csv', old)
    new_p = _write(tmp_path, 'new.csv', new)

    res = merge_klines.merge_one(old_p, new_p, tolerance=0.001,
                                 dry_run=False, backup=False)
    assert res['status'] == 'merged'
    assert res['added'] == 30

    merged = pd.read_csv(old_p, parse_dates=['timestamp'])
    assert len(merged) == 130
    assert not merged['timestamp'].duplicated().any()
    assert merged['timestamp'].is_monotonic_increasing


def test_refuses_when_overlap_prices_disagree(tmp_path):
    """A different instrument (e.g. perp vs spot) must be rejected, not spliced."""
    old = _frame('2026-01-01', 100, base=100.0)
    new = _frame('2026-01-01', 130, base=140.0).iloc[80:]  # ~40% higher series

    old_p = _write(tmp_path, 'old.csv', old)
    new_p = _write(tmp_path, 'new.csv', new)

    res = merge_klines.merge_one(old_p, new_p, tolerance=0.001,
                                 dry_run=False, backup=False)
    assert res['status'] == 'refused_mismatch'
    assert res['added'] == 0
    # The original file must be untouched.
    assert len(pd.read_csv(old_p)) == 100


def test_refuses_when_there_is_no_overlap_to_verify(tmp_path):
    """Without shared bars there is no evidence the series match; refuse."""
    old = _frame('2026-01-01', 50)
    new = _frame('2026-06-01', 50, base=500.0)

    old_p = _write(tmp_path, 'old.csv', old)
    new_p = _write(tmp_path, 'new.csv', new)

    res = merge_klines.merge_one(old_p, new_p, tolerance=0.001,
                                 dry_run=False, backup=False)
    assert res['status'] == 'refused_no_overlap'
    assert len(pd.read_csv(old_p)) == 50


def test_tolerates_small_rounding_differences(tmp_path):
    """Float formatting differences between sources must not block a merge."""
    old = _frame('2026-01-01', 100)
    new = _frame('2026-01-01', 120).iloc[80:].copy()
    new['close'] = new['close'] * 1.00001  # 0.001% -- well inside tolerance

    old_p = _write(tmp_path, 'old.csv', old)
    new_p = _write(tmp_path, 'new.csv', new)

    res = merge_klines.merge_one(old_p, new_p, tolerance=0.001,
                                 dry_run=False, backup=False)
    assert res['status'] == 'merged'
    assert res['added'] == 20


def test_noop_when_new_file_adds_nothing(tmp_path):
    """A stale re-upload should be recognised as redundant, not rewritten."""
    old = _frame('2026-01-01', 100)
    new = _frame('2026-01-01', 100).iloc[50:]

    old_p = _write(tmp_path, 'old.csv', old)
    new_p = _write(tmp_path, 'new.csv', new)

    res = merge_klines.merge_one(old_p, new_p, tolerance=0.001,
                                 dry_run=False, backup=False)
    assert res['status'] == 'noop'
    assert res['added'] == 0


def test_dry_run_does_not_write(tmp_path):
    old = _frame('2026-01-01', 100)
    new = _frame('2026-01-01', 130).iloc[80:]

    old_p = _write(tmp_path, 'old.csv', old)
    new_p = _write(tmp_path, 'new.csv', new)

    merge_klines.merge_one(old_p, new_p, tolerance=0.001,
                           dry_run=True, backup=False)
    assert len(pd.read_csv(old_p)) == 100


def test_missing_column_raises(tmp_path):
    bad = _frame('2026-01-01', 10).drop(columns=['volume'])
    path = _write(tmp_path, 'bad.csv', bad)
    with pytest.raises(ValueError, match="missing columns"):
        merge_klines.load_ohlcv(path)


def test_duplicate_timestamps_in_source_are_collapsed(tmp_path):
    df = _frame('2026-01-01', 10)
    df = pd.concat([df, df.iloc[[5]]])  # inject a duplicate bar
    path = _write(tmp_path, 'dup.csv', df)
    loaded = merge_klines.load_ohlcv(path)
    assert len(loaded) == 10
    assert not loaded.index.duplicated().any()


def test_creates_file_when_none_exists(tmp_path):
    new = _frame('2026-01-01', 40)
    new_p = _write(tmp_path, 'new.csv', new)
    old_p = str(tmp_path / 'does_not_exist.csv')

    res = merge_klines.merge_one(old_p, new_p, tolerance=0.001,
                                 dry_run=False, backup=False)
    assert res['status'] == 'created'
    assert len(pd.read_csv(old_p)) == 40

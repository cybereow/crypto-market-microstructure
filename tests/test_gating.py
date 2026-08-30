import numpy as np
import pandas as pd

from src.gating import GateConfig, effective_threshold, apply_gate, select_non_overlapping


def test_misalignment_raises_the_bar_and_alignment_lowers_it():
    """The core of the dynamic-threshold idea: a trade fighting BTC's trend
    must be held to a higher confidence standard than one riding it."""
    cfg = GateConfig(base_threshold=0.55, misalignment_penalty=0.08,
                     alignment_bonus=0.02, use_novelty=False)

    thr = effective_threshold(cfg, btc_alignment=np.array([1.0, 0.0]))

    assert thr[0] == 0.53   # aligned -> slightly lower bar
    assert thr[1] == 0.63   # against BTC -> notably higher bar
    assert thr[1] > thr[0]


def test_unknown_regime_is_treated_conservatively():
    """A NaN regime means 'we could not establish market context'. That must
    be treated like misalignment, not like alignment."""
    cfg = GateConfig(base_threshold=0.55, use_novelty=False)

    thr = effective_threshold(cfg, btc_alignment=np.array([np.nan]))

    assert thr[0] > cfg.base_threshold


def test_novelty_raises_the_bar():
    cfg = GateConfig(base_threshold=0.55, use_alignment=False,
                     use_novelty=True, novelty_penalty=0.06)

    thr = effective_threshold(cfg, is_novel=np.array([False, True]))

    assert thr[0] == 0.55
    assert np.isclose(thr[1], 0.61)


def test_adjustments_stack():
    """A novel trade that also fights BTC should face both penalties."""
    cfg = GateConfig(base_threshold=0.55, misalignment_penalty=0.08,
                     novelty_penalty=0.06, max_offset=0.20)

    thr = effective_threshold(cfg, btc_alignment=np.array([0.0]),
                              is_novel=np.array([True]))

    assert np.isclose(thr[0], 0.69)


def test_band_is_relative_so_high_win_rate_geometries_still_gate():
    """Regression test for a real bug: with an absolute [0.45, 0.85] ceiling,
    a high-win-rate barrier geometry (base rate ~85%, calibrated threshold
    ~0.91) had its threshold clamped down to 0.85, which admitted nearly
    every candidate and silently turned the gate into a no-op. The band must
    be relative to the base threshold instead.
    """
    cfg = GateConfig(base_threshold=0.91, misalignment_penalty=0.08,
                     use_novelty=False)

    thr = effective_threshold(cfg, btc_alignment=np.array([0.0, 1.0]))

    assert thr[0] > 0.91   # penalty still applies up here
    assert thr.max() <= 0.99


def test_disabled_switches_are_truly_inert():
    """Each component must be independently switchable, so the ablation
    table measures the component and not a side effect."""
    cfg = GateConfig(base_threshold=0.55, use_alignment=False, use_novelty=False)

    thr = effective_threshold(cfg, btc_alignment=np.array([0.0]),
                              is_novel=np.array([True]))

    assert thr[0] == 0.55


def test_per_trade_base_threshold_is_respected():
    """Pooled walk-forward output carries a per-fold calibrated threshold;
    each trade must be judged against its own fold's threshold, never a
    single global one (which would leak later folds into earlier trades).
    """
    cfg = GateConfig(use_alignment=False, use_novelty=False)

    thr = effective_threshold(cfg, base_threshold=np.array([0.5, 0.9]))

    assert thr[0] == 0.5
    assert thr[1] == 0.9


def test_hard_reject_skips_novel_trades_regardless_of_confidence():
    cfg = GateConfig(base_threshold=0.55, use_alignment=False,
                     use_novelty=True, novelty_hard_reject=True)

    take, _ = apply_gate(np.array([0.99, 0.99]), cfg,
                         is_novel=np.array([False, True]))

    assert take[0] and not take[1]


def test_select_non_overlapping_enforces_one_position_at_a_time():
    """A backtest that lets a second trade open while the first is live
    assumes unlimited capital and reports a win rate over trades no bot
    could have taken."""
    trades = pd.DataFrame({
        'entry_pos': [0, 2, 10],
        'exit_pos': [5, 7, 15],
    }, index=pd.date_range("2023-01-01", periods=3, freq="4h"))

    kept = select_non_overlapping(trades, np.array([True, True, True]))

    assert list(kept['entry_pos']) == [0, 10]  # the entry at 2 was still busy


def test_select_non_overlapping_tracks_assets_independently():
    """Positions on different assets are genuinely concurrent and must not
    block each other."""
    trades = pd.DataFrame({
        'entry_pos': [0, 1],
        'exit_pos': [5, 6],
        'asset': ['ETH', 'SOL'],
    }, index=pd.date_range("2023-01-01", periods=2, freq="4h"))

    kept = select_non_overlapping(trades, np.array([True, True]))

    assert len(kept) == 2


def test_select_non_overlapping_handles_duplicate_timestamps():
    """Regression test: pooled multi-asset frames repeat timestamps. Earlier
    label-based (.loc) selection returned every row sharing a timestamp,
    inflating the trade count above the number of candidates that existed.
    """
    idx = pd.DatetimeIndex(["2023-01-01"] * 4)
    trades = pd.DataFrame({
        'entry_pos': [0, 0, 10, 10],
        'exit_pos': [5, 5, 15, 15],
        'asset': ['ETH', 'SOL', 'ETH', 'SOL'],
    }, index=idx)

    kept = select_non_overlapping(trades, np.array([True] * 4))

    assert len(kept) == 4  # not 16

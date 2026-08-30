"""The decision layer: turn a calibrated P(win) into a take/skip decision
using a *context-dependent* threshold instead of one global constant.

Why this exists as its own layer. A single `--confidence 0.55` constant
implicitly asserts that 0.55 means the same thing in every market context.
It does not. A long breakout while BTC is trending up and a long breakout
while BTC is dumping are not the same bet, even when the model outputs the
same number for both — the second one carries a market-factor headwind the
model has only partial ability to price. Likewise, 0.55 from a sample whose
decision path was well-travelled in training is much more trustworthy than
0.55 extrapolated from a nearly-empty leaf.

Rather than hoping the model absorbs these effects, we apply them where
decisions are actually made, as explicit, inspectable, individually
switchable adjustments:

  base threshold
    + alignment_penalty   if the trade fights BTC's trend
    - alignment_bonus     if the trade rides BTC's trend
    + novelty_penalty     if the trade sits in a rarely-visited leaf region

Every adjustment is additive in probability space and the result is clipped
to a sane band, so no combination of knobs can produce a threshold of 0.99
(take nothing) or 0.2 (take everything) by accident.

IMPORTANT (and the reason `describe()` exists): each knob here is a degree
of freedom that can be tuned to fit noise. Four knobs on a few thousand
trades will happily manufacture a beautiful in-sample number. These must be
set once on defaults and validated with walk-forward retraining
(scripts/backtest_meta_ml_walkforward.py), never tuned per-fold until the
result looks good.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class GateConfig:
    """All decision-layer knobs in one place, with conservative defaults.

    Defaults are deliberately modest (0.05-0.08 in probability space): large
    enough to matter, small enough that if the underlying hypothesis is
    wrong the damage is bounded.
    """
    base_threshold: float = 0.55

    # Cross-asset regime (Idea 1 + 2)
    use_alignment: bool = True
    misalignment_penalty: float = 0.08   # trading against BTC -> demand more confidence
    alignment_bonus: float = 0.02        # trading with BTC -> accept slightly less

    # Out-of-distribution guard (Idea 4)
    use_novelty: bool = True
    novelty_penalty: float = 0.06        # rare decision path -> demand more confidence
    novelty_hard_reject: bool = False    # or skip such trades outright

    # Safety band, expressed RELATIVE to base_threshold rather than as
    # absolute probabilities.
    #
    # This matters more than it looks. An absolute band like [0.45, 0.85]
    # silently breaks under high-win-rate barrier geometries: when the
    # profit target is much closer than the stop, the base rate of wins is
    # ~85%, so a calibrated threshold legitimately lands at 0.90+. An
    # absolute 0.85 ceiling would clamp it back down and admit essentially
    # every candidate — turning the gate into a no-op precisely where
    # selectivity is needed most. A relative band adapts to whatever
    # probability scale the geometry produces.
    max_offset: float = 0.15   # how far above base_threshold adjustments may push
    min_offset: float = 0.10   # and how far below
    hard_floor: float = 0.05   # absolute sanity bounds only
    hard_ceiling: float = 0.99

    def describe(self) -> dict:
        return asdict(self)

    def band(self) -> tuple:
        """Resolved (low, high) clip bounds for this config's base."""
        low = max(self.base_threshold - self.min_offset, self.hard_floor)
        high = min(self.base_threshold + self.max_offset, self.hard_ceiling)
        return low, high


def effective_threshold(config: GateConfig, btc_alignment=None, is_novel=None,
                        n: int = None, base_threshold=None) -> np.ndarray:
    """Per-trade threshold as an array.

    Args:
        btc_alignment: array of 1.0 (aligned with BTC) / 0.0 (against), or
            None to disable the adjustment for this call. NaN entries are
            treated as misaligned — the conservative reading of "we could
            not establish the market regime".
        is_novel: boolean array, True where the decision path was rare.
        n: length to broadcast to when both context arrays are None.
        base_threshold: optional per-trade base (array) overriding
            config.base_threshold. Needed when evaluating pooled
            walk-forward output, where each trade's base threshold is the
            one calibrated in its OWN fold — using a single global base
            there would leak later folds' calibration into earlier trades.
    """
    for candidate in (btc_alignment, is_novel, base_threshold):
        if candidate is not None:
            length = len(candidate)
            break
    else:
        if n is None:
            raise ValueError("effective_threshold needs btc_alignment, is_novel, "
                             "base_threshold, or n.")
        length = n

    if base_threshold is None:
        thr = np.full(length, float(config.base_threshold))
    else:
        thr = np.asarray(base_threshold, dtype=float).copy()

    if config.use_alignment and btc_alignment is not None:
        align = np.asarray(btc_alignment, dtype=float)
        aligned = align >= 0.5
        unknown = np.isnan(align)
        thr = np.where(aligned & ~unknown, thr - config.alignment_bonus, thr)
        thr = np.where(~aligned | unknown, thr + config.misalignment_penalty, thr)

    if config.use_novelty and is_novel is not None:
        novel = np.asarray(is_novel, dtype=bool)
        thr = np.where(novel, thr + config.novelty_penalty, thr)

    # Clip relative to each trade's own base, so a per-fold base threshold
    # keeps its own band instead of being squeezed by the config's scalar.
    if base_threshold is None:
        low, high = config.band()
    else:
        base_arr = np.asarray(base_threshold, dtype=float)
        low = np.maximum(base_arr - config.min_offset, config.hard_floor)
        high = np.minimum(base_arr + config.max_offset, config.hard_ceiling)
    return np.clip(thr, low, high)


def apply_gate(p_win, config: GateConfig, btc_alignment=None, is_novel=None,
               base_threshold=None):
    """Returns (take_mask, thresholds) for a batch of candidate trades.

    `take_mask` is True where the model's confidence clears that trade's own
    context-adjusted bar. When `novelty_hard_reject` is set, novel trades are
    rejected regardless of confidence — the "I don't recognize this
    situation, sit it out" branch.
    """
    p_win = np.asarray(p_win, dtype=float)
    thresholds = effective_threshold(config, btc_alignment=btc_alignment,
                                     is_novel=is_novel, n=len(p_win),
                                     base_threshold=base_threshold)
    take = p_win >= thresholds

    if config.use_novelty and config.novelty_hard_reject and is_novel is not None:
        take = take & ~np.asarray(is_novel, dtype=bool)

    return take, thresholds


def select_non_overlapping(trades, take_mask):
    """Enforce the single-position constraint a real bot has: among the
    trades the gate approved, walk forward in time and skip any that would
    open while a previously-taken position on the same asset is still live.

    Without this, a backtest silently assumes unlimited simultaneous
    capital and reports a win rate over trades that could never all have
    been taken. Expects `entry_pos`/`exit_pos` columns (bar indices) as
    produced by triple_barrier_labels, and handles multiple assets by
    tracking each asset's busy-until independently.
    """
    approved = trades[np.asarray(take_mask, dtype=bool)]
    if approved.empty:
        return approved

    # Positional (iloc) selection, NOT label-based .loc: these frames are
    # indexed by entry timestamp and pool multiple assets, so the same
    # timestamp appears many times. Selecting duplicate labels with .loc
    # silently returns every row sharing that timestamp, inflating the
    # trade count above the number of candidates that existed.
    has_asset = 'asset' in approved.columns
    order = np.argsort(approved['entry_pos'].to_numpy(), kind='stable')
    entry_pos = approved['entry_pos'].to_numpy()
    exit_pos = approved['exit_pos'].to_numpy()
    assets = (approved['asset'].to_numpy() if has_asset
              else np.zeros(len(approved), dtype=int))

    busy_until = {}
    keep_positions = []
    for i in order:
        key = assets[i]
        if entry_pos[i] < busy_until.get(key, -1):
            continue
        keep_positions.append(i)
        busy_until[key] = exit_pos[i]

    return approved.iloc[sorted(keep_positions)]

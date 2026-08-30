"""Precision-targeted threshold calibration.

The problem with `scoring='f1'` (what train_meta_ml.py used): F1 is the
harmonic mean of precision and recall, so it rewards a model for *taking
trades*. A trading system does not want recall. Missing a good trade costs
nothing but opportunity; taking a bad one costs money. Optimizing F1 picks
hyperparameters that fire often at mediocre precision, which is the direct
opposite of a high-win-rate objective.

The tempting fix is a custom asymmetric objective (hand-written
gradient/hessian that punishes false positives harder). That is risky here:
with a few thousand pooled trades and a heavily-regularized shallow model,
a hand-rolled objective is easy to get subtly wrong and hard to diagnose,
and it changes what the probabilities *mean*, breaking the Kelly sizing
downstream which needs a calibrated P(win).

This module gets the same precision-first behaviour by leaving the model's
probability estimates alone and moving the *decision threshold* instead:

  1. `precision_at_threshold_scorer` — a drop-in sklearn scorer for the
     hyperparameter search that measures precision on the highest-confidence
     slice of predictions, so the search selects models that RANK well at
     the top of the book, which is the only region a selective strategy
     ever trades.
  2. `calibrate_threshold_for_precision` — walks the precision-recall curve
     to find the lowest threshold that still achieves a target precision
     (e.g. 65% win rate), with a minimum-support floor so we don't "achieve"
     90% precision on 3 trades.

Both operate on held-out predictions, never on the data used to fit.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, precision_recall_curve


def top_quantile_precision(y_true, y_score, quantile: float = 0.8,
                           min_support: int = 10) -> float:
    """Precision among predictions in the top `1 - quantile` of scores.

    This is the metric that matches how the strategy actually behaves: it
    only ever takes the most confident candidates, so model selection should
    reward being right *there*, not on average across all candidates.

    Returns 0.0 when the confident slice is too small to be meaningful,
    which makes the hyperparameter search avoid degenerate models that
    concentrate all scores into a tie.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) == 0:
        return 0.0

    thr = np.quantile(y_score, quantile)
    mask = y_score >= thr
    if mask.sum() < min_support:
        return 0.0
    return float(y_true[mask].mean())


def precision_at_threshold_scorer(quantile: float = 0.8, min_support: int = 10):
    """sklearn scorer wrapping `top_quantile_precision`, for use as
    RandomizedSearchCV(scoring=...). Needs probabilities, so
    needs_proba=True.
    """
    return make_scorer(
        top_quantile_precision,
        response_method='predict_proba',
        quantile=quantile,
        min_support=min_support,
    )


def calibrate_threshold_for_precision(y_true, y_score, target_precision: float = 0.65,
                                      min_trades: int = 30,
                                      fallback_quantile: float = 0.8) -> dict:
    """Find the lowest score threshold whose precision >= `target_precision`
    while still admitting at least `min_trades` predictions.

    Lowest-qualifying rather than highest-precision on purpose: among
    thresholds that hit the target win rate we want the one that trades the
    most, because a strategy that clears its win-rate goal on 200 trades is
    far more trustworthy (and more profitable) than one that clears it on 12.

    The `min_trades` floor is the guard against the classic
    threshold-calibration trap — pushing the threshold to 0.99 to report a
    spectacular win rate on a sample too small to mean anything.

    If no threshold satisfies both constraints, falls back to
    `fallback_quantile` of the score distribution and reports
    `target_met=False` so the caller can surface the miss honestly instead
    of silently shipping a threshold that does not do what was asked.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) == 0:
        return {'threshold': 0.5, 'precision': 0.0, 'n_trades': 0,
                'target_met': False, 'reason': 'empty input'}

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    best = None
    for prec, thr in zip(precisions[:-1], thresholds):
        n = int((y_score >= thr).sum())
        if prec >= target_precision and n >= min_trades:
            # Thresholds are ascending, so the first qualifying one is the
            # lowest -> most trades.
            best = {'threshold': float(thr), 'precision': float(prec), 'n_trades': n,
                    'target_met': True, 'reason': 'target precision reached'}
            break

    if best is not None:
        return best

    thr = float(np.quantile(y_score, fallback_quantile))
    mask = y_score >= thr
    n = int(mask.sum())
    prec = float(y_true[mask].mean()) if n > 0 else 0.0
    return {'threshold': thr, 'precision': prec, 'n_trades': n, 'target_met': False,
            'reason': f'no threshold reached {target_precision:.0%} precision with '
                      f'>= {min_trades} trades; fell back to the '
                      f'{fallback_quantile:.0%} score quantile'}


def precision_threshold_table(y_true, y_score, thresholds=None) -> pd.DataFrame:
    """Diagnostic sweep: win rate and trade count at a range of thresholds.

    Printed by the walk-forward validator so the precision/volume tradeoff
    is visible rather than hidden behind one chosen number — the honest way
    to present "we can hit 90% win rate" is next to the column showing how
    many trades survive at that level.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if thresholds is None:
        thresholds = np.arange(0.40, 0.86, 0.05)

    rows = []
    for thr in thresholds:
        mask = y_score >= thr
        n = int(mask.sum())
        rows.append({
            'threshold': round(float(thr), 3),
            'n_trades': n,
            'win_rate': float(y_true[mask].mean()) if n > 0 else float('nan'),
            'pct_of_candidates': n / len(y_true) if len(y_true) else 0.0,
        })
    return pd.DataFrame(rows)

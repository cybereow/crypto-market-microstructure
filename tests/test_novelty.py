import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.novelty import LeafNoveltyDetector


@pytest.fixture(scope="module")
def fitted():
    """A model trained on a tight cluster of samples, so anything far from
    that cluster is genuinely out-of-distribution for it."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, size=(400, 5)),
                     columns=[f"f{i}" for i in range(5)])
    y = (X['f0'] + rng.normal(0, 0.5, 400) > 0).astype(int)

    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=42,
                          eval_metric='logloss')
    model.fit(X, y)
    detector = LeafNoveltyDetector().fit(model, X)
    return model, detector, X


def test_novelty_flags_far_out_of_distribution_samples(fitted):
    """Samples placed far outside the training cluster must be scored as
    more novel, on average, than the training data itself. This is the
    property the OOD guard depends on.
    """
    model, detector, X = fitted

    far = pd.DataFrame(np.full((50, 5), 50.0), columns=X.columns)
    assert detector.score(model, far).mean() > detector.score(model, X).mean()


def test_training_data_is_mostly_not_novel(fitted):
    """The cutoff is calibrated at the 10th percentile of training support,
    so by construction ~10% of training rows flag — never most of them.
    A detector that called its own training data unfamiliar would reject
    everything in production."""
    model, detector, X = fitted

    flagged = detector.is_novel(model, X).mean()
    assert flagged < 0.25


def test_support_is_higher_for_familiar_samples(fitted):
    model, detector, X = fitted

    far = pd.DataFrame(np.full((50, 5), 50.0), columns=X.columns)
    assert detector.support(model, X).mean() > detector.support(model, far).mean()


def test_round_trip_serialization_preserves_decisions(fitted):
    """The detector is persisted as JSON alongside the model; a reloaded
    detector must make identical decisions, otherwise the deployed strategy
    differs from the validated one."""
    model, detector, X = fitted

    reloaded = LeafNoveltyDetector.from_dict(detector.to_dict())

    assert reloaded.cutoff_ == detector.cutoff_
    assert reloaded.n_train_ == detector.n_train_
    np.testing.assert_array_equal(
        reloaded.is_novel(model, X), detector.is_novel(model, X))
    np.testing.assert_allclose(
        reloaded.support(model, X), detector.support(model, X))


def test_score_before_fit_raises():
    """Fail loudly rather than silently scoring everything as familiar."""
    det = LeafNoveltyDetector()
    with pytest.raises(RuntimeError):
        det.support(None, pd.DataFrame({'a': [1.0]}))

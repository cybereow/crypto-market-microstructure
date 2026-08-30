"""Leaf-frequency novelty detection — an out-of-distribution guard that
reuses the model you already trained.

The problem it solves: a classifier's probability output is only meaningful
for inputs that resemble its training data. When market conditions drift
into a state the model never saw (a new volatility regime, a structural
break), it does not report "I don't know" — it reports a confident number
computed by extrapolating rules learned elsewhere. Those confidently-wrong
predictions are where selective strategies bleed, because a high threshold
actively *selects* for them.

The textbook fix is a second model (Isolation Forest / one-class SVM) that
scores inputs for novelty. That means another artifact to train, version,
save, and keep in sync with the feature list — and a second thing that can
silently break.

This module gets novelty detection for free from the existing XGBoost model
via `predict(pred_leaf=True)`, which reports which leaf each tree routed the
sample into. A trained forest already partitions feature space; the leaf
tuple IS a learned, model-relevant fingerprint of "what kind of situation is
this". So:

  * At fit time, record how often each (tree, leaf) was reached by training
    samples.
  * At inference, look up the leaves a new sample lands in. If it
    consistently lands in leaves that were nearly empty during training, the
    sample sits in a thinly-populated corner of the model's own learned
    space — its probability is an extrapolation and should be distrusted.

This is strictly better than a generic distance-based novelty score for our
purpose, because it measures novelty in the geometry the model actually
uses for its decisions, not in raw feature space where irrelevant
directions count equally.

No extra artifact: the counts are derived from the model plus the training
matrix, so they are rebuilt deterministically at every retrain.
"""
import numpy as np


class LeafNoveltyDetector:
    """Scores samples by how well-supported their decision paths were in
    training.

    Usage:
        det = LeafNoveltyDetector().fit(model, X_train)
        scores = det.score(model, X_new)   # higher = more novel/unfamiliar
        rare = det.is_novel(model, X_new)  # bool mask vs the fitted cutoff
    """

    def __init__(self, rare_percentile: float = 10.0):
        """`rare_percentile`: a sample is flagged novel when its support
        score falls below this percentile of the training distribution.
        10 means "the 10% least-familiar-looking inputs", i.e. we accept
        losing a tenth of candidates to buy protection against
        extrapolation.
        """
        self.rare_percentile = rare_percentile
        self.leaf_counts_ = None   # list of dict{leaf_id: count}, one per tree
        self.n_train_ = 0
        self.cutoff_ = None
        self.ref_support_ = None   # training-set support scale, for batch-independent scores

    @staticmethod
    def _leaves(model, X) -> np.ndarray:
        """(n_samples, n_trees) matrix of leaf indices.

        Uses the sklearn wrapper's `apply` when available (XGBClassifier
        exposes it) and falls back to the Booster's pred_leaf, so this works
        whichever object the caller passes.
        """
        if hasattr(model, 'apply'):
            leaves = model.apply(X)
        else:  # raw Booster
            import xgboost as xgb
            leaves = model.predict(xgb.DMatrix(X), pred_leaf=True)
        leaves = np.asarray(leaves)
        if leaves.ndim == 1:  # single-tree edge case
            leaves = leaves.reshape(-1, 1)
        return leaves.astype(int)

    def fit(self, model, X_train):
        """Record per-tree leaf occupancy on the training set."""
        leaves = self._leaves(model, X_train)
        self.n_train_ = leaves.shape[0]
        self.leaf_counts_ = []
        for t in range(leaves.shape[1]):
            ids, counts = np.unique(leaves[:, t], return_counts=True)
            self.leaf_counts_.append(dict(zip(ids.tolist(), counts.tolist())))

        # Calibrate the "what counts as rare" cutoff on the training
        # distribution itself, so the threshold is in the right units
        # regardless of model size or dataset size.
        train_support = self._support(model, X_train)
        self.cutoff_ = float(np.percentile(train_support, self.rare_percentile))

        # Reference scale for `score()`, captured at fit time. Normalizing
        # by the *batch* maximum instead would make a sample's novelty score
        # depend on which other samples happen to be scored alongside it —
        # so a batch consisting entirely of OOD samples would report zero
        # novelty for all of them.
        self.ref_support_ = float(max(train_support.max(), 1e-12))
        return self

    def _support(self, model, X) -> np.ndarray:
        """Mean training occupancy (as a fraction of the training set) of
        the leaves a sample visits. High = familiar territory.
        """
        if self.leaf_counts_ is None:
            raise RuntimeError("LeafNoveltyDetector.fit must be called first.")

        leaves = self._leaves(model, X)
        n_trees = min(leaves.shape[1], len(self.leaf_counts_))
        support = np.zeros(leaves.shape[0], dtype=float)
        for t in range(n_trees):
            counts = self.leaf_counts_[t]
            support += np.array([counts.get(int(leaf), 0) for leaf in leaves[:, t]],
                                dtype=float)
        return support / (max(n_trees, 1) * max(self.n_train_, 1))

    def score(self, model, X) -> np.ndarray:
        """Novelty score in [0, 1]: higher = less familiar.

        Defined as 1 - support/training_support_scale, so it reads naturally
        as "how unusual is this" and — crucially — is an absolute per-sample
        property, independent of the rest of the batch.
        """
        if self.ref_support_ is None:
            raise RuntimeError("LeafNoveltyDetector.fit must be called first.")
        support = self._support(model, X)
        return np.clip(1.0 - support / self.ref_support_, 0.0, 1.0)

    def is_novel(self, model, X) -> np.ndarray:
        """Boolean mask: True where the sample's decision path was rarely
        travelled in training (support below the fitted cutoff).
        """
        return self._support(model, X) < self.cutoff_

    def support(self, model, X) -> np.ndarray:
        """Raw normalized support, exposed for diagnostics/reporting."""
        return self._support(model, X)

    def to_dict(self) -> dict:
        """Serializable state. JSON keys must be strings, so leaf ids are
        stringified here and parsed back in `from_dict`.
        """
        return {
            'rare_percentile': self.rare_percentile,
            'n_train': self.n_train_,
            'cutoff': self.cutoff_,
            'ref_support': self.ref_support_,
            'leaf_counts': [{str(k): v for k, v in tree.items()}
                            for tree in (self.leaf_counts_ or [])],
        }

    @classmethod
    def from_dict(cls, state: dict) -> "LeafNoveltyDetector":
        det = cls(rare_percentile=state.get('rare_percentile', 10.0))
        det.n_train_ = state.get('n_train', 0)
        det.cutoff_ = state.get('cutoff')
        det.ref_support_ = state.get('ref_support')
        det.leaf_counts_ = [{int(k): v for k, v in tree.items()}
                            for tree in state.get('leaf_counts', [])]
        return det


def save_novelty(detector: LeafNoveltyDetector, path: str) -> None:
    import json
    with open(path, 'w') as f:
        json.dump(detector.to_dict(), f)


def load_novelty(path: str):
    """Returns None when the artifact is absent, so callers can degrade
    gracefully to "no OOD filtering" instead of failing.
    """
    import json
    import os
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return LeafNoveltyDetector.from_dict(json.load(f))

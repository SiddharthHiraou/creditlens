"""Stacked ensemble: a logistic meta-learner over the three GBDT base models.

This exists as a **ceiling reference**, not as a serving candidate. It tells you
how much signal the individual models are leaving on the table. If the stack
barely beats the best single model, the champion is close to the achievable
limit on these features and further tuning is wasted effort.

The meta-learner is fitted on validation predictions, not training predictions.
Base models fit their own training data nearly perfectly, so stacking on
in-sample predictions teaches the meta-learner that whichever model overfits
hardest is the most trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_SEED


@dataclass
class StackedEnsemble:
    bases: list
    meta: LogisticRegression
    feature_names: list[str]
    base_names: list[str]

    def _meta_matrix(self, x) -> np.ndarray:
        return np.column_stack([b.predict_pd(x) for b in self.bases])

    def predict_pd(self, x) -> np.ndarray:
        return self.meta.predict_proba(self._meta_matrix(x))[:, 1]

    def weights(self) -> dict[str, float]:
        return dict(zip(self.base_names, self.meta.coef_[0].tolist(), strict=True))


def fit_stack(bases: list, x_meta, y_meta: np.ndarray) -> StackedEnsemble:
    """Fit the meta-learner on held-out (validation) base predictions."""
    matrix = np.column_stack([b.predict_pd(x_meta) for b in bases])
    meta = LogisticRegression(
        C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED
    )
    meta.fit(matrix, y_meta)
    return StackedEnsemble(
        bases=bases,
        meta=meta,
        feature_names=bases[0].feature_names,
        base_names=[b.name for b in bases],
    )

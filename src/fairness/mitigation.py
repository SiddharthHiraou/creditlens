"""Bias mitigation experiments, and what they cost.

The claim this module supports is deliberately narrow: disparity was measured,
mitigation options were tested, and the accuracy cost of each was quantified.
It does not claim bias was removed.

**A legal caveat that belongs before any of the code.** Group-specific decision
thresholds are the most effective mitigation here and are also, in US consumer
lending, generally **unlawful**: setting a different cutoff by a protected class
is disparate treatment even when the intent is to reduce disparate impact. The
threshold optimiser below is therefore an *analytical instrument* — it
establishes the frontier, showing how much disparity is attributable to the
cutoff versus to the model, and what parity would cost if it were achievable.
It is not a deployable recommendation.

The deployable options, and what this module can say about each:

* **A single stricter or looser cutoff** — legal, and the tradeoff curve shows
  what it does to both approval rate and disparity.
* **Dropping the protected attribute and its proxies from the model** — legal,
  measurable here, and usually less effective than expected because the
  information survives in correlated features.
* **Reweighting or constrained training** — legal; the exponentiated gradient
  reduction is included for the frontier it produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from src.fairness.report import fairness_report


class _ScorePassthrough(BaseEstimator, ClassifierMixin):
    """Minimal sklearn estimator whose prediction *is* the score.

    ThresholdOptimizer post-processes a continuous score into group-specific
    cutoffs; it needs an estimator to call. This hands the score back unchanged.

    Inherits from ``BaseEstimator`` because Fairlearn runs ``check_is_fitted``,
    which needs sklearn's tag machinery, and carries a trailing-underscore
    attribute so that check passes under ``prefit=True``.
    """

    def __init__(self) -> None:
        self.fitted_ = True
        self.classes_ = np.array([0, 1])

    def fit(self, x, y=None):  # noqa: D102 - trivial
        self.fitted_ = True
        return self

    def predict(self, x):  # noqa: D102 - trivial
        return np.asarray(x, dtype=float).ravel()


@dataclass(frozen=True)
class MitigationResult:
    strategy: str
    auc: float
    auc_cost: float
    approval_rate: float
    disparate_impact: float
    equal_opportunity_difference: float
    bad_rate_among_approved: float
    passes_four_fifths: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "auc": round(self.auc, 4),
            "auc_cost": round(self.auc_cost, 4),
            "approval_rate": round(self.approval_rate, 4),
            "disparate_impact": round(self.disparate_impact, 4),
            "equal_opportunity_difference": round(self.equal_opportunity_difference, 4),
            "bad_rate_among_approved": round(self.bad_rate_among_approved, 4),
            "passes_four_fifths": self.passes_four_fifths,
            "note": self.note,
        }


def _evaluate(
    strategy: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    approved: np.ndarray,
    sensitive: np.ndarray,
    baseline_auc: float,
    *,
    note: str = "",
) -> MitigationResult:
    y = np.asarray(y_true).astype(int)
    rep = fairness_report(y, approved, sensitive, attribute="mitigation")
    auc = float(roc_auc_score(y, -np.asarray(scores, dtype=float)))
    approved_mask = np.asarray(approved).astype(bool)
    return MitigationResult(
        strategy=strategy,
        auc=auc,
        auc_cost=baseline_auc - auc,
        approval_rate=float(approved_mask.mean()),
        disparate_impact=rep.disparate_impact,
        equal_opportunity_difference=rep.equal_opportunity_difference,
        bad_rate_among_approved=float(y[approved_mask].mean())
        if approved_mask.any()
        else float("nan"),
        passes_four_fifths=rep.passes_four_fifths,
        note=note,
    )


def cutoff_tradeoff_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    sensitive: np.ndarray,
    *,
    approval_rates: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> pl.DataFrame:
    """What a single, legal, group-blind cutoff does to disparity.

    Usually the most important row in a fairness report: it shows how much of
    the observed disparity is a property of *where the cutoff sits* rather than
    of the model, and it is the only lever here that can actually be pulled.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    baseline_auc = float(roc_auc_score(y, -s))

    rows = []
    for rate in approval_rates:
        cutoff = float(np.quantile(s, 1 - rate))
        approved = s >= cutoff
        res = _evaluate(
            f"single cutoff @ {rate:.0%} approval",
            y,
            s,
            approved,
            sensitive,
            baseline_auc,
            note=f"score >= {cutoff:.1f}",
        )
        rows.append({"target_approval_rate": rate, "cutoff": cutoff, **res.as_dict()})
    return pl.DataFrame(rows)


def threshold_optimizer_frontier(
    y_true: np.ndarray,
    scores: np.ndarray,
    sensitive: np.ndarray,
    *,
    constraints: tuple[str, ...] = ("demographic_parity", "equalized_odds"),
    seed: int = 20260820,
) -> pl.DataFrame:
    """Fairlearn ThresholdOptimizer frontier — analytical only, not deployable.

    Fits group-specific thresholds that satisfy the named constraint, then
    reports the accuracy and approval-rate cost. See the module docstring: this
    establishes what parity would cost, it does not propose shipping it.
    """
    from fairlearn.postprocessing import ThresholdOptimizer

    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    baseline_auc = float(roc_auc_score(y, -s))
    # Favourable outcome is repayment, so the optimiser is fitted on 1 - y.
    favourable = 1 - y

    rows = []
    for constraint in constraints:
        try:
            # The "estimator" must hand ThresholdOptimizer a continuous score
            # to cut on. A DummyClassifier emits a constant, which silently
            # produces a degenerate solution that approves everyone at DI 1.0 --
            # a passing fairness number that means nothing. Pass the score
            # through instead.
            opt = ThresholdOptimizer(
                estimator=_ScorePassthrough(),
                constraints=constraint,
                objective="accuracy_score",
                prefit=True,
                predict_method="predict",
            )
            opt.fit(s.reshape(-1, 1), favourable, sensitive_features=sensitive)
            approved = (
                np.asarray(
                    opt.predict(
                        s.reshape(-1, 1),
                        sensitive_features=sensitive,
                        random_state=seed,
                    )
                ).astype(int)
                == 1
            )
        except Exception as exc:  # noqa: BLE001 - report rather than crash the report
            rows.append(
                {
                    "strategy": f"threshold_optimizer[{constraint}]",
                    "auc": float("nan"),
                    "auc_cost": float("nan"),
                    "approval_rate": float("nan"),
                    "disparate_impact": float("nan"),
                    "equal_opportunity_difference": float("nan"),
                    "bad_rate_among_approved": float("nan"),
                    "passes_four_fifths": False,
                    "note": f"failed: {type(exc).__name__}",
                }
            )
            continue

        rows.append(
            _evaluate(
                f"threshold_optimizer[{constraint}]",
                y,
                s,
                approved,
                sensitive,
                baseline_auc,
                note="group-specific thresholds — analytical only, unlawful to deploy",
            ).as_dict()
        )
    return pl.DataFrame(rows)


def drop_attribute_experiment(
    train_matrix: np.ndarray,
    train_y: np.ndarray,
    test_matrix: np.ndarray,
    test_y: np.ndarray,
    feature_names: list[str],
    sensitive_test: np.ndarray,
    drop: list[str],
    *,
    approve_rate: float = 0.60,
) -> pl.DataFrame:
    """Retrain without the named features and re-measure.

    The usual result, and the one worth documenting: removing a protected
    attribute moves disparity far less than expected, because the information
    survives in correlated features. "We don't use age" is not a fairness
    control.
    """
    from src.models.gbdt import fit_lightgbm

    keep = [f for f in feature_names if f not in set(drop)]
    idx = [feature_names.index(f) for f in keep]

    rows = []
    for label, cols, names in (
        ("with protected features", list(range(len(feature_names))), feature_names),
        (f"without {', '.join(drop)}", idx, keep),
    ):
        model = fit_lightgbm(
            train_matrix[:, cols],
            train_y,
            names,
            eval_set=(test_matrix[:, cols], test_y),
        )
        pd_hat = model.predict_pd(test_matrix[:, cols])
        score = -pd_hat  # higher is safer
        cutoff = float(np.quantile(score, 1 - approve_rate))
        approved = score >= cutoff
        res = _evaluate(
            label,
            test_y,
            score,
            approved,
            sensitive_test,
            float(roc_auc_score(test_y, pd_hat)),
        )
        rows.append(
            {
                **res.as_dict(),
                "auc": round(float(roc_auc_score(test_y, pd_hat)), 4),
                "auc_cost": 0.0,
            }
        )
    return pl.DataFrame(rows)

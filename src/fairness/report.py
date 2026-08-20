"""Fairness measurement across protected-class proxies.

Scope, stated up front because it is the thing most portfolio projects get
wrong: this module **measures** disparity and quantifies what removing it would
cost. It does not claim to eliminate bias, and no output here should be read as
a model being "fair". A model risk team measures, quantifies the tradeoff, and
documents the decision — that is what this reproduces.

Metrics, and why each is here:

* **Disparate impact ratio** with the four-fifths flag. The EEOC's rule of thumb
  is that a selection rate below 80% of the most-selected group's warrants
  scrutiny. It is the standard first screen, and it is a screen, not a verdict.
* **Equal opportunity difference** — gap in true positive rate. Among applicants
  who would actually have repaid, does the model approve them at equal rates?
* **Equalized odds difference** — the worse of the TPR and FPR gaps. Stricter,
  and generally impossible to satisfy simultaneously with calibration.
* **Calibration by group** — does a 5% predicted PD mean 5% for every group? A
  model can pass selection-rate parity and still be systematically wrong about
  one group's risk, which is the failure that actually harms borrowers.

Note the tension deliberately: calibration by group and equalized odds cannot
both hold when base rates differ across groups. That is a mathematical result,
not an implementation gap, and the choice of which to prioritise is a policy
decision that belongs in the model card.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

FOUR_FIFTHS = 0.80


@dataclass(frozen=True)
class GroupFairnessReport:
    attribute: str
    by_group: pl.DataFrame
    disparate_impact: float
    equal_opportunity_difference: float
    equalized_odds_difference: float
    selection_rate_difference: float
    passes_four_fifths: bool
    reference_group: str
    worst_group: str

    def summary(self) -> dict:
        return {
            "attribute": self.attribute,
            "disparate_impact": round(self.disparate_impact, 4),
            "passes_four_fifths": self.passes_four_fifths,
            "equal_opportunity_difference": round(self.equal_opportunity_difference, 4),
            "equalized_odds_difference": round(self.equalized_odds_difference, 4),
            "selection_rate_difference": round(self.selection_rate_difference, 4),
            "reference_group": self.reference_group,
            "worst_group": self.worst_group,
        }


def _rate(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def group_metrics(
    y_true: np.ndarray,
    approved: np.ndarray,
    sensitive: np.ndarray,
    pd_hat: np.ndarray | None = None,
) -> pl.DataFrame:
    """Per-group selection, error and calibration metrics.

    ``approved`` is the binary decision (approve vs not), ``y_true`` is the
    default label where 1 = bad. A "good" outcome for equal-opportunity
    purposes is ``y_true == 0``: among applicants who would have repaid, who
    got approved?
    """
    y = np.asarray(y_true).astype(int)
    a = np.asarray(approved).astype(bool)
    g = np.asarray(sensitive).astype(str)

    rows = []
    for value in sorted(set(g.tolist())):
        m = g == value
        good, bad = m & (y == 0), m & (y == 1)
        row = {
            "group": value,
            "n": int(m.sum()),
            "share": float(m.mean()),
            "selection_rate": _rate(a[m]),
            "observed_bad_rate": _rate(y[m].astype(bool)),
            # TPR here = approval rate among those who would repay.
            "tpr_good_approved": _rate(a[good]) if good.any() else float("nan"),
            # FPR here = approval rate among those who would default.
            "fpr_bad_approved": _rate(a[bad]) if bad.any() else float("nan"),
        }
        if pd_hat is not None:
            p = np.asarray(pd_hat, dtype=float)
            row["mean_predicted_pd"] = float(p[m].mean())
            # Positive means the model overstates this group's risk.
            row["calibration_gap"] = float(p[m].mean() - y[m].mean())
        rows.append(row)
    return pl.DataFrame(rows).sort("group")


def fairness_report(
    y_true: np.ndarray,
    approved: np.ndarray,
    sensitive: np.ndarray,
    *,
    attribute: str,
    pd_hat: np.ndarray | None = None,
) -> GroupFairnessReport:
    by_group = group_metrics(y_true, approved, sensitive, pd_hat)

    rates = by_group["selection_rate"].to_numpy()
    groups = by_group["group"].to_list()
    best_i, worst_i = int(np.nanargmax(rates)), int(np.nanargmin(rates))
    di = float(rates[worst_i] / rates[best_i]) if rates[best_i] > 0 else float("nan")

    tpr = by_group["tpr_good_approved"].to_numpy()
    fpr = by_group["fpr_bad_approved"].to_numpy()
    eod = float(np.nanmax(tpr) - np.nanmin(tpr))
    # Equalized odds is the worse of the two gaps, not their average.
    eqo = float(max(eod, np.nanmax(fpr) - np.nanmin(fpr)))

    return GroupFairnessReport(
        attribute=attribute,
        by_group=by_group,
        disparate_impact=di,
        equal_opportunity_difference=eod,
        equalized_odds_difference=eqo,
        selection_rate_difference=float(rates[best_i] - rates[worst_i]),
        passes_four_fifths=bool(di >= FOUR_FIFTHS),
        reference_group=groups[best_i],
        worst_group=groups[worst_i],
    )


def fairlearn_metrics(
    y_true: np.ndarray, approved: np.ndarray, sensitive: np.ndarray
) -> dict[str, float]:
    """Cross-check against Fairlearn's own implementations.

    Computing these two ways is cheap insurance: a hand-rolled fairness metric
    that quietly disagrees with the reference implementation is exactly the kind
    of error that survives review and then fails an audit.
    """
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
    )

    y = np.asarray(y_true).astype(int)
    # Fairlearn's convention is a positive prediction; here that is "approved",
    # and the favourable *outcome* label is repayment, so the label is inverted.
    return {
        "demographic_parity_difference": float(
            demographic_parity_difference(y, approved, sensitive_features=sensitive)
        ),
        "demographic_parity_ratio": float(
            demographic_parity_ratio(y, approved, sensitive_features=sensitive)
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(1 - y, approved, sensitive_features=sensitive)
        ),
    }


def calibration_by_group(
    y_true: np.ndarray, pd_hat: np.ndarray, sensitive: np.ndarray, *, n_bins: int = 5
) -> pl.DataFrame:
    """Predicted vs observed default rate per group per score band.

    The question a regulator asks: does a 700 score mean the same real default
    rate for every group? Selection-rate parity can hold while this fails.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pd_hat, dtype=float)
    g = np.asarray(sensitive).astype(str)

    ranks = np.argsort(np.argsort(p, kind="mergesort"), kind="mergesort")
    band = np.minimum((ranks * n_bins) // max(len(p), 1), n_bins - 1)

    return (
        pl.DataFrame({"group": g, "band": band.astype(np.int32), "y": y, "pd": p})
        .group_by(["group", "band"])
        .agg(
            n=pl.len(),
            mean_predicted=pl.col("pd").mean(),
            observed=pl.col("y").mean(),
        )
        .with_columns(gap=pl.col("mean_predicted") - pl.col("observed"))
        .sort(["band", "group"])
    )


def age_band(days_birth: np.ndarray) -> np.ndarray:
    """ECOA-relevant age bands from Home Credit's negative day offsets."""
    years = -np.asarray(days_birth, dtype=float) / 365.25
    return np.select(
        [years < 25, years < 35, years < 45, years < 55, years < 65],
        ["18-24", "25-34", "35-44", "45-54", "55-64"],
        default="65+",
    )

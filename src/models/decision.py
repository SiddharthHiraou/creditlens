"""PD -> credit score -> decision.

The model produces a probability. A lender needs a *decision*, and the mapping
between them is policy, not modelling. Keeping it in its own module means the
cutoff can move without retraining, which is exactly how a real credit policy
committee operates.

Score scaling reuses the scorecard's PDO convention so the GBDT and the
scorecard land on the same 300-850 scale and are directly comparable:

    score = offset + factor * ln(odds_good)

Three outcomes, not two. A "refer" band exists because the applications nearest
the cutoff are the ones where a model is least certain and a human adds the most
value -- and because a straight approve/decline split gives an underwriter
nothing to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import polars as pl

from src.models.scorecard import BASE_ODDS, BASE_SCORE, PDO

SCORE_MIN = 300.0
SCORE_MAX = 850.0

_FACTOR = PDO / np.log(2)
_OFFSET = BASE_SCORE - _FACTOR * np.log(BASE_ODDS)


class Decision(StrEnum):
    APPROVE = "approve"
    REFER = "refer"
    DECLINE = "decline"


@dataclass(frozen=True)
class DecisionPolicy:
    """Cutoffs in score space, where a credit committee actually sets them.

    ``approve_at`` and ``refer_at`` are score thresholds: at or above
    ``approve_at`` is an approve, below ``refer_at`` is a decline, and the band
    between is referred to manual review.

    The defaults are calibrated to *this* book and are not universal. The
    600-at-50:1 PDO anchor fixes the score *scale*, not the cutoff: a 620 cutoff
    implies a PD under 1%, which on a portfolio with a 16.8% bad rate would
    decline essentially every applicant. Real lenders set the cutoff from the
    through-the-door score distribution and a target approval rate, which is
    what :meth:`from_approval_rate` does.
    """

    approve_at: float = 545.0
    refer_at: float = 528.0
    lgd: float = 0.65
    version: str = "policy-v1"

    def __post_init__(self) -> None:
        if self.refer_at > self.approve_at:
            raise ValueError(
                f"refer_at ({self.refer_at}) must not exceed approve_at ({self.approve_at})"
            )

    @classmethod
    def from_approval_rate(
        cls,
        scores: np.ndarray,
        *,
        approve_rate: float = 0.60,
        refer_rate: float = 0.10,
        lgd: float = 0.65,
        version: str = "policy-derived",
    ) -> DecisionPolicy:
        """Place cutoffs to hit a target approval rate on an observed population.

        ``approve_rate`` is the share approved outright; ``refer_rate`` is the
        additional share sent to manual review below it.
        """
        if not 0 < approve_rate < 1:
            raise ValueError(f"approve_rate must be in (0, 1), got {approve_rate}")
        if not 0 <= refer_rate < 1 - approve_rate:
            raise ValueError(
                f"refer_rate {refer_rate} must leave room below approve_rate {approve_rate}"
            )
        s = np.asarray(scores, dtype=float)
        return cls(
            approve_at=float(np.quantile(s, 1 - approve_rate)),
            refer_at=float(np.quantile(s, 1 - approve_rate - refer_rate)),
            lgd=lgd,
            version=version,
        )


def pd_to_score(pd_hat: np.ndarray | float) -> np.ndarray:
    """Map probability of default onto the 300-850 scale. Higher = safer."""
    p = np.clip(np.asarray(pd_hat, dtype=float), 1e-6, 1 - 1e-6)
    return np.clip(_OFFSET + _FACTOR * np.log((1 - p) / p), SCORE_MIN, SCORE_MAX)


def score_to_pd(score: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`pd_to_score`, used to express a cutoff as a PD."""
    s = np.asarray(score, dtype=float)
    odds_good = np.exp((s - _OFFSET) / _FACTOR)
    return 1.0 / (1.0 + odds_good)


def decide(score: np.ndarray, policy: DecisionPolicy) -> np.ndarray:
    """Vectorised approve / refer / decline."""
    s = np.asarray(score, dtype=float)
    return np.where(
        s >= policy.approve_at,
        Decision.APPROVE.value,
        np.where(s >= policy.refer_at, Decision.REFER.value, Decision.DECLINE.value),
    )


def expected_loss(pd_hat: np.ndarray, exposure: np.ndarray, policy: DecisionPolicy) -> np.ndarray:
    """EL = PD x LGD x EAD.

    This is why calibration is not optional. On the uncalibrated champion the
    mean PD was 0.43 against a true rate of 0.17, so every EL here would have
    been overstated by roughly 2.5x while AUC looked healthy.
    """
    return np.asarray(pd_hat, dtype=float) * policy.lgd * np.asarray(exposure, dtype=float)


def decision_frame(
    pd_hat: np.ndarray,
    policy: DecisionPolicy,
    *,
    exposure: np.ndarray | None = None,
    ids: np.ndarray | None = None,
) -> pl.DataFrame:
    """One row per application: PD, score, decision, and expected loss."""
    score = pd_to_score(pd_hat)
    data: dict[str, np.ndarray] = {
        "pd": np.asarray(pd_hat, dtype=float),
        "score": score,
        "decision": decide(score, policy),
    }
    if exposure is not None:
        data["exposure"] = np.asarray(exposure, dtype=float)
        data["expected_loss"] = expected_loss(pd_hat, exposure, policy)
    df = pl.DataFrame(data)
    if ids is not None:
        df = df.with_columns(SK_ID_CURR=pl.Series(ids)).select(
            ["SK_ID_CURR", *[c for c in df.columns]]
        )
    return df


def portfolio_summary(
    pd_hat: np.ndarray,
    y_true: np.ndarray,
    policy: DecisionPolicy,
    *,
    exposure: np.ndarray | None = None,
) -> pl.DataFrame:
    """Volume, bad rate and expected loss per decision band.

    The bad rate must fall from decline to approve. If it does not, the cutoff
    is placed somewhere the score does not separate.
    """
    score = pd_to_score(pd_hat)
    df = pl.DataFrame(
        {
            "decision": decide(score, policy),
            "y": np.asarray(y_true).astype(int),
            "pd": np.asarray(pd_hat, dtype=float),
            "exposure": np.asarray(exposure, dtype=float)
            if exposure is not None
            else np.ones(len(pd_hat)),
        }
    )
    order = {Decision.DECLINE.value: 0, Decision.REFER.value: 1, Decision.APPROVE.value: 2}
    return (
        df.group_by("decision")
        .agg(
            n=pl.len(),
            share=pl.len() / df.height,
            observed_bad_rate=pl.col("y").mean(),
            mean_pd=pl.col("pd").mean(),
            expected_loss=(pl.col("pd") * policy.lgd * pl.col("exposure")).sum(),
        )
        .with_columns(
            sort_key=pl.col("decision").replace_strict(order, default=99, return_dtype=pl.Int32)
        )
        .sort("sort_key")
        .drop("sort_key")
    )

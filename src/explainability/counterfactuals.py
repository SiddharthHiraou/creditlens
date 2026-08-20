"""Minimum feasible change that would flip a declined decision.

A reason code tells an applicant *why* they were declined. A counterfactual
tells them *what would have to be different* — which is the more useful half of
an adverse action notice and the part most implementations skip.

Search is a constrained one-dimensional sweep per actionable feature, then a
greedy combination. Constrained in three ways, each of which matters:

**Actionable features only.** You cannot tell someone to have had fewer
delinquencies last year. Past conduct, external bureau scores, employment
tenure and every protected or immutable attribute are excluded from the search
space entirely — not down-weighted, excluded. What remains is what an applicant
could plausibly change within a few months: how much they ask for, how it is
structured, and how much revolving balance they carry.

**Direction-constrained.** A proposal only ever moves a feature the way that
reduces risk. The monotonic constraint vector already encodes which way that is
for 47 features; the rest use an explicit direction.

**Bounded by observed data.** Proposals are clipped to the training
distribution's 1st-99th percentile. A recommendation to reduce a loan request
to a value no applicant has ever made is not a recommendation.

**Searched on the raw score, judged on the calibrated one.** The champion is
isotonic-calibrated, and isotonic regression is a *step function*: the
calibrated PD takes only as many distinct values as the fit has steps, so
sweeping a feature produces a staircase with long flat stretches and the search
has nothing to follow. The sweep therefore runs against the uncalibrated
(continuous) model output, while the target is the raw value whose calibrated
score reaches the cutoff -- obtainable because isotonic is monotone, so
``calibrated_score >= target`` is exactly ``raw_pd <= raw_threshold``. Reported
PDs are always the calibrated ones.

**Coupled, not independent.** This is the part most implementations get wrong.
Feature selection keeps *ratios* rather than raw amounts, so a naive
single-feature sweep would propose reducing ``RATIO_credit_to_income`` while
leaving ``RATIO_annuity_to_income`` untouched — an incoherent applicant who
borrows less but repays the same. Proposals are therefore expressed as
**levers**: one real-world action that propagates to every model feature it
would actually move, consistently. ``LEVERS`` defines them; the per-feature
sweep in :meth:`CounterfactualSearch.search` is retained only as a diagnostic
of the model's response surface, not as advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# Features a borrower can realistically move within a few months, with the
# direction that lowers predicted risk (-1 = decrease it, +1 = increase it).
ACTIONABLE: dict[str, int] = {
    # loan structure — ask for less, or spread it further
    "AMT_CREDIT": -1,
    "AMT_ANNUITY": -1,
    "RATIO_annuity_to_income": -1,
    "RATIO_credit_to_income": -1,
    "RATIO_goods_to_income": -1,
    "RATIO_annuity_to_credit": -1,
    "RATIO_credit_to_goods": -1,
    "RATIO_downpayment": +1,
    "RATIO_residual_income": +1,
    "RATIO_residual_income_per_member": +1,
    "XSRC_new_credit_to_bureau_debt": -1,
    "XSRC_total_debt_to_income": -1,
    "XSRC_bureau_debt_to_income": -1,
    # revolving balances — pay them down
    "CC_util_mean": -1,
    "CC_util_max": -1,
    "CC_util_latest": -1,
    "CC_util_mean_3m": -1,
    "CC_share_months_over_90pct": -1,
    "CC_n_months_over_limit": -1,
    "CC_balance_mean": -1,
    "CC_balance_max": -1,
    # outstanding debt — repay it
    "BURO_debt_total": -1,
    "BURO_active_debt_total": -1,
    "BURO_debt_to_credit": -1,
    "BURO_active_debt_to_credit": -1,
    "BURO_overdue_total": -1,
    "BURO_active_overdue_total": -1,
}

# The subset that maps one-to-one onto a single instruction an applicant can
# act on, and whose knock-on features move in the same direction anyway.
ACTIONABLE_LEVERS = frozenset(
    {
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "CC_util_mean",
        "CC_util_max",
        "CC_util_latest",
        "CC_balance_mean",
        "BURO_debt_total",
        "BURO_overdue_total",
        "BURO_active_overdue_total",
    }
)

# Never proposed, under any circumstances. Immutable, protected, or historical.
IMMUTABLE = frozenset(
    {
        "DAYS_BIRTH",
        "STAB_age_years",
        "STAB_age_band",
        "CODE_GENDER",
        "NAME_FAMILY_STATUS",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "NAME_EDUCATION_TYPE",
        "DAYS_EMPLOYED",
        "STAB_employed_years",
        "OCCUPATION_TYPE",
        "REGION_RATING_CLIENT",
    }
)


# ---------------------------------------------------------------------------
# Levers: one real-world action -> every model feature it actually moves.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lever:
    """A single action an applicant can take, and its effect on the features.

    ``apply`` receives the feature row and a magnitude ``t`` in (0, 1] and
    returns a modified copy. Magnitude means whatever is natural for the
    action: for "borrow less" it is the fractional reduction in the amount.
    """

    key: str
    description: str
    max_magnitude: float

    def apply(self, row: np.ndarray, names: list[str], t: float) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _scale(row: np.ndarray, names: list[str], feature: str, factor: float) -> None:
        if feature in names:
            i = names.index(feature)
            if np.isfinite(row[i]):
                row[i] = row[i] * factor

    @staticmethod
    def _get(row: np.ndarray, names: list[str], feature: str) -> float | None:
        if feature not in names:
            return None
        v = float(row[names.index(feature)])
        return v if np.isfinite(v) else None

    @staticmethod
    def _set(row: np.ndarray, names: list[str], feature: str, value: float) -> None:
        if feature in names:
            row[names.index(feature)] = value


class BorrowLess(Lever):
    """Request a smaller loan.

    Every amount-derived ratio scales by (1 - t) together, and residual income
    rises by the annuity that is no longer owed. Deriving the annuity as
    ``RATIO_annuity_to_income x AMT_INCOME_TOTAL`` keeps the residual-income
    update exact rather than approximate.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "key", "borrow_less")
        object.__setattr__(self, "description", "Request a smaller loan amount")
        object.__setattr__(self, "max_magnitude", 0.60)

    def apply(self, row: np.ndarray, names: list[str], t: float) -> np.ndarray:
        out = row.copy()
        factor = 1.0 - t
        annuity_ratio = self._get(out, names, "RATIO_annuity_to_income")
        income = self._get(out, names, "AMT_INCOME_TOTAL")

        for f in (
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "RATIO_credit_to_income",
            "RATIO_annuity_to_income",
            "RATIO_goods_to_income",
            "XSRC_new_credit_to_bureau_debt",
            "XSRC_credit_to_prev_credit",
            "XSRC_annuity_to_prev_annuity",
        ):
            self._scale(out, names, f, factor)

        # residual income = income - 12 * annuity, so it rises by the annuity saved
        if annuity_ratio is not None and income is not None:
            saved = t * 12.0 * annuity_ratio * income
            for f in ("RATIO_residual_income",):
                cur = self._get(out, names, f)
                if cur is not None:
                    self._set(out, names, f, cur + saved)
            members = self._get(out, names, "CNT_FAM_MEMBERS") or 1.0
            cur = self._get(out, names, "RATIO_residual_income_per_member")
            if cur is not None:
                self._set(
                    out,
                    names,
                    "RATIO_residual_income_per_member",
                    cur + saved / max(members, 1.0),
                )

        # total debt to income falls by the share attributable to the new loan
        credit_ratio = self._get(out, names, "RATIO_credit_to_income")
        total = self._get(out, names, "XSRC_total_debt_to_income")
        if credit_ratio is not None and total is not None:
            self._set(out, names, "XSRC_total_debt_to_income", max(total - t * credit_ratio, 0.0))
        return out


class PayDownRevolving(Lever):
    """Reduce revolving card balances before reapplying."""

    def __init__(self) -> None:
        object.__setattr__(self, "key", "pay_down_revolving")
        object.__setattr__(self, "description", "Reduce outstanding revolving card balances")
        object.__setattr__(self, "max_magnitude", 1.0)

    def apply(self, row: np.ndarray, names: list[str], t: float) -> np.ndarray:
        out = row.copy()
        factor = 1.0 - t
        for f in (
            "CC_util_mean",
            "CC_util_max",
            "CC_util_latest",
            "CC_util_mean_3m",
            "CC_util_mean_6m",
            "CC_util_mean_12m",
            "CC_util_max_3m",
            "CC_util_max_6m",
            "CC_util_max_12m",
            "CC_balance_mean",
            "CC_balance_max",
            "CC_share_months_over_90pct",
            "CC_util_recent_vs_overall",
        ):
            self._scale(out, names, f, factor)
        if t > 0.5:
            self._set(out, names, "CC_n_months_over_limit", 0.0)
        return out


class ClearArrears(Lever):
    """Bring past-due balances current."""

    def __init__(self) -> None:
        object.__setattr__(self, "key", "clear_arrears")
        object.__setattr__(self, "description", "Bring past-due balances current")
        object.__setattr__(self, "max_magnitude", 1.0)

    def apply(self, row: np.ndarray, names: list[str], t: float) -> np.ndarray:
        out = row.copy()
        factor = 1.0 - t
        for f in (
            "BURO_overdue_total",
            "BURO_overdue_max",
            "BURO_active_overdue_total",
            "BURO_overdue_to_debt",
            "BURO_overdue_recency_wtd",
            "BURO_overdue_line_share",
            "BURO_n_overdue_lines",
            "BURO_days_overdue_max",
            "BURO_days_overdue_mean",
        ):
            self._scale(out, names, f, factor)
        return out


class RepayExistingDebt(Lever):
    """Repay existing non-revolving obligations."""

    def __init__(self) -> None:
        object.__setattr__(self, "key", "repay_existing_debt")
        object.__setattr__(self, "description", "Repay part of the existing outstanding debt")
        object.__setattr__(self, "max_magnitude", 0.75)

    def apply(self, row: np.ndarray, names: list[str], t: float) -> np.ndarray:
        out = row.copy()
        factor = 1.0 - t
        for f in (
            "BURO_debt_total",
            "BURO_debt_mean",
            "BURO_debt_max",
            "BURO_active_debt_total",
            "BURO_debt_to_credit",
            "BURO_active_debt_to_credit",
            "BURO_debt_recency_wtd",
            "BURO_debt_recency_wtd_norm",
            "XSRC_bureau_debt_to_income",
            "XSRC_total_debt_to_income",
        ):
            self._scale(out, names, f, factor)
        cur = self._get(out, names, "XSRC_new_credit_to_bureau_debt")
        if cur is not None and factor > 1e-9:
            self._set(out, names, "XSRC_new_credit_to_bureau_debt", cur / factor)
        return out


LEVERS: tuple[Lever, ...] = (BorrowLess(), PayDownRevolving(), ClearArrears(), RepayExistingDebt())


@dataclass(frozen=True)
class LeverProposal:
    """The minimum magnitude of one action that reaches the target score."""

    lever: str
    description: str
    magnitude: float
    pd_before: float
    pd_after: float
    score_before: float
    score_after: float
    flips_decision: bool
    features_moved: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "lever": self.lever,
            "description": self.description,
            "magnitude": round(self.magnitude, 4),
            "pd_before": round(self.pd_before, 6),
            "pd_after": round(self.pd_after, 6),
            "score_before": round(self.score_before, 1),
            "score_after": round(self.score_after, 1),
            "flips_decision": self.flips_decision,
            "features_moved": list(self.features_moved),
        }


@dataclass(frozen=True)
class Counterfactual:
    feature: str
    current_value: float
    proposed_value: float
    direction: str
    pd_before: float
    pd_after: float
    score_before: float
    score_after: float
    flips_decision: bool
    is_single_action: bool

    @property
    def relative_change(self) -> float:
        if abs(self.current_value) < 1e-12:
            return float("inf")
        return (self.proposed_value - self.current_value) / abs(self.current_value)

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "direction": self.direction,
            "relative_change": self.relative_change,
            "pd_before": self.pd_before,
            "pd_after": self.pd_after,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "flips_decision": self.flips_decision,
            "is_single_action": self.is_single_action,
        }


@dataclass
class CounterfactualSearch:
    """Bounded, direction-constrained search over actionable features."""

    model: object
    feature_names: list[str]
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_reference(
        cls, model, feature_names: list[str], reference: np.ndarray, *, clip: float = 1.0
    ) -> CounterfactualSearch:
        """Bounds taken from the reference (training) population's tails."""
        ref = np.asarray(reference, dtype=float)
        return cls(
            model=model,
            feature_names=list(feature_names),
            lower=np.nanpercentile(ref, clip, axis=0),
            upper=np.nanpercentile(ref, 100 - clip, axis=0),
        )

    def _pd(self, x: np.ndarray) -> np.ndarray:
        """Calibrated PD — what gets reported and what the decision uses."""
        return np.asarray(self.model.predict_pd(np.asarray(x, dtype=np.float32)))

    def _raw_pd(self, x: np.ndarray) -> np.ndarray:
        """Uncalibrated PD — continuous, so the search has a surface to climb."""
        fn = getattr(self.model, "predict_pd_uncalibrated", None)
        if fn is None:
            return self._pd(x)
        return np.asarray(fn(np.asarray(x, dtype=np.float32)))

    def raw_threshold(self, target_score: float, *, grid: int = 4000) -> float:
        """Largest raw PD whose calibrated score still reaches ``target_score``.

        Isotonic is monotone, so the calibrated decision boundary corresponds to
        a single raw-PD threshold. Recovering it lets the search work on the
        continuous surface while still answering the calibrated question.
        """
        from src.models.decision import pd_to_score

        calibrator = getattr(self.model, "calibrator", None)
        raw_grid = np.linspace(1e-6, 1 - 1e-6, grid)
        cal = calibrator.predict(raw_grid) if calibrator is not None else raw_grid
        ok = pd_to_score(np.clip(cal, 1e-6, 1 - 1e-6)) >= target_score
        if not ok.any():
            return 0.0
        return float(raw_grid[ok].max())

    def search(
        self,
        x_row: np.ndarray,
        target_score: float,
        *,
        steps: int = 24,
        single_action_only: bool = False,
    ) -> list[Counterfactual]:
        """One proposal per actionable feature, cheapest change first.

        Each feature is swept from its current value toward its bound, and the
        first value reaching ``target_score`` is reported. Features that cannot
        reach it alone are reported with their best achievable improvement and
        ``flips_decision=False``.
        """
        from src.models.decision import pd_to_score

        row = np.asarray(x_row, dtype=np.float32).ravel()
        pd_before = float(self._pd(row.reshape(1, -1))[0])
        score_before = float(pd_to_score(pd_before))

        results: list[Counterfactual] = []
        for name, direction in ACTIONABLE.items():
            if name in IMMUTABLE or name not in self.feature_names:
                continue
            if single_action_only and name not in ACTIONABLE_LEVERS:
                continue
            i = self.feature_names.index(name)
            current = float(row[i])
            if not np.isfinite(current):
                continue  # no value to move

            bound = float(self.upper[i] if direction > 0 else self.lower[i])
            if (direction > 0 and bound <= current) or (direction < 0 and bound >= current):
                continue  # already at or beyond the feasible edge

            grid = np.linspace(current, bound, steps + 1)[1:]
            probe = np.tile(row, (len(grid), 1))
            probe[:, i] = grid
            pds = self._pd(probe)
            scores = pd_to_score(pds)

            hit = np.where(scores >= target_score)[0]
            idx = int(hit[0]) if hit.size else int(np.argmax(scores))
            results.append(
                Counterfactual(
                    feature=name,
                    current_value=current,
                    proposed_value=float(grid[idx]),
                    direction="decrease" if direction < 0 else "increase",
                    pd_before=pd_before,
                    pd_after=float(pds[idx]),
                    score_before=score_before,
                    score_after=float(scores[idx]),
                    flips_decision=bool(hit.size),
                    is_single_action=name in ACTIONABLE_LEVERS,
                )
            )

        # Feasible single-feature flips first, ranked by how little has to move;
        # then the near-misses, ranked by how far they got.
        return sorted(
            results,
            key=lambda c: (not c.flips_decision, abs(c.relative_change), -c.score_after),
        )

    # -- lever search: the interface that produces applicant-facing advice ----

    def propose_levers(
        self, x_row: np.ndarray, target_score: float, *, steps: int = 30
    ) -> list[LeverProposal]:
        """Minimum magnitude of each single action that reaches the target.

        Returns one proposal per lever, feasible ones first and ranked by how
        little the applicant has to do. A lever that cannot reach the target
        alone is still returned, with its best achievable score, so the caller
        can stack actions or tell the applicant plainly that no single change
        is enough.
        """
        from src.models.decision import pd_to_score

        row = np.asarray(x_row, dtype=np.float32).ravel()
        pd_before = float(self._pd(row.reshape(1, -1))[0])
        score_before = float(pd_to_score(pd_before))
        raw_target = self.raw_threshold(target_score)

        out: list[LeverProposal] = []
        for lever in LEVERS:
            magnitudes = np.linspace(0.0, lever.max_magnitude, steps + 1)[1:]
            probes = np.vstack([lever.apply(row, self.feature_names, float(t)) for t in magnitudes])
            moved = tuple(
                self.feature_names[i] for i in np.where(~_equal_with_nan(probes[-1], row))[0]
            )
            if not moved:
                continue  # this lever touches nothing the model uses

            probes = self._clip(probes, origin=row)
            raw = self._raw_pd(probes)

            # A proposal must never leave the applicant worse off than doing
            # nothing. Magnitudes that raise the raw PD are discarded outright
            # rather than ranked, so an unhelpful lever is reported as absent
            # instead of as bad advice.
            raw_before = float(self._raw_pd(row.reshape(1, -1))[0])
            usable = np.where(raw <= raw_before + 1e-12)[0]
            if usable.size == 0:
                continue

            # Smallest magnitude that crosses the raw threshold; failing that,
            # the magnitude that gets the raw PD lowest.
            hit = usable[raw[usable] <= raw_target]
            idx = int(hit[0]) if hit.size else int(usable[np.argmin(raw[usable])])
            cal_pd = float(self._pd(probes[idx].reshape(1, -1))[0])
            score_after = float(pd_to_score(cal_pd))
            out.append(
                LeverProposal(
                    lever=lever.key,
                    description=lever.description,
                    magnitude=float(magnitudes[idx]),
                    pd_before=pd_before,
                    pd_after=cal_pd,
                    score_before=score_before,
                    score_after=score_after,
                    flips_decision=bool(score_after >= target_score),
                    features_moved=moved,
                )
            )

        return sorted(out, key=lambda c: (not c.flips_decision, c.magnitude, -c.score_after))

    def stack_levers(
        self, x_row: np.ndarray, target_score: float, *, max_actions: int = 3, steps: int = 30
    ) -> tuple[list[LeverProposal], bool]:
        """Apply levers greedily until the decision flips or the budget runs out.

        The common outcome for a deep decline is that no single action suffices,
        and saying so honestly is more useful than proposing an impossible one.
        """
        row = np.asarray(x_row, dtype=np.float32).ravel().copy()
        used: set[str] = set()
        chosen: list[LeverProposal] = []

        for _ in range(max_actions):
            options = [
                p
                for p in self.propose_levers(row, target_score, steps=steps)
                if p.lever not in used
            ]
            if not options:
                break
            best = min(options, key=lambda c: c.pd_after)
            if best.pd_after >= best.pd_before - 1e-12:
                break  # nothing left that helps
            lever = next(lv for lv in LEVERS if lv.key == best.lever)
            row = self._clip(
                lever.apply(row, self.feature_names, best.magnitude).reshape(1, -1),
                origin=row,
            )[0].astype(np.float32)
            used.add(best.lever)
            chosen.append(best)
            if best.flips_decision:
                return chosen, True
        return chosen, False

    def _clip(self, x: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
        """Hold proposals inside the observed training range.

        A recommendation to reach a value no applicant has ever presented is
        not a recommendation. NaNs are preserved -- absent history stays absent.

        The bounds are widened to include the applicant's own starting value.
        Without that, an applicant already *better* than the 1st percentile on
        some feature gets dragged back toward it: the lever lowers the value,
        the clip raises it past where it began, and the "proposal" comes back
        with a higher PD than doing nothing. Clipping exists to keep proposals
        realistic, not to penalise someone for starting ahead of the pack.
        """
        arr = np.atleast_2d(np.asarray(x, dtype=float)).copy()
        lo = np.broadcast_to(self.lower, arr.shape).copy()
        hi = np.broadcast_to(self.upper, arr.shape).copy()

        if origin is not None:
            o = np.asarray(origin, dtype=float).ravel()
            finite_o = np.isfinite(o)
            lo[:, finite_o] = np.minimum(lo[:, finite_o], o[finite_o])
            hi[:, finite_o] = np.maximum(hi[:, finite_o], o[finite_o])

        finite = np.isfinite(arr)
        arr[finite] = np.clip(arr[finite], lo[finite], hi[finite])
        return arr.astype(np.float32)

    def greedy_combination(
        self, x_row: np.ndarray, target_score: float, *, max_changes: int = 3, steps: int = 24
    ) -> tuple[list[Counterfactual], bool]:
        """Stack changes until the decision flips or the budget runs out.

        Used when no single feature can flip the decision on its own, which is
        the common case for a deep decline.
        """
        from src.models.decision import pd_to_score

        row = np.asarray(x_row, dtype=np.float32).ravel().copy()
        chosen: list[Counterfactual] = []
        for _ in range(max_changes):
            options = self.search(row, target_score, steps=steps)
            if not options:
                break
            best = max(options, key=lambda c: c.score_after)
            if best.score_after <= float(pd_to_score(self._pd(row.reshape(1, -1))[0])) + 1e-9:
                break  # no further improvement available
            row[self.feature_names.index(best.feature)] = best.proposed_value
            chosen.append(best)
            if best.flips_decision:
                return chosen, True
        return chosen, False


def to_frame(results: list[Counterfactual]) -> pl.DataFrame:
    if not results:
        return pl.DataFrame(
            schema={
                "feature": pl.Utf8,
                "current_value": pl.Float64,
                "proposed_value": pl.Float64,
                "direction": pl.Utf8,
                "relative_change": pl.Float64,
                "pd_before": pl.Float64,
                "pd_after": pl.Float64,
                "score_before": pl.Float64,
                "score_after": pl.Float64,
                "flips_decision": pl.Boolean,
                "is_single_action": pl.Boolean,
            }
        )
    return pl.DataFrame([r.as_dict() for r in results])


def _equal_with_nan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise equality treating NaN == NaN as True."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    return (a == b) | both_nan


def levers_to_frame(results: list[LeverProposal]) -> pl.DataFrame:
    if not results:
        return pl.DataFrame(
            schema={
                "lever": pl.Utf8,
                "description": pl.Utf8,
                "magnitude": pl.Float64,
                "pd_before": pl.Float64,
                "pd_after": pl.Float64,
                "score_before": pl.Float64,
                "score_after": pl.Float64,
                "flips_decision": pl.Boolean,
                "features_moved": pl.List(pl.Utf8),
            }
        )
    return pl.DataFrame([r.as_dict() for r in results])

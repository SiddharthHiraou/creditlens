"""Pydantic v2 request and response contracts.

Validation here is the first line of defence against a garbage-in decision.
Every field a model depends on carries a range, and the ranges are the ones the
Pandera ingestion schemas already enforce — a payload that would have been
rejected at training time is rejected at serving time too.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Money = Annotated[float, Field(gt=0, le=1e9)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class ApplicationIn(BaseModel):
    """A single credit application.

    Only application-level fields are supplied by the caller. Bureau and
    repayment history is looked up server-side from the feature cache by
    ``sk_id_curr`` — the client neither has it nor should be trusted with it.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sk_id_curr: int = Field(ge=1, description="Applicant identifier; keys the history lookup.")
    amt_income_total: Money
    amt_credit: Money
    amt_annuity: Money
    amt_goods_price: Money | None = None
    days_birth: int = Field(
        le=-6574, ge=-36525, description="Negative days from application; 18-100 years."
    )
    days_employed: int = Field(le=0, ge=-36525)
    days_id_publish: int = Field(default=-1000, le=0, ge=-36525)
    cnt_children: int = Field(default=0, ge=0, le=20)
    cnt_fam_members: int = Field(default=1, ge=1, le=25)
    region_rating_client: int = Field(default=2, ge=1, le=3)
    name_contract_type: Literal["Cash loans", "Revolving loans"] = "Cash loans"
    code_gender: Literal["F", "M", "XNA"] = "XNA"
    flag_own_car: Literal["Y", "N"] = "N"
    flag_own_realty: Literal["Y", "N"] = "N"
    name_education_type: str = "Secondary / secondary special"
    name_family_status: str = "Single / not married"
    occupation_type: str = "Laborers"
    ext_source_1: Probability | None = None
    ext_source_2: Probability | None = None
    ext_source_3: Probability | None = None

    @field_validator("cnt_fam_members")
    @classmethod
    def family_must_include_children(cls, v: int, info) -> int:
        children = info.data.get("cnt_children", 0)
        if v < children + 1:
            raise ValueError(
                f"cnt_fam_members ({v}) must be at least cnt_children + 1 ({children + 1})"
            )
        return v

    @field_validator("amt_goods_price")
    @classmethod
    def goods_within_an_order_of_credit(cls, v: float | None, info) -> float | None:
        credit = info.data.get("amt_credit")
        if v is not None and credit is not None and not (0.1 * credit <= v <= 10 * credit):
            raise ValueError(
                f"amt_goods_price ({v:,.0f}) is implausible against amt_credit ({credit:,.0f})"
            )
        return v


class ReasonCodeOut(BaseModel):
    rank: int
    family: str
    label: str
    phrase: str
    actionable: bool
    contribution: float
    driving_features: list[str]


class ShapContribution(BaseModel):
    feature: str
    value: float | None
    shap: float


class ScoreOut(BaseModel):
    """The decision record returned to the caller and written to the audit log."""

    decision_id: str
    sk_id_curr: int
    pd: Probability
    score: float = Field(ge=300, le=850)
    decision: Literal["approve", "refer", "decline"]
    expected_loss: float
    reason_codes: list[ReasonCodeOut]
    shap_values: list[ShapContribution]
    model_version: str
    feature_spec_fingerprint: str
    policy_version: str
    history_found: bool = Field(
        description="False when the applicant had no cached history and was scored thin-file."
    )
    latency_ms: float
    scored_at: dt.datetime


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applications: list[ApplicationIn] = Field(min_length=1)


class BatchAccepted(BaseModel):
    job_id: str
    n_submitted: int
    status: Literal["queued", "running", "complete", "failed"]
    submitted_at: dt.datetime


class BatchStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "failed"]
    n_submitted: int
    n_scored: int
    submitted_at: dt.datetime
    completed_at: dt.datetime | None = None
    error: str | None = None
    results: list[ScoreOut] | None = None


class OverrideIn(BaseModel):
    """An underwriter overriding the model.

    Justification is mandatory and length-checked. An override with no reason
    is unreviewable, and the audit trail is the entire point of recording it.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    new_decision: Literal["approve", "refer", "decline"]
    justification: str = Field(min_length=20, max_length=2000)
    underwriter_id: str = Field(min_length=1, max_length=64)


class DecisionRecord(BaseModel):
    decision_id: str
    sk_id_curr: int
    pd: float
    score: float
    decision: str
    model_version: str
    feature_spec_fingerprint: str
    policy_version: str
    reason_codes: list[ReasonCodeOut]
    features: dict[str, float | None]
    scored_at: dt.datetime
    overridden: bool = False
    override_decision: str | None = None
    override_justification: str | None = None
    override_by: str | None = None
    overridden_at: dt.datetime | None = None


class ModelMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_type: str
    trained_at: str | None
    feature_spec_version: int
    feature_spec_fingerprint: str
    n_features: int
    n_monotonic_constraints: int
    features: list[str]
    performance: dict[str, float]
    calibration: dict[str, float]
    serving_backend: Literal["onnx", "native"]
    reason_code_mapping_version: int


class DriftOut(BaseModel):
    score_psi: float
    verdict: str
    is_alarm: bool
    baseline_mean_pd: float
    current_mean_pd: float
    n_baseline: int
    n_current: int
    worst_features: list[dict]
    computed_at: dt.datetime


class CutoffSimulationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve_at: float | None = Field(default=None, ge=300, le=850)
    approve_rate: float | None = Field(default=None, gt=0, lt=1)
    refer_band: float = Field(default=0.10, ge=0, lt=0.5)
    lgd: float = Field(default=0.65, gt=0, le=1)
    interest_margin: float = Field(
        default=0.12, gt=0, le=1, description="Annual margin earned on a performing loan."
    )

    @model_validator(mode="after")
    def exactly_one_target(self) -> CutoffSimulationIn:
        """Require exactly one of approve_at / approve_rate.

        A model validator, not a field validator: Pydantic v2 skips field
        validators for fields left at their default, so the "neither supplied"
        case silently passed and the endpoint fell through to a 500.
        """
        if (self.approve_rate is None) == (self.approve_at is None):
            raise ValueError("Supply exactly one of approve_at or approve_rate.")
        return self


class CutoffSimulationOut(BaseModel):
    approve_at: float
    refer_at: float
    approval_rate: float
    refer_rate: float
    bad_rate_among_approved: float
    expected_loss: float
    interest_income: float
    estimated_profit: float
    n_evaluated: int
    bands: list[dict]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str


class ReadyOut(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    feature_spec_loaded: bool
    database: bool
    cache: str
    detail: str | None = None

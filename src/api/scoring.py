"""The serving path: payload + cached history -> decision + reasons.

This module is the one place where training and serving could silently diverge,
so it derives everything it can from shared code rather than reimplementing it:
the application-level feature functions are the *same* ones the training
pipeline calls, and the column order comes from the feature spec's fingerprint
rather than from a hardcoded list.

What the request supplies: application-level fields only.
What the server supplies: the 161 relational history features, from the cache.
A caller must not be able to assert their own bureau record.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from src.api.schemas import ApplicationIn, ReasonCodeOut, ScoreOut, ShapContribution
from src.explainability.reason_codes import ReasonCodeMapper
from src.features.application import application_features
from src.features.spec import FeatureSpec
from src.models.decision import DecisionPolicy, decide, expected_loss, pd_to_score

# Payload field -> the column name the feature pipeline expects.
_PAYLOAD_TO_COLUMN = {
    "sk_id_curr": "SK_ID_CURR",
    "amt_income_total": "AMT_INCOME_TOTAL",
    "amt_credit": "AMT_CREDIT",
    "amt_annuity": "AMT_ANNUITY",
    "amt_goods_price": "AMT_GOODS_PRICE",
    "days_birth": "DAYS_BIRTH",
    "days_employed": "DAYS_EMPLOYED",
    "days_id_publish": "DAYS_ID_PUBLISH",
    "cnt_children": "CNT_CHILDREN",
    "cnt_fam_members": "CNT_FAM_MEMBERS",
    "region_rating_client": "REGION_RATING_CLIENT",
    "name_contract_type": "NAME_CONTRACT_TYPE",
    "code_gender": "CODE_GENDER",
    "flag_own_car": "FLAG_OWN_CAR",
    "flag_own_realty": "FLAG_OWN_REALTY",
    "name_education_type": "NAME_EDUCATION_TYPE",
    "name_family_status": "NAME_FAMILY_STATUS",
    "occupation_type": "OCCUPATION_TYPE",
    "ext_source_1": "EXT_SOURCE_1",
    "ext_source_2": "EXT_SOURCE_2",
    "ext_source_3": "EXT_SOURCE_3",
}


@dataclass
class ScoringService:
    model: Any
    spec: FeatureSpec
    mapper: ReasonCodeMapper
    policy: DecisionPolicy
    cache: Any
    shap_service: Any | None = None
    model_version: str = "unknown"
    shap_top_k: int = 10

    # -- feature assembly ----------------------------------------------------

    def _application_frame(self, application: ApplicationIn) -> pl.DataFrame:
        """Payload -> a one-row frame with the training column names.

        ``AMT_GOODS_PRICE`` defaults to the credit amount when absent, matching
        how the ratio features behave for a cash loan with no goods attached.
        """
        payload = application.model_dump()
        row = {column: payload.get(field) for field, column in _PAYLOAD_TO_COLUMN.items()}
        if row["AMT_GOODS_PRICE"] is None:
            row["AMT_GOODS_PRICE"] = row["AMT_CREDIT"]
        return pl.DataFrame([row])

    def build_vector(self, application: ApplicationIn) -> tuple[np.ndarray, dict, bool]:
        """Assemble the spec-ordered feature vector for one application."""
        frame = self._application_frame(application)
        # The same transformer the training pipeline uses. Reimplementing these
        # ratios here is the classic way training and serving drift apart.
        derived = application_features(frame.lazy()).collect()

        history = self.cache.get(application.sk_id_curr)
        history_found = history is not None

        values: dict[str, float | None] = {}
        derived_columns = set(derived.columns)
        for name in self.spec.features:
            if name in derived_columns:
                raw = derived[name][0]
            elif history_found:
                raw = history.get(name)
            else:
                raw = None  # thin file: absent history stays absent
            values[name] = None if raw is None else float(raw)

        vector = np.array(
            [np.nan if v is None else v for v in values.values()], dtype=np.float32
        ).reshape(1, -1)
        return vector, values, history_found

    # -- scoring -------------------------------------------------------------

    def score(
        self,
        application: ApplicationIn,
        *,
        decision_id: str,
        explain: str = "auto",
    ) -> tuple[ScoreOut, dict]:
        """Score one application.

        ``explain`` controls the SHAP pass, which is ~92% of the request's cost
        (6.6ms against 0.08ms for the prediction itself):

        * ``auto`` — the default. The decision is computed first, and SHAP runs
          only when the applicant was not approved. An approval needs no adverse
          action reasons, so on a 60%-approval book this skips the expensive
          work on most requests.
        * ``always`` — compute regardless, for an underwriter inspecting an
          approval.
        * ``never`` — skip entirely, for bulk rescoring where only the decision
          matters.
        """
        started = time.perf_counter()

        vector, values, history_found = self.build_vector(application)
        pd_hat = float(np.asarray(self.model.predict_pd(vector)).ravel()[0])
        score = float(np.asarray(pd_to_score(pd_hat)).ravel()[0])
        decision = str(np.asarray(decide(np.array([score]), self.policy)).ravel()[0])
        loss = float(
            np.asarray(
                expected_loss(np.array([pd_hat]), np.array([application.amt_credit]), self.policy)
            ).ravel()[0]
        )

        reason_codes: list[ReasonCodeOut] = []
        shap_out: list[ShapContribution] = []
        wants_shap = explain == "always" or (explain == "auto" and decision != "approve")
        if self.shap_service is not None and wants_shap:
            contributions = self.shap_service.values(vector)[0]
            # Reason codes are an adverse action artifact. An approved
            # applicant has not been denied anything, so issuing them "reasons"
            # would be both meaningless and, if it ever reached a letter,
            # misleading.
            if decision != "approve":
                reason_codes = [
                    ReasonCodeOut(**rc.as_dict())
                    for rc in self.mapper.explain(self.spec.features, contributions)
                ]
            order = np.argsort(-np.abs(contributions))[: self.shap_top_k]
            shap_out = [
                ShapContribution(
                    feature=self.spec.features[i],
                    value=values[self.spec.features[i]],
                    shap=float(contributions[i]),
                )
                for i in order
            ]

        latency_ms = (time.perf_counter() - started) * 1000
        scored_at = dt.datetime.now(dt.UTC).replace(tzinfo=None)

        response = ScoreOut(
            decision_id=decision_id,
            sk_id_curr=application.sk_id_curr,
            pd=pd_hat,
            score=score,
            decision=decision,
            expected_loss=loss,
            reason_codes=reason_codes,
            shap_values=shap_out,
            model_version=self.model_version,
            feature_spec_fingerprint=self.spec.fingerprint,
            policy_version=self.policy.version,
            history_found=history_found,
            latency_ms=round(latency_ms, 3),
            scored_at=scored_at,
        )

        audit_row = {
            "decision_id": decision_id,
            "sk_id_curr": application.sk_id_curr,
            "pd": pd_hat,
            "score": score,
            "decision": decision,
            "expected_loss": loss,
            "model_version": self.model_version,
            "feature_spec_fingerprint": self.spec.fingerprint,
            "policy_version": self.policy.version,
            "reason_codes": [rc.model_dump() for rc in reason_codes],
            "features": values,
            "history_found": history_found,
            "latency_ms": round(latency_ms, 3),
            "scored_at": scored_at,
        }
        return response, audit_row

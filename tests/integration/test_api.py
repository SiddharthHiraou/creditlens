"""API contract tests. Every endpoint, every auth path, every error shape.

Runs against TestClient with SQLite and the in-process cache, so the suite needs
nothing running. The latency numbers in the README come from Locust against a
real uvicorn server instead — TestClient bypasses the ASGI server and network
stack and would flatter the tail.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.settings import Settings
from src.config import ARTIFACTS

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "champion_model.joblib").exists(),
    reason="champion not trained; run `make train` first",
)

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("api") / "audit.db"
    settings = Settings(
        api_keys=KEY,
        database_url=f"sqlite:///{db}",
        redis_url=None,
        rate_limit_per_minute=100_000,
        warm_cache_on_startup=True,
    )
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="module")
def payload():
    path = ARTIFACTS / "sample_payloads.json"
    if not path.exists():
        pytest.skip("no sample payloads; run `make warm-cache`")
    return json.loads(path.read_text())[0]


# -- health -----------------------------------------------------------------


def test_health_needs_no_api_key(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_every_dependency(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["model_loaded"] and body["feature_spec_loaded"] and body["database"]
    # The cache must name its real backend; a silent downgrade is worse than none.
    assert body["cache"] in {"memory", "redis"} or body["cache"].startswith("memory")


# -- auth and rate limiting --------------------------------------------------


def test_scoring_requires_an_api_key(client, payload):
    assert client.post("/v1/score", json=payload).status_code == 401


def test_a_wrong_api_key_is_rejected(client, payload):
    r = client.post("/v1/score", json=payload, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_rate_limit_returns_429_with_retry_after(payload, tmp_path):
    settings = Settings(
        api_keys=KEY,
        database_url=f"sqlite:///{tmp_path / 'rl.db'}",
        redis_url=None,
        rate_limit_per_minute=3,
        warm_cache_on_startup=False,
    )
    with TestClient(create_app(settings)) as c:
        codes = [c.post("/v1/score", json=payload, headers=HEADERS).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_every_response_carries_a_request_id(client):
    r = client.get("/health")
    assert r.headers.get("X-Request-ID")


def test_a_supplied_request_id_is_echoed_back(client):
    r = client.get("/health", headers={"X-Request-ID": "trace-me-123"})
    assert r.headers["X-Request-ID"] == "trace-me-123"


# -- scoring ------------------------------------------------------------------


def test_score_returns_a_complete_decision(client, payload):
    body = client.post("/v1/score", json=payload, headers=HEADERS).json()
    assert 0.0 <= body["pd"] <= 1.0
    assert 300 <= body["score"] <= 850
    assert body["decision"] in {"approve", "refer", "decline"}
    assert body["expected_loss"] >= 0
    assert body["model_version"] and body["feature_spec_fingerprint"]
    assert body["latency_ms"] > 0
    assert body["history_found"] is True


def test_a_declined_application_carries_four_distinct_reasons(client, payload):
    """The Phase 3 guarantee, enforced at the API boundary."""
    body = client.post("/v1/score", json=payload, headers=HEADERS).json()
    if body["decision"] == "approve":
        pytest.skip("sample applicant is approved")
    assert len(body["reason_codes"]) == 4
    assert len({rc["family"] for rc in body["reason_codes"]}) == 4
    assert [rc["rank"] for rc in body["reason_codes"]] == [1, 2, 3, 4]


def test_an_approval_is_not_given_adverse_action_reasons(client):
    """Reason codes are an adverse action artifact; an approval was not denied."""
    payloads = json.loads((ARTIFACTS / "sample_payloads.json").read_text())
    for p in payloads[:40]:
        body = client.post("/v1/score", json=p, headers=HEADERS).json()
        if body["decision"] == "approve":
            assert body["reason_codes"] == []
            return
    pytest.skip("no approvals in the sample")


def test_explain_never_skips_the_expensive_pass(client, payload):
    body = client.post("/v1/score?explain=never", json=payload, headers=HEADERS).json()
    assert body["shap_values"] == []
    assert body["reason_codes"] == []


def test_explain_always_returns_shap_even_for_an_approval(client):
    payloads = json.loads((ARTIFACTS / "sample_payloads.json").read_text())
    for p in payloads[:40]:
        body = client.post("/v1/score?explain=always", json=p, headers=HEADERS).json()
        if body["decision"] == "approve":
            assert body["shap_values"]
            assert body["reason_codes"] == []  # still no adverse action reasons
            return
    pytest.skip("no approvals in the sample")


def test_an_unknown_applicant_is_scored_thin_file_not_rejected(client, payload):
    """A cache miss means no history, which is a real segment, not an error."""
    body = client.post(
        "/v1/score", json={**payload, "sk_id_curr": 999_999_999}, headers=HEADERS
    ).json()
    assert body["history_found"] is False
    assert 0.0 <= body["pd"] <= 1.0


def test_invalid_payloads_are_rejected_with_422(client, payload):
    for broken, field in (
        ({**payload, "days_birth": -100}, "days_birth"),
        ({**payload, "amt_income_total": -5}, "amt_income_total"),
        ({**payload, "ext_source_2": 1.5}, "ext_source_2"),
        ({**payload, "code_gender": "Z"}, "code_gender"),
    ):
        r = client.post("/v1/score", json=broken, headers=HEADERS)
        assert r.status_code == 422, field


def test_unknown_fields_are_rejected_rather_than_ignored(client, payload):
    """A typo'd field silently ignored is a silently wrong decision."""
    r = client.post("/v1/score", json={**payload, "amt_incom_total": 1}, headers=HEADERS)
    assert r.status_code == 422


# -- audit trail ---------------------------------------------------------------


def test_a_decision_is_retrievable_and_fully_reconstructible(client, payload):
    scored = client.post("/v1/score", json=payload, headers=HEADERS).json()
    record = client.get(f"/v1/decisions/{scored['decision_id']}", headers=HEADERS).json()

    assert record["decision"] == scored["decision"]
    assert record["pd"] == pytest.approx(scored["pd"])
    assert record["model_version"] == scored["model_version"]
    assert record["feature_spec_fingerprint"] == scored["feature_spec_fingerprint"]
    # The full feature vector, so a reason code can be re-derived later.
    assert len(record["features"]) > 50
    assert record["overridden"] is False


def test_an_unknown_decision_id_is_404(client):
    assert client.get("/v1/decisions/not-a-real-id", headers=HEADERS).status_code == 404


def test_override_is_recorded_without_mutating_the_original(client, payload):
    scored = client.post("/v1/score", json=payload, headers=HEADERS).json()
    original = scored["decision"]
    new = "approve" if original != "approve" else "decline"

    r = client.post(
        f"/v1/decisions/{scored['decision_id']}/override",
        json={
            "new_decision": new,
            "justification": "Applicant supplied verified additional income documentation.",
            "underwriter_id": "uw-42",
        },
        headers=HEADERS,
    )
    assert r.status_code == 200
    record = r.json()
    assert record["overridden"] is True
    assert record["override_decision"] == new
    assert record["override_by"] == "uw-42"
    # The model's own decision must survive intact; it is the evidence.
    assert record["decision"] == original


def test_override_requires_a_substantive_justification(client, payload):
    scored = client.post("/v1/score", json=payload, headers=HEADERS).json()
    r = client.post(
        f"/v1/decisions/{scored['decision_id']}/override",
        json={"new_decision": "approve", "justification": "ok", "underwriter_id": "uw-1"},
        headers=HEADERS,
    )
    assert r.status_code == 422


def test_overriding_to_the_same_decision_is_a_conflict(client, payload):
    scored = client.post("/v1/score", json=payload, headers=HEADERS).json()
    r = client.post(
        f"/v1/decisions/{scored['decision_id']}/override",
        json={
            "new_decision": scored["decision"],
            "justification": "No change intended, testing the guard behaviour here.",
            "underwriter_id": "uw-1",
        },
        headers=HEADERS,
    )
    assert r.status_code == 409


# -- batch ---------------------------------------------------------------------


def test_batch_accepts_and_completes(client, payload):
    payloads = json.loads((ARTIFACTS / "sample_payloads.json").read_text())[:5]
    r = client.post("/v1/score/batch", json={"applications": payloads}, headers=HEADERS)
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    status = client.get(f"/v1/score/batch/{job_id}", headers=HEADERS).json()
    assert status["status"] == "complete"
    assert status["n_scored"] == 5
    assert len(status["results"]) == 5


def test_an_oversized_batch_is_refused(client, payload):
    settings_limit = 1000
    r = client.post(
        "/v1/score/batch",
        json={"applications": [payload] * (settings_limit + 1)},
        headers=HEADERS,
    )
    assert r.status_code == 413


def test_an_unknown_job_id_is_404(client):
    assert client.get("/v1/score/batch/nope", headers=HEADERS).status_code == 404


# -- metadata, drift, simulation ------------------------------------------------


def test_model_metadata_describes_what_is_actually_serving(client):
    body = client.get("/v1/model/metadata", headers=HEADERS).json()
    assert body["n_features"] == len(body["features"])
    assert body["serving_backend"] in {"onnx", "native"}
    assert body["n_monotonic_constraints"] > 0
    assert body["feature_spec_fingerprint"]
    assert body["performance"]["auc"] > 0.7


def test_drift_reports_psi_against_the_training_baseline(client, payload):
    payloads = json.loads((ARTIFACTS / "sample_payloads.json").read_text())
    for p in payloads[:40]:
        client.post("/v1/score?explain=never", json=p, headers=HEADERS)
    body = client.get("/v1/monitoring/drift", headers=HEADERS).json()
    assert body["score_psi"] >= 0
    assert body["verdict"] in {"stable", "moderate shift", "significant shift"}
    assert body["n_current"] >= 30


def test_cutoff_simulation_by_target_approval_rate(client):
    body = client.post("/v1/simulate/cutoff", json={"approve_rate": 0.6}, headers=HEADERS).json()
    assert body["approval_rate"] == pytest.approx(0.6, abs=0.02)
    assert 300 <= body["approve_at"] <= 850
    assert body["refer_at"] <= body["approve_at"]
    assert body["estimated_profit"] == pytest.approx(
        body["interest_income"] - body["expected_loss"]
    )


def test_a_stricter_cutoff_approves_fewer_and_better_loans(client):
    strict = client.post("/v1/simulate/cutoff", json={"approve_rate": 0.4}, headers=HEADERS).json()
    loose = client.post("/v1/simulate/cutoff", json={"approve_rate": 0.8}, headers=HEADERS).json()
    assert strict["approval_rate"] < loose["approval_rate"]
    assert strict["bad_rate_among_approved"] < loose["bad_rate_among_approved"]


def test_simulation_requires_exactly_one_target(client):
    both = client.post(
        "/v1/simulate/cutoff", json={"approve_rate": 0.6, "approve_at": 550}, headers=HEADERS
    )
    neither = client.post("/v1/simulate/cutoff", json={}, headers=HEADERS)
    assert both.status_code == 422
    assert neither.status_code == 422


def test_profit_curve_identifies_a_maximum(client):
    body = client.get("/v1/simulate/cutoff/curve", headers=HEADERS).json()
    assert body["points"]
    assert body["profit_maximising"] is not None
    best = body["profit_maximising"]["profit"]
    assert all(p["profit"] <= best + 1e-6 for p in body["points"])


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/v1/score" in schema["paths"]
    assert "/v1/decisions/{decision_id}/override" in schema["paths"]


def test_policy_cutoff_is_on_the_score_scale_not_the_pd_scale(client):
    """Regression guard for a bug that approved everyone.

    Cutoffs are quantiles of the 300-850 score distribution. Deriving them from
    the 0-1 PD distribution instead puts the threshold near 0.05, which every
    score clears — so the API approves every applicant and still looks healthy
    on every other check.
    """
    payloads = json.loads((ARTIFACTS / "sample_payloads.json").read_text())
    decisions = [
        client.post("/v1/score?explain=never", json=p, headers=HEADERS).json()["decision"]
        for p in payloads[:80]
    ]
    assert len(set(decisions)) > 1, f"all {len(decisions)} applicants got '{decisions[0]}'"
    approve_share = decisions.count("approve") / len(decisions)
    assert 0.2 < approve_share < 0.9, f"approval share {approve_share:.0%} is not a real policy"


def test_score_and_decision_are_consistent_with_the_policy(client):
    """A higher score must never receive a worse decision."""
    payloads = json.loads((ARTIFACTS / "sample_payloads.json").read_text())
    rank = {"decline": 0, "refer": 1, "approve": 2}
    scored = sorted(
        (r["score"], rank[r["decision"]])
        for r in (
            client.post("/v1/score?explain=never", json=p, headers=HEADERS).json()
            for p in payloads[:80]
        )
    )
    ranks = [r for _, r in scored]
    assert ranks == sorted(ranks), "decision bands are not monotone in score"


# -- adverse action memo --------------------------------------------------------


def _first_non_approval(client):
    payloads = json.loads((ARTIFACTS / "sample_payloads.json").read_text())
    for p in payloads[:40]:
        body = client.post("/v1/score", json=p, headers=HEADERS).json()
        if body["decision"] != "approve":
            return body
    pytest.skip("no non-approvals in the sample")


def test_memo_is_generated_from_the_stored_decision(client):
    scored = _first_non_approval(client)
    body = client.post(
        "/v1/memo", json={"decision_id": scored["decision_id"]}, headers=HEADERS
    ).json()

    assert body["decision_id"] == scored["decision_id"]
    assert body["memo"]["summary"] and body["memo"]["detail"] and body["memo"]["next_steps"]
    assert body["prompt_hash"] and body["prompt_version"]


def test_memo_cites_only_families_the_decision_carried(client):
    """The grounding rule, enforced at the API boundary."""
    scored = _first_non_approval(client)
    body = client.post(
        "/v1/memo", json={"decision_id": scored["decision_id"]}, headers=HEADERS
    ).json()

    issued = {rc["family"] for rc in scored["reason_codes"]}
    assert set(body["memo"]["reason_families_cited"]) <= issued


def test_memo_states_the_actual_requested_amount(client):
    """Regression guard for a real bug.

    AMT_CREDIT is not in the feature spec — selection pruned it — so the amount
    cannot come from the stored feature vector. Before `exposure` was recorded
    on the decision, every memo read "the application for 1".
    """
    scored = _first_non_approval(client)
    record = client.get(f"/v1/decisions/{scored['decision_id']}", headers=HEADERS).json()
    amount = record["exposure"]
    assert amount and amount > 1000

    body = client.post(
        "/v1/memo", json={"decision_id": scored["decision_id"]}, headers=HEADERS
    ).json()
    assert f"{round(amount):,}" in body["memo"]["summary"]


def test_memo_is_refused_for_an_approval(client):
    payloads = json.loads((ARTIFACTS / "sample_payloads.json").read_text())
    for p in payloads[:40]:
        body = client.post("/v1/score", json=p, headers=HEADERS).json()
        if body["decision"] == "approve":
            r = client.post("/v1/memo", json={"decision_id": body["decision_id"]}, headers=HEADERS)
            assert r.status_code == 409
            return
    pytest.skip("no approvals in the sample")


def test_memo_is_refused_when_the_decision_carries_no_reasons(client):
    payloads = json.loads((ARTIFACTS / "sample_payloads.json").read_text())
    for p in payloads[:40]:
        body = client.post("/v1/score?explain=never", json=p, headers=HEADERS).json()
        if body["decision"] != "approve":
            r = client.post("/v1/memo", json={"decision_id": body["decision_id"]}, headers=HEADERS)
            assert r.status_code == 409
            return
    pytest.skip("no non-approvals in the sample")


def test_memo_for_an_unknown_decision_is_404(client):
    r = client.post("/v1/memo", json={"decision_id": "nope"}, headers=HEADERS)
    assert r.status_code == 404


def test_memo_requires_auth(client):
    assert client.post("/v1/memo", json={"decision_id": "x"}).status_code == 401

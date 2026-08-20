"""The three tools the analyst copilot may call.

The guardrail that matters most is in :func:`query_portfolio_stats`: **the model
never emits SQL.** It picks a named query from a whitelist and supplies typed
parameters. Letting a model write SQL against a production database — even
read-only, even with a careful prompt — makes the prompt the security boundary,
and a prompt is not a security boundary.

The other two tools are read-only by construction: one reads the MLflow registry,
one retrieves passages from documents in this repository.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ARTIFACTS, DOCS

# ---------------------------------------------------------------------------
# Tool 1 — portfolio statistics over a whitelist of parameterised queries
# ---------------------------------------------------------------------------

# Each entry is a named, fixed query. The model chooses a name and supplies
# parameters; it cannot compose, extend, or inject SQL.
PORTFOLIO_QUERIES: dict[str, str] = {
    "decision_mix": "Counts and shares by decision band at the current policy.",
    "bad_rate_by_band": "Observed bad rate and mean predicted PD per decision band.",
    "approval_rate_at_cutoff": "Approval rate, bad rate and expected loss at a given score cutoff.",
    "score_percentiles": "Score distribution percentiles across the out-of-time fold.",
    "bad_rate_by_vintage": "Observed bad rate and mean predicted PD by origination cohort.",
    "worst_drifting_features": "Features with the highest characteristic stability index.",
    "reason_code_frequency": "How often each reason-code family appears among declines.",
}


class UnknownQueryError(ValueError):
    pass


def _holdout() -> dict[str, np.ndarray]:
    path = ARTIFACTS / "serving_holdout.npz"
    if not path.exists():
        raise FileNotFoundError("No serving holdout. Run `make train`.")
    data = np.load(path)
    return {"score": data["score"], "y": data["y"], "exposure": data["exposure"]}


def _demo(name: str) -> Any:
    path = Path(__file__).resolve().parents[2] / "frontend" / "public" / "data" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No {name}.json. Run `make demo-data`.")
    return json.loads(path.read_text())


def query_portfolio_stats(query_name: str, **params: Any) -> dict:
    """Run one whitelisted portfolio query.

    Args:
        query_name: One of ``PORTFOLIO_QUERIES``.
        **params: Typed parameters for that query.
    """
    if query_name not in PORTFOLIO_QUERIES:
        raise UnknownQueryError(
            f"Unknown query {query_name!r}. Available: {sorted(PORTFOLIO_QUERIES)}"
        )

    if query_name == "decision_mix":
        return {"bands": _demo("portfolio")["bands"]}

    if query_name == "bad_rate_by_band":
        return {
            "bands": [
                {k: b[k] for k in ("decision", "n", "badRate", "meanPd")}
                for b in _demo("portfolio")["bands"]
            ]
        }

    if query_name == "approval_rate_at_cutoff":
        cutoff = float(params.get("score_cutoff", 539.0))
        lgd = float(params.get("lgd", 0.65))
        h = _holdout()
        approved = h["score"] >= cutoff
        if not approved.any():
            return {"score_cutoff": cutoff, "approval_rate": 0.0, "n_approved": 0}
        from src.models.decision import score_to_pd

        pd_hat = score_to_pd(h["score"])
        return {
            "score_cutoff": cutoff,
            "approval_rate": round(float(approved.mean()), 4),
            "n_approved": int(approved.sum()),
            "bad_rate_among_approved": round(float(h["y"][approved].mean()), 4),
            "expected_loss": round(
                float((pd_hat[approved] * lgd * h["exposure"][approved]).sum()), 2
            ),
        }

    if query_name == "score_percentiles":
        h = _holdout()
        qs = params.get("percentiles", [1, 10, 25, 50, 75, 90, 99])
        return {
            "percentiles": {str(q): round(float(np.percentile(h["score"], q)), 1) for q in qs},
            "n": int(h["score"].size),
        }

    if query_name == "bad_rate_by_vintage":
        return {"vintages": _demo("portfolio")["vintages"]}

    if query_name == "worst_drifting_features":
        top = int(params.get("top_n", 10))
        return {"features": _demo("monitoring")["featureCsi"][:top]}

    if query_name == "reason_code_frequency":
        counts: Counter[str] = Counter()
        for applicant in _demo("applicants"):
            for rc in applicant.get("reasonCodes", []):
                counts[rc["family"]] += 1
        total = sum(counts.values()) or 1
        return {
            "families": [
                {"family": f, "n": n, "share": round(n / total, 4)} for f, n in counts.most_common()
            ],
            "n_declines_sampled": len([a for a in _demo("applicants") if a.get("reasonCodes")]),
        }

    raise UnknownQueryError(query_name)  # pragma: no cover - exhaustive above


# ---------------------------------------------------------------------------
# Tool 2 — model metrics from the registry
# ---------------------------------------------------------------------------


def get_model_metrics(metric_names: list[str] | None = None) -> dict:
    """Read the champion's registered metrics.

    Prefers the MLflow registry so the answer reflects what is actually serving
    rather than a number pasted into a document. Falls back to the metrics JSON
    the training run wrote when no tracking store is present.
    """
    metrics: dict[str, Any] = {}
    source = "artifacts"

    db = ARTIFACTS / "mlflow.db"
    if db.exists():
        try:
            import mlflow

            mlflow.set_tracking_uri(f"sqlite:///{db}")
            client = mlflow.MlflowClient()
            experiments = [e for e in client.search_experiments() if e.name == "creditlens"]
            if experiments:
                runs = client.search_runs([experiments[0].experiment_id], max_results=1)
                if runs:
                    metrics = dict(runs[0].data.metrics)
                    source = "mlflow"
        except Exception:  # noqa: BLE001 - fall back rather than fail the tool
            metrics = {}

    if not metrics:
        payload = json.loads((ARTIFACTS / "phase2_metrics.json").read_text())
        champion = payload["champion_calibrated_test"]
        metrics = {
            "oot_auc": champion["auc"],
            "oot_gini": champion["gini"],
            "oot_ks": champion["ks"],
            "oot_brier": champion["brier"],
            "score_psi": payload["score_psi_train_vs_oot"],
            **{f"{name}_oot_auc": r["test_oot"]["auc"] for name, r in payload["metrics"].items()},
        }

    if metric_names:
        metrics = {k: v for k, v in metrics.items() if k in set(metric_names)}

    return {
        "source": source,
        "metrics": {k: round(float(v), 6) for k, v in sorted(metrics.items())},
    }


# ---------------------------------------------------------------------------
# Tool 3 — retrieval over the written policy and the model card
# ---------------------------------------------------------------------------

CORPUS_FILES = ("credit_policy.md", "target_definition.md", "fairness_findings.md")

_TOKEN = re.compile(r"[a-z0-9]+")

# Longest-first so "ing" is tried before "s" and "retraining" -> "retrain".
_SUFFIXES = ("ations", "ation", "ingly", "ing", "edly", "ies", "ed", "es", "ly", "s")


def _stem(token: str) -> str:
    """Crude suffix stripping.

    Not a real stemmer, and it does not need to be. Without it BM25 fails on
    plain morphology -- "retrain" does not match "retraining", so a question
    about retraining retrieves nothing from the section that governs it. Six
    lines of suffix stripping fixes the common cases over a corpus this small.
    """
    for suffix in _SUFFIXES:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokenise(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower())]


@dataclass(frozen=True)
class Passage:
    source: str
    heading: str
    text: str


def _split_sections(path: Path) -> list[Passage]:
    """Split a markdown document on headings.

    Section-level chunks rather than fixed windows: a policy clause is a unit of
    meaning, and cutting one in half produces retrievals that answer nothing.
    """
    lines = path.read_text().splitlines()
    passages: list[Passage] = []
    heading = path.stem
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if len(body) > 40:
            passages.append(Passage(source=path.name, heading=heading, text=body))

    for line in lines:
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
            buffer = []
        else:
            buffer.append(line)
    flush()
    return passages


def _corpus() -> list[Passage]:
    out: list[Passage] = []
    for name in CORPUS_FILES:
        path = DOCS / name
        if path.exists():
            out.extend(_split_sections(path))
    return out


def search_credit_policy(question: str, top_k: int = 3) -> dict:
    """Retrieve the policy passages most relevant to a question.

    Scored with BM25 over the section corpus. The brief specifies pgvector for
    production, and that is the right destination — dense retrieval handles
    paraphrase, which BM25 does not. BM25 is used here because it needs no
    embedding service, is deterministic, and is genuinely adequate over a
    corpus of roughly two hundred sections written in the same vocabulary as
    the questions.
    """
    passages = _corpus()
    if not passages:
        return {"question": question, "passages": [], "note": "corpus unavailable"}

    # Heading terms count triple. A section titled "3.2 Thin bureau file" is a
    # far stronger signal for a question about thin bureau files than a preamble
    # that happens to repeat the vocabulary, and flat weighting ranks the
    # preamble first.
    docs = [_tokenise(p.text) + _tokenise(p.heading) * 3 for p in passages]
    query = _tokenise(question)

    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n, 1)
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))

    k1, b = 1.5, 0.75
    scores = []
    for doc in docs:
        counts = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query:
            if term not in counts:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            tf = counts[term]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
        scores.append(score)

    order = np.argsort(scores)[::-1][:top_k]
    return {
        "question": question,
        "passages": [
            {
                "source": passages[i].source,
                "heading": passages[i].heading,
                "score": round(float(scores[i]), 3),
                "text": passages[i].text[:1200],
            }
            for i in order
            if scores[i] > 0
        ],
    }


TOOL_REGISTRY = {
    "query_portfolio_stats": query_portfolio_stats,
    "get_model_metrics": get_model_metrics,
    "search_credit_policy": search_credit_policy,
}

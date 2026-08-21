"""Portfolio analyst copilot.

A tool-calling agent over three read-only tools. Runs on the mid tier because
the work is genuinely open-ended: deciding which statistic answers a question,
and reading a policy clause back accurately, is not templated.

**What the copilot cannot do**, enforced by construction rather than by asking:

* It cannot write SQL. ``query_portfolio_stats`` takes a query *name* from a
  whitelist plus typed parameters. A prompt is not a security boundary.
* It cannot make or change a decision. It has no tool that writes anything.
* It cannot see an individual applicant's record. No tool exposes one.

Every answer is logged with the model, the prompt hash, the tools called and the
cost.

Note on parameters: the mid tier removed ``temperature`` — sending it returns a
400. The memo generator sets ``temperature=0`` because it runs on Haiku, where
the parameter still exists. Those two facts look inconsistent and are not.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

from src.llm.client import (
    COPILOT_MODEL,
    CredentialsUnavailableError,
    Usage,
    cost_usd,
    credentials_available,
    get_client,
    log_call,
    prompt_hash,
)
from src.llm.schemas import CopilotAnswer
from src.llm.tools import (
    PORTFOLIO_QUERIES,
    get_model_metrics,
    query_portfolio_stats,
    search_credit_policy,
)

PROMPT_VERSION = "copilot-v1"
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "copilot_system.txt").read_text()
MAX_TOOL_TURNS = 6


def _build_tools():
    """Wrap the three functions as SDK tools.

    Defined inside a function so importing this module does not require the
    ``anthropic`` package — the offline path and the tests should not need it.
    """
    from anthropic import beta_tool

    @beta_tool
    def portfolio_stats(query_name: str, params_json: str = "{}") -> str:
        """Run a named portfolio query and return its result as JSON.

        Args:
            query_name: One of the whitelisted query names.
            params_json: JSON object of parameters for that query, e.g.
                {"score_cutoff": 560} or {"top_n": 5}.
        """
        import json

        params = json.loads(params_json) if params_json else {}
        return json.dumps(query_portfolio_stats(query_name, **params), default=str)

    @beta_tool
    def model_metrics(metric_names_csv: str = "") -> str:
        """Return the champion model's registered performance metrics as JSON.

        Args:
            metric_names_csv: Optional comma-separated metric names to filter to.
        """
        import json

        names = [n.strip() for n in metric_names_csv.split(",") if n.strip()] or None
        return json.dumps(get_model_metrics(names), default=str)

    @beta_tool
    def credit_policy(question: str, top_k: int = 3) -> str:
        """Retrieve relevant passages from the credit policy and governance docs.

        Args:
            question: The analyst's question, in their own words.
            top_k: How many passages to return.
        """
        import json

        return json.dumps(search_credit_policy(question, top_k=top_k), default=str)

    return [portfolio_stats, model_metrics, credit_policy]


def _system_prompt() -> str:
    """System prompt plus the live query whitelist.

    Injecting the whitelist means the model is told exactly what it may ask for,
    rather than guessing a query name and getting an error back.
    """
    catalogue = "\n".join(f"- {name}: {desc}" for name, desc in PORTFOLIO_QUERIES.items())
    return f"{SYSTEM_PROMPT}\n\nAvailable portfolio queries:\n{catalogue}\n"


def ask(question: str, *, allow_offline: bool = True) -> CopilotAnswer:
    """Answer one analyst question."""
    started = time.perf_counter()
    system = _system_prompt()
    digest = prompt_hash(system, question, PROMPT_VERSION)

    if not credentials_available():
        if not allow_offline:
            raise CredentialsUnavailableError("No Anthropic credentials and allow_offline=False.")
        answer, calls = _offline_answer(question)
        result = CopilotAnswer(
            question=question,
            answer=answer,
            tools_called=calls,
            model="offline-retrieval",
            prompt_hash=digest,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            generated_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
            offline=True,
        )
        _log(result)
        return result

    client = get_client()
    runner = client.beta.messages.tool_runner(
        model=COPILOT_MODEL,
        max_tokens=4096,
        # No `temperature` here: it is removed on this tier and returns a 400.
        system=system,
        tools=_build_tools(),
        messages=[{"role": "user", "content": question}],
        max_iterations=MAX_TOOL_TURNS,
    )
    final = runner.until_done()

    calls: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for message in getattr(runner, "messages", []) or []:
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "tool_use":
                calls.append({"tool": block.name, "input": block.input})
    for block in final.content:
        if block.type == "text":
            text_parts.append(block.text)

    usage = Usage.from_response(final)
    result = CopilotAnswer(
        question=question,
        answer="\n".join(text_parts).strip(),
        tools_called=calls,
        model=COPILOT_MODEL,
        prompt_hash=digest,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost_usd(COPILOT_MODEL, usage.input_tokens, usage.output_tokens),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        generated_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
        offline=False,
    )
    _log(result)
    return result


def _log(result: CopilotAnswer) -> None:
    log_call(
        {
            "kind": "copilot",
            "model": result.model,
            "prompt_hash": result.prompt_hash,
            "tools_called": [c["tool"] for c in result.tools_called],
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "offline": result.offline,
        }
    )


def _offline_answer(question: str) -> tuple[str, list[dict]]:
    """Retrieval-only response used when no credentials are configured.

    Returns the passages a model would have been handed, clearly labelled as
    retrieval rather than an answer. Presenting stitched-together passages as a
    synthesised answer would misrepresent what produced them.
    """
    hits = search_credit_policy(question, top_k=3)
    calls = [{"tool": "credit_policy", "input": {"question": question}}]

    if not hits["passages"]:
        return (
            "No Anthropic credentials are configured, so this is retrieval only "
            "and no passages matched the question.",
            calls,
        )

    lines = [
        "No Anthropic credentials are configured, so this is retrieval output "
        "rather than a synthesised answer. The most relevant passages are:",
        "",
    ]
    for passage in hits["passages"]:
        lines.append(f"{passage['source']} · {passage['heading']}")
        lines.append(f"  {passage['text'][:400].strip()}")
        lines.append("")
    return "\n".join(lines).strip(), calls

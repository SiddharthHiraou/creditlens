# ADR 0010: The LLM narrates; it never decides, and it never writes SQL

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 6

## Context

Two GenAI features: an adverse action memo generator and a portfolio analyst
copilot. Both sit next to a regulated decision process, which makes the
guardrails more consequential than the prompts.

## Decisions

### Two tiers, chosen for the workload

Memos are high volume and templated → **Claude Haiku 4.5**. The copilot answers
open questions over statistics and policy text and needs real reasoning →
**Claude Sonnet 5**. Unit economics: $0.0025 per memo, and since only
non-approvals receive one, **$1.00 per 1,000 decisions** at a 40% non-approval
rate. Copilot usage is analyst-driven and does not scale with decision volume,
so it is priced separately rather than folded into the per-decision figure.

An inconsistency that is not one: the memo generator sets `temperature=0` and the
copilot sets no temperature at all. The parameter was removed on the mid tier and
returns a 400 there; it still exists on Haiku, where identical inputs should
produce an identical memo.

### The memo generator cannot invent a reason

Enforced three ways rather than asked for once:

1. The prompt states it.
2. The output is parsed into a Pydantic model that **rejects decision language** —
   "I recommend approving" fails validation, it is not softened.
3. `validate_grounding` checks every cited reason family against the families the
   decision actually carried. A memo citing one it was not given is **discarded**,
   not repaired.

The API route reinforces this: the memo is generated from the **stored decision**,
read back out of the audit log, never from a payload the caller supplies. A caller
cannot hand the model a reason the decision did not carry.

Approvals are refused with a 409. An adverse action notice explains a denial;
there is nothing adverse about an approval.

### The copilot cannot write SQL

`query_portfolio_stats` takes a query **name** from a whitelist plus typed
parameters. The model never emits SQL. Letting a model write queries against a
production database — read-only, careful prompt, all of it — makes the prompt the
security boundary, and a prompt is not a security boundary.

None of the three tools writes anything, and none exposes an individual
applicant's record.

### Prompts receive banded figures

Income reaches the memo prompt as a 25k band. The prompt never receives the
feature vector or the date of birth, and the audit log records the decision id,
prompt hash and accounting — never applicant attributes. A log that quietly
accumulates incomes and birth dates is a privacy incident waiting to be found.

## Consequences

**No credentials were available in this build, so live calls are unverified.**
Both features have deterministic offline paths — a template memo and
retrieval-only copilot output — subject to the same validation and clearly
flagged `offline=True`. The dashboard says so on the page rather than presenting
template text as generated. This is the one part of the project whose live
behaviour has not been exercised, and it should not be described as if it had.

**Retrieval is BM25, not pgvector.** The brief specifies pgvector and that is the
right destination — dense retrieval handles paraphrase, which BM25 does not.
BM25 is used here because it needs no embedding service and is deterministic. Two
concessions were needed to make it adequate: heading terms are weighted triple
(otherwise a preamble section outranks the clause that actually answers the
question), and crude suffix stripping was added (otherwise "retrain" does not
match "retraining" and a question about retraining retrieves nothing from the
section that governs it).

**Every memo is attributable.** Model, prompt version and prompt hash are stored
with the output, so a memo issued months ago traces to the exact prompt that
produced it even after the prompt has been rewritten.

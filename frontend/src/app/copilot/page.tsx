import { Badge, Card, Cell, Metric, Note, PageHeader, Row, Table } from "@/components/ui";
import { getGenai } from "@/lib/data";
import { num } from "@/lib/format";

export const metadata = { title: "Copilot — CreditLens" };

export default function CopilotPage() {
  const genai = getGenai();
  const econ = genai.unitEconomics;

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Analyst"
        title="Portfolio copilot and memo drafting"
        lede="Two grounded LLM features. A memo generator that narrates reason codes an underwriter can send, and a tool-calling copilot that answers questions over portfolio statistics, model metrics and the written credit policy."
      />

      {!genai.credentialsConfigured && (
        <Note tone="warn">
          <strong>No Anthropic credentials are configured in this build.</strong> The
          examples below come from the deterministic offline paths — a template memo and
          retrieval-only copilot output — not from a model. They are labelled as such
          rather than presented as generated text. With a key set, the same code calls
          the API and the label disappears.
        </Note>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="Memo tier" value="Haiku 4.5" hint="High volume, templated, low temperature." />
        <Metric label="Copilot tier" value="Sonnet 5" hint="Open questions over policy and stats." />
        <Metric label="Cost per memo" value={`$${econ.cost_per_memo_usd.toFixed(4)}`} />
        <Metric
          label="Cost per 1,000 decisions"
          value={`$${econ.cost_per_1000_decisions_usd.toFixed(2)}`}
          hint={`${econ.memos_per_1000_decisions} memos — only non-approvals get one.`}
        />
      </div>

      <Card
        title="Adverse action memos"
        subtitle="The model narrates reason codes it was handed. It cannot introduce one, and it cannot change the decision."
      >
        <div className="space-y-5">
          {genai.memos.map((m) => (
            <article key={m.skIdCurr} className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <header className="mb-3 flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs">applicant {m.skIdCurr}</span>
                <Badge tone={m.decision === "refer" ? "warn" : "bad"}>{m.decision}</Badge>
                <Badge>{m.model}</Badge>
                <span className="ml-auto font-mono text-[10px] text-[var(--text-muted)]">
                  {m.promptVersion} · {m.promptHash}
                </span>
              </header>
              <p className="text-sm leading-relaxed">{m.memo.summary}</p>
              <p className="mt-2 text-sm leading-relaxed text-[var(--text-muted)]">{m.memo.detail}</p>
              <p className="mt-2 text-sm leading-relaxed text-[var(--text-muted)]">{m.memo.next_steps}</p>
              <p className="mt-3 font-mono text-[10px] text-[var(--text-muted)]">
                cited: {m.memo.reason_families_cited.join(" · ")}
              </p>
            </article>
          ))}
        </div>
        <div className="mt-5 space-y-3">
          <Note>
            Every memo is validated before it is returned. The output is parsed into a
            schema that rejects decision language, and a grounding check discards any memo
            citing a reason family the decision did not carry. A failed memo is thrown
            away, not repaired.
          </Note>
          <Note>
            The prompt receives the decision, the reason codes, the requested amount and a
            25k income band. It never receives the applicant&apos;s feature vector or date
            of birth. Every call is logged with the model, prompt hash and cost.
          </Note>
        </div>
      </Card>

      <Card
        title="Analyst copilot"
        subtitle="Three read-only tools. The model picks a named query and supplies parameters — it never writes SQL."
      >
        <div className="space-y-5">
          {genai.copilot.map((a) => (
            <article key={a.question} className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <p className="text-sm font-medium">{a.question}</p>
              <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-xs leading-relaxed text-[var(--text-muted)]">
                {a.answer}
              </pre>
              <p className="mt-3 flex flex-wrap items-center gap-2 font-mono text-[10px] text-[var(--text-muted)]">
                <Badge>{a.model}</Badge>
                {a.toolsCalled.map((t) => (
                  <Badge key={t} tone="accent">{t}</Badge>
                ))}
              </p>
            </article>
          ))}
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Whitelisted portfolio queries" subtitle="The complete set the copilot may call.">
          <Table head={["Query", "Returns"]} dense>
            {genai.portfolioQueries.map((q) => (
              <Row key={q.name}>
                <Cell align="left" mono={false}>
                  <span className="font-mono text-xs">{q.name}</span>
                </Cell>
                <Cell align="left" mono={false}>
                  <span className="text-xs text-[var(--text-muted)]">{q.description}</span>
                </Cell>
              </Row>
            ))}
          </Table>
          <div className="mt-4">
            <Note>
              The model chooses a name from this list and supplies typed parameters. It
              cannot compose, extend or inject SQL, because it never emits SQL. Letting a
              model write queries against a production database — even read-only, even
              with a careful prompt — makes the prompt the security boundary, and a prompt
              is not a security boundary.
            </Note>
          </div>
        </Card>

        <Card title="Guardrails">
          <ul className="space-y-2 text-sm leading-relaxed text-[var(--text-muted)]">
            <li>• The LLM never sets, changes or overrides a lending decision. Neither feature has a tool that writes anything.</li>
            <li>• The memo generator receives a decision already made and writes prose about it.</li>
            <li>• Reason families cited are checked against the families issued; a mismatch discards the memo.</li>
            <li>• Prompts receive banded figures, never raw applicant records.</li>
            <li>• Every call is logged with model, prompt hash, token counts and cost.</li>
            <li>• Retrieval is over documents in this repository, and every passage cites its source and heading.</li>
          </ul>
        </Card>
      </div>

      <Card title="Unit economics" subtitle="Why the tiers are split the way they are.">
        <Table head={["", "Memo generator", "Analyst copilot"]}>
          <Row><Cell align="left" mono={false}>Model</Cell><Cell>claude-haiku-4-5</Cell><Cell>claude-sonnet-5</Cell></Row>
          <Row><Cell align="left" mono={false}>Input / output per 1M</Cell><Cell>$1.00 / $5.00</Cell><Cell>$2.00 / $10.00</Cell></Row>
          <Row><Cell align="left" mono={false}>Volume driver</Cell><Cell align="left" mono={false}>Decisions</Cell><Cell align="left" mono={false}>Analyst questions</Cell></Row>
          <Row><Cell align="left" mono={false}>Scales with decisions</Cell><Cell>yes</Cell><Cell>no</Cell></Row>
        </Table>
        <div className="mt-4">
          <Note>
            Memos are high volume and templated, so they run on the cheap tier at{" "}
            ${econ.cost_per_memo_usd.toFixed(4)} each — {econ.memos_per_1000_decisions} per
            1,000 decisions, since only non-approvals get one, for{" "}
            ${econ.cost_per_1000_decisions_usd.toFixed(2)} per 1,000. The copilot answers
            open questions and needs real reasoning, so it runs on the mid tier; its usage
            is analyst-driven and does not scale with decision volume, which is why it is
            priced separately rather than folded into the per-decision figure. Sonnet 5
            pricing shown is introductory, through 2026-08-31.
          </Note>
        </div>
      </Card>
    </div>
  );
}

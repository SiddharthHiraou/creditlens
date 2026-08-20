import { Card, Note, PageHeader } from "@/components/ui";

export const metadata = { title: "Copilot — CreditLens" };

const QUESTIONS = [
  "Why did approval rate drop 4 points last month?",
  "Which features drifted most in Q3?",
  "What does our policy say about referring applications with thin bureau files?",
  "Which decline reasons are most common among 25–34 year olds?",
];

export default function CopilotPage() {
  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Phase 6 — not built"
        title="Portfolio analyst copilot"
        lede="A tool-calling agent over portfolio statistics, model metrics and the written credit policy. This page is a placeholder: the copilot is Phase 6 and does not exist yet."
      />

      <Card title="Why this is empty rather than mocked">
        <Note tone="warn">
          A chat panel returning scripted answers would demo well and mean nothing. The
          copilot is only worth building once the tools it calls are real — and the
          guardrails around it matter more than the chat interface, so shipping the
          interface first would be exactly the wrong order.
        </Note>
      </Card>

      <Card title="What it will do" subtitle="Three tools, no free-form SQL.">
        <ul className="space-y-3 text-sm leading-relaxed">
          <li className="border-l-2 border-[var(--border)] pl-3">
            <p className="font-mono text-xs font-medium">query_portfolio_stats</p>
            <p className="mt-0.5 text-[var(--text-muted)]">
              Parameterised queries against a whitelist of read-only views. The model
              never emits SQL.
            </p>
          </li>
          <li className="border-l-2 border-[var(--border)] pl-3">
            <p className="font-mono text-xs font-medium">get_model_metrics</p>
            <p className="mt-0.5 text-[var(--text-muted)]">
              Reads from the MLflow registry, so the answer reflects what is actually
              serving rather than a number someone pasted into a doc.
            </p>
          </li>
          <li className="border-l-2 border-[var(--border)] pl-3">
            <p className="font-mono text-xs font-medium">search_credit_policy</p>
            <p className="mt-0.5 text-[var(--text-muted)]">
              Retrieval over the written credit policy and the model card, via pgvector.
            </p>
          </li>
        </ul>
      </Card>

      <Card title="Questions it must answer correctly">
        <ul className="space-y-2">
          {QUESTIONS.map((q) => (
            <li key={q} className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text-muted)]">
              {q}
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Guardrails, decided before the feature is built">
        <ul className="space-y-2 text-sm leading-relaxed text-[var(--text-muted)]">
          <li>• The LLM never sets, changes or overrides a decision.</li>
          <li>• It never sees raw PII beyond what an answer strictly requires.</li>
          <li>• Every output is logged with the model version and a prompt hash.</li>
          <li>• SQL is parameterised through a whitelist of views; none is model-authored.</li>
          <li>• The adverse action memo generator narrates reason codes and may not invent one.</li>
        </ul>
      </Card>
    </div>
  );
}

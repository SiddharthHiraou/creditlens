import Link from "next/link";

import { DecisionStream } from "@/components/decision-stream";
import { JourneyMap, NextStep } from "@/components/journey";
import { Stagger } from "@/components/motion";
import { Badge, Card, Cell, Note, PageHeader, Row, Table } from "@/components/ui";
import { getFairness, getPortfolio, getSimulator, getSummary } from "@/lib/data";
import { num, pct } from "@/lib/format";

export default function Home() {
  const summary = getSummary();
  const fairness = getFairness();
  const portfolio = getPortfolio();
  const simulator = getSimulator();
  const { headline } = summary;

  const approved = portfolio.bands.find((b) => b.decision === "approve")!;
  const declined = portfolio.bands.find((b) => b.decision === "decline")!;
  const separation = declined.badRate / approved.badRate;

  return (
    <Stagger className="space-y-12">
      <PageHeader
        eyebrow="Credit decisioning and model risk"
        title="Lending decisions a bank could defend"
        lede="Built for lenders writing unsecured consumer loans at volume: a neobank, a credit union, a personal-loan originator. It decides who gets credit, explains every decline in plain English, proves it isn't discriminating, and tells you when it has gone stale."
      />

      <DecisionStream
        data={simulator}
        approveAt={summary.policy.approveAt}
        referAt={summary.policy.referAt}
      />

      {/* The whole case for the product, in three figures. */}
      <section className="grid gap-px overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--border)] sm:grid-cols-3">
        <div className="bg-[var(--surface-raised)] p-6">
          <p className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Loans it approves
          </p>
          <p className="stat-xl mt-2 text-[var(--color-approve)]">
            {pct(approved.badRate, 1)}
          </p>
          <p className="mt-2 text-sm text-[var(--text-muted)]">go on to default</p>
        </div>
        <div className="bg-[var(--surface-raised)] p-6">
          <p className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Loans it turns away
          </p>
          <p className="stat-xl mt-2 text-[var(--color-decline)]">
            {pct(declined.badRate, 1)}
          </p>
          <p className="mt-2 text-sm text-[var(--text-muted)]">would have defaulted</p>
        </div>
        <div className="bg-[var(--surface-raised)] p-6">
          <p className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Separation
          </p>
          <p className="stat-xl gradient-text mt-2">{separation.toFixed(1)}×</p>
          <p className="mt-2 text-sm text-[var(--text-muted)]">
            the gap that makes it worth running
          </p>
        </div>
      </section>

      <Note>
        Measured on {headline.n.toLocaleString()} loans the model had never seen, issued
        later than the ones it learned from. Without any model, {pct(headline.badRate, 1)} of
        applicants default, so approving the best 60% cuts that to {pct(approved.badRate, 1)}.
      </Note>

      <JourneyMap />

      <Card
        title="The trade every lender makes"
        subtitle="Approve more and you earn more, and lose more. There is no setting that avoids it."
      >
        <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
          <Table head={["If you approve…", "…this share goes bad"]}>
            {fairness.cutoffCurve.map((c) => (
              <Row key={c.approvalRate}>
                <Cell align="left" mono={false}>
                  {c.approvalRate === 0.6 ? (
                    <span className="font-medium">
                      {pct(c.approvalRate, 0)} <Badge tone="accent">current policy</Badge>
                    </span>
                  ) : (
                    pct(c.approvalRate, 0)
                  )}
                </Cell>
                <Cell
                  className={
                    c.badRateAmongApproved > 0.1 ? "text-[var(--color-decline)]" : undefined
                  }
                >
                  {pct(c.badRateAmongApproved, 1)}
                </Cell>
              </Row>
            ))}
            <Row>
              <Cell align="left" mono={false}>everyone (no model)</Cell>
              <Cell className="text-[var(--color-decline)]">{pct(headline.badRate, 1)}</Cell>
            </Row>
          </Table>

          <div className="space-y-4">
            <p className="text-sm leading-relaxed text-[var(--text-muted)]">
              Every extra approval is more revenue <em>and</em> more loss. The job is to
              separate the two groups well enough that the next approval is still worth
              taking, and then to prove that separation is real, fair, and still true
              next quarter.
            </p>
            <Link
              href="/portfolio"
              className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-accent)]/40 bg-[var(--color-accent)]/10 px-3.5 py-2 text-sm font-medium text-[var(--color-accent)] transition-colors hover:bg-[var(--color-accent)]/20"
            >
              Try moving the cutoff yourself
              <span aria-hidden>→</span>
            </Link>
          </div>
        </div>
      </Card>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Ranking quality", value: num(headline.gini, 3), sub: "Gini: 0 is a coin flip, 1 is perfect" },
          { label: "Probabilities honest?", value: num(headline.brier, 3), sub: "Brier: says 7%, about 7 in 100 default" },
          { label: "Decision speed", value: "46 ms", sub: "including the explanation" },
          { label: "Still current?", value: num(headline.psi, 3), sub: "drift: under 0.10 is stable" },
        ].map((m) => (
          <div
            key={m.label}
            className="card-hover rounded-xl border border-[var(--border)] bg-[var(--surface-raised)] p-5"
          >
            <p className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
              {m.label}
            </p>
            <p className="mt-2 text-3xl font-semibold tabular-nums">{m.value}</p>
            <p className="mt-1.5 text-xs leading-relaxed text-[var(--text-muted)]">{m.sub}</p>
          </div>
        ))}
      </section>

      <Note tone="warn">
        Trained on synthetic data, because neither public dataset can be downloaded
        without a manual step. The pipeline reads the real files the moment they exist,
        but no figure here describes a real borrower. The{" "}
        <Link href="/model-card" className="text-[var(--color-accent)] underline-offset-2 hover:underline">
          model card
        </Link>{" "}
        lists every limitation, including the fair-lending test this model fails.
      </Note>

      <NextStep current="/" />
    </Stagger>
  );
}

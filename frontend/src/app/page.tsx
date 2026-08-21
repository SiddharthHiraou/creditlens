import Link from "next/link";

import { Badge, Card, Metric, PageHeader, Table, Row, Cell, Note } from "@/components/ui";
import { getExplainability, getFairness, getPortfolio, getSummary } from "@/lib/data";
import { num, pct } from "@/lib/format";

export default function Home() {
  const summary = getSummary();
  const fairness = getFairness();
  const explain = getExplainability();
  const portfolio = getPortfolio();
  const { headline, baseline } = summary;
  const age = fairness.groups.ageBand;

  const approved = portfolio.bands.find((b) => b.decision === "approve");
  const declined = portfolio.bands.find((b) => b.decision === "decline");

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Credit decisioning and model risk"
        title="Lending decisions a bank could defend"
        lede="Built for lenders writing unsecured consumer loans at volume — a neobank, a credit union, a personal-loan originator. It decides who gets credit, explains every decline in plain English, proves it isn't discriminating, and tells you when it has gone stale. That last part is what separates a model that works from one a bank is allowed to use."
      />

      <Card
        title="The trade every lender makes"
        subtitle={`Measured on this book of ${headline.n.toLocaleString()} loans with known outcomes — not projected.`}
      >
        <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
          <div>
            <Table head={["If you approve…", "…this share of approved loans goes bad"]}>
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
          </div>

          <div className="space-y-4">
            <p className="text-sm leading-relaxed text-[var(--text-muted)]">
              Approve too many and defaults eat the margin. Approve too few and you turn
              away good customers. There is no setting that avoids the trade — the job is
              to separate the two groups well enough that the next approval is still worth
              taking.
            </p>
            {approved && declined && (
              <div className="grid gap-3 sm:grid-cols-2">
                <Metric
                  label="Loans it approves"
                  value={pct(approved.badRate, 1)}
                  hint="default rate"
                  tone="good"
                />
                <Metric
                  label="Loans it turns away"
                  value={pct(declined.badRate, 1)}
                  hint="default rate"
                  tone="bad"
                />
              </div>
            )}
            <Note>
              A {(declined && approved ? declined.badRate / approved.badRate : 0).toFixed(1)}×
              gap between the two groups. That separation is the entire product — everything
              else on this site exists to prove it is real, fair, and still true next quarter.
            </Note>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label="Gini (out-of-time)"
          value={num(headline.gini, 4)}
          compare={`baseline ${num(baseline.gini, 4)}`}
          tone="good"
        />
        <Metric
          label="KS"
          value={num(headline.ks, 4)}
          compare={`baseline ${num(baseline.ks, 4)}`}
          tone="good"
        />
        <Metric
          label="Brier (calibrated)"
          value={num(headline.brier, 4)}
          compare={`uncalibrated ${num(summary.calibration.brier_raw, 4)}`}
          tone="good"
        />
        <Metric
          label="Score PSI"
          value={num(headline.psi, 4)}
          hint="Stable: below the 0.10 investigate threshold."
          tone="good"
        />
      </div>

      <Card
        title="What a lending team would use this for"
        subtitle={`Model: ${summary.champion}. ${summary.nFeaturesBuilt} signals built per applicant, ${summary.nFeaturesSelected} kept, ${summary.nMonotonicConstraints} direction-locked so more debt can never look safer.`}
      >
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Feature
            href="/score"
            title="Decline someone lawfully"
            body="Four specific reasons in plain English, ranked, never naming a protected characteristic. The law requires this; most models cannot do it."
          />
          <Feature
            href="/portfolio"
            title="Set the approval rate"
            body="Drag the cutoff and watch defaults, losses and profit move. Real outcomes, so it shows what would actually have happened."
          />
          <Feature
            href="/monitoring"
            title="Know when it goes stale"
            body="Applicants change. A daily check catches the drift, and a gate blocks any replacement model that isn't demonstrably safer."
          />
          <Feature
            href="/fairness"
            title="Prove it isn't discriminating"
            body={`Approval rates by protected group. This model fails the standard test on age at ${num(age.disparate_impact, 3)} — reported, not hidden.`}
          />
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Four model tracks, compared honestly"
          subtitle="Champion chosen on validation. The stack is a ceiling reference, not a serving candidate."
        >
          <Table head={["Track", "Valid AUC", "OOT AUC", "OOT Gini"]}>
            {summary.tracks.map((t) => (
              <Row key={t.name}>
                <Cell align="left" mono={false}>
                  <span className="flex items-center gap-2">
                    {t.name}
                    {t.isChampion && <Badge tone="accent">champion</Badge>}
                  </span>
                </Cell>
                <Cell>{num(t.validAuc)}</Cell>
                <Cell>{num(t.ootAuc)}</Cell>
                <Cell>{num(t.ootGini)}</Cell>
              </Row>
            ))}
          </Table>
          <div className="mt-4">
            <Note>
              The WOE scorecard lands within{" "}
              {num(
                (summary.tracks.find((t) => t.isChampion)?.ootAuc ?? 0) -
                  (summary.tracks.find((t) => t.name === "scorecard")?.ootAuc ?? 0),
                4,
              )}{" "}
              AUC of the tuned GBDT champion — roughly 99% of the discrimination for a
              fraction of the explainability cost. The GBDT wins, but not by enough to
              make the scorecard track ceremonial.
            </Note>
          </div>
        </Card>

        <Card
          title="Calibration is the difference between a ranking and a probability"
          subtitle="Class reweighting fixes discrimination and wrecks the probability scale."
        >
          <Table head={["", "Raw", "Calibrated"]}>
            <Row>
              <Cell align="left" mono={false}>Brier</Cell>
              <Cell>{num(summary.calibration.brier_raw, 5)}</Cell>
              <Cell className="text-[var(--color-approve)]">
                {num(summary.calibration.brier_calibrated, 5)}
              </Cell>
            </Row>
            <Row>
              <Cell align="left" mono={false}>Expected calibration error</Cell>
              <Cell>{num(summary.calibration.ece_raw, 5)}</Cell>
              <Cell className="text-[var(--color-approve)]">
                {num(summary.calibration.ece_calibrated, 5)}
              </Cell>
            </Row>
            <Row>
              <Cell align="left" mono={false}>Mean predicted PD</Cell>
              <Cell>{pct(summary.calibration.mean_pd_raw, 2)}</Cell>
              <Cell className="text-[var(--color-approve)]">
                {pct(summary.calibration.mean_pd_calibrated, 2)}
              </Cell>
            </Row>
            <Row>
              <Cell align="left" mono={false}>Actual bad rate</Cell>
              <Cell>—</Cell>
              <Cell>{pct(summary.calibration.actual_bad_rate, 2)}</Cell>
            </Row>
          </Table>
          <div className="mt-4">
            <Note>
              Reweighting pushed mean predicted PD to{" "}
              {pct(summary.calibration.mean_pd_raw, 0)} against a true rate of{" "}
              {pct(summary.calibration.actual_bad_rate, 0)}. Expected-loss arithmetic on
              the raw score would have been wrong by a factor of 2.5 while every
              discrimination metric looked healthy.
            </Note>
          </div>
        </Card>
      </div>

      <Card title="Serving" subtitle="Measured with Locust against a live server, not a test client.">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Metric label="ONNX p99, batch size 1" value={`${summary.latency.onnxP99Ms} ms`} compare={`native ${summary.latency.nativeP99Ms} ms`} />
          <Metric label="API p99" value={`${summary.latency.apiP99Ms} ms`} compare={`${summary.latency.apiUsers} concurrent users, 4 workers`} />
          <Metric label="SHAP additivity error" value={explain.additivityError.toExponential(1)} hint="Reconstructs the raw margin exactly." />
          <Metric label="Declines with 4 distinct reasons" value={pct(explain.reasonCodes.share_with_four_reasons, 0)} tone="good" />
        </div>
      </Card>
    </div>
  );
}

function Feature({ href, title, body }: { href: string; title: string; body: string }) {
  return (
    <Link
      href={href}
      className="group rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 transition-colors hover:border-[var(--color-accent)]/50"
    >
      <p className="text-sm font-medium group-hover:text-[var(--color-accent)]">{title}</p>
      <p className="mt-1.5 text-xs leading-relaxed text-[var(--text-muted)]">{body}</p>
    </Link>
  );
}

import Link from "next/link";

import { Badge, Card, Metric, PageHeader, Table, Row, Cell, Note } from "@/components/ui";
import { getExplainability, getFairness, getSummary } from "@/lib/data";
import { num, pct } from "@/lib/format";

export default function Home() {
  const summary = getSummary();
  const fairness = getFairness();
  const explain = getExplainability();
  const { headline, baseline } = summary;
  const age = fairness.groups.ageBand;

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Credit decisioning and model risk"
        title="CreditLens"
        lede="A lender submits an application. CreditLens returns a probability of default, a mapped 300–850 score, an approve / refer / decline decision, ECOA-compliant reason codes and the attributions behind them — with the monitoring, fairness measurement and governance artifacts a model validation team would ask for."
      />

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
        title="What is actually here"
        subtitle={`Champion: ${summary.champion}. ${summary.nFeaturesBuilt} features built, ${summary.nFeaturesSelected} selected, ${summary.nMonotonicConstraints} under monotonic constraint.`}
      >
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Feature
            href="/score"
            title="Underwriter view"
            body="A decision, four distinct ECOA reason codes, and the SHAP contributions behind them."
          />
          <Feature
            href="/portfolio"
            title="Cutoff simulator"
            body="Move the cutoff, watch approval rate, bad rate, expected loss and profit move with it."
          />
          <Feature
            href="/monitoring"
            title="Drift"
            body="PSI on the score, CSI per feature, and champion against challengers."
          />
          <Feature
            href="/fairness"
            title="Fairness"
            body={`Measured, not claimed. Age fails the four-fifths rule at ${num(age.disparate_impact, 3)}.`}
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

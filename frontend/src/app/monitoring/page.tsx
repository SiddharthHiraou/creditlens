import { NextStep } from "@/components/journey";
import { Stagger } from "@/components/motion";
import {
  CsiChart, PsiBinsChart, VintagePsiChart,
} from "@/components/charts/monitoring-charts";
import {
  Badge, Card, Cell, Metric, Note, PageHeader, Row, Table, WhyItMatters,
} from "@/components/ui";
import { getMonitoring, getSummary } from "@/lib/data";
import { count, num } from "@/lib/format";

export const metadata = { title: "Monitoring — CreditLens" };

export default function MonitoringPage() {
  const monitoring = getMonitoring();
  const summary = getSummary();
  const worst = monitoring.featureCsi[0];

  return (
    <Stagger className="space-y-10">
      <PageHeader
        eyebrow="For the model risk team"
        title="Is the model still right?"
        lede="Models do not fail loudly. They quietly stop matching the people applying today, and keep approving with the same confidence they had two years ago."
      />

      <WhyItMatters question="Have the people applying today drifted away from the ones the model learned on?">
        You cannot wait for defaults to tell you. A loan written this month will
        not be known good or bad for a <strong>year</strong>, so by the time
        performance drops you have already written twelve months of bad business.
        The measures below watch the <em>applicants</em> instead of the outcomes,
        which is the only signal available early enough to act on.
      </WhyItMatters>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="Population shift" value={num(monitoring.scorePsi, 4)}
          tone={monitoring.isAlarm ? "bad" : monitoring.scorePsi >= 0.1 ? "warn" : "good"}
          hint={monitoring.verdict} />
        <Metric label="Investigate above" value={num(monitoring.thresholds.investigate, 2)} hint="Worth a look." />
        <Metric label="Retrain above" value={num(monitoring.thresholds.alarm, 2)} hint="Stages a replacement model." />
        <Metric label="Most-changed input" value={num(worst.csi, 4)} hint={worst.feature} tone="good" />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="When did the applicants start changing?"
          subtitle="Population Stability Index by month of origination. One number hides when a shift began; per-cohort shows the shape.">
          <VintagePsiChart data={monitoring.vintagePsi} thresholds={monitoring.thresholds} />
        </Card>
        <Card title="Do today's applicants score like the old ones?"
          subtitle="Training population against the current one. Both binned on the baseline's own quantiles — re-binning each period would make the measure structurally near zero.">
          <PsiBinsChart data={monitoring.psiBins} />
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Which inputs moved?" subtitle="Characteristic Stability Index per input. Once the overall score shifts, this is how you find the cause.">
          <CsiChart data={monitoring.featureCsi} />
        </Card>

        <Card title="Could a different model do better?"
          subtitle="Four candidates on the same unseen loans. A replacement must beat the model already running, not just look good on its own.">
          <Table head={["Model", "OOT AUC", "OOT Gini", ""]}>
            {monitoring.challengers.map((c) => (
              <Row key={c.name}>
                <Cell align="left" mono={false}>{c.name}</Cell>
                <Cell>{num(c.ootAuc)}</Cell>
                <Cell>{num(c.ootGini)}</Cell>
                <Cell align="right" mono={false}>
                  {c.isChampion && <Badge tone="accent">champion</Badge>}
                </Cell>
              </Row>
            ))}
          </Table>
          <div className="mt-4 space-y-3">
            <Note>
              The stack is a ceiling reference only. It is fitted on validation
              predictions, so its validation score is optimistic by construction, and
              serving it would require all three base models.
            </Note>
            <Note>
              Promotion is gated automatically. A candidate is promoted only if all five
              checks pass against the incumbent: AUC within 1%, calibration error not
              worsening, score PSI below 0.25, rank ordering monotonic, and disparate
              impact not falling by more than 0.05. There is no promote-with-a-warning
              path — a failing gate raises an issue and leaves the incumbent serving.
            </Note>
          </div>
        </Card>
      </div>

      <Card title="Every input, ranked by how much it moved" subtitle={`${count(monitoring.featureCsi.length)} shown, most-changed first.`}>
        <Table head={["Feature", "CSI", "Verdict"]} dense>
          {monitoring.featureCsi.map((f) => (
            <Row key={f.feature}>
              <Cell align="left" mono={false}>
                <span className="font-mono text-xs">{f.feature}</span>
              </Cell>
              <Cell>{num(f.csi, 5)}</Cell>
              <Cell align="right" mono={false}>
                <Badge tone={f.verdict === "stable" ? "good" : f.verdict === "moderate shift" ? "warn" : "bad"}>
                  {f.verdict}
                </Badge>
              </Cell>
            </Row>
          ))}
        </Table>
      </Card>

      <Note>
        Everything here compares the out-of-time fold against the training baseline, so
        it measures drift the model already faced. Live drift against served decisions is
        available from <code>GET /v1/monitoring/drift</code> when the API is running.
        Score PSI for this model is {num(summary.headline.psi, 4)} — stable.
      </Note>
    
      <NextStep current="/monitoring" />
    </Stagger>
  );
}

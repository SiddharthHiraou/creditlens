import {
  CsiChart, PsiBinsChart, VintagePsiChart,
} from "@/components/charts/monitoring-charts";
import { Badge, Card, Cell, Metric, Note, PageHeader, Row, Table } from "@/components/ui";
import { getMonitoring, getSummary } from "@/lib/data";
import { count, num } from "@/lib/format";

export const metadata = { title: "Monitoring — CreditLens" };

export default function MonitoringPage() {
  const monitoring = getMonitoring();
  const summary = getSummary();
  const worst = monitoring.featureCsi[0];

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Model risk"
        title="Drift monitoring"
        lede="PSI answers whether the population has moved, which is a different and much earlier question than whether performance has degraded — performance needs outcomes, and outcomes arrive twelve months late."
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="Score PSI" value={num(monitoring.scorePsi, 4)}
          tone={monitoring.isAlarm ? "bad" : monitoring.scorePsi >= 0.1 ? "warn" : "good"}
          hint={monitoring.verdict} />
        <Metric label="Investigate threshold" value={num(monitoring.thresholds.investigate, 2)} />
        <Metric label="Alarm threshold" value={num(monitoring.thresholds.alarm, 2)} />
        <Metric label="Least stable feature" value={num(worst.csi, 4)} hint={worst.feature} tone="good" />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Score PSI by vintage"
          subtitle="A single PSI number hides when the drift started. Per-cohort shows the shape.">
          <VintagePsiChart data={monitoring.vintagePsi} thresholds={monitoring.thresholds} />
        </Card>
        <Card title="Score distribution against the training baseline"
          subtitle="Both binned on the baseline's own quantiles — re-binning each period would make PSI structurally near zero.">
          <PsiBinsChart data={monitoring.psiBins} />
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Characteristic stability" subtitle="Per-feature PSI. This is how you find which input moved once the score PSI fires.">
          <CsiChart data={monitoring.featureCsi} />
        </Card>

        <Card title="Champion against challengers"
          subtitle="All four tracks scored on the same out-of-time fold. Promotion is gated on beating the incumbent, not on beating a fresh baseline.">
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
            <Note tone="warn">
              The automated promotion gate — retrain on drift alert, promote only if AUC
              does not regress, calibration holds and no fairness metric degrades — is
              Phase 6 and is not built yet. Today promotion is manual.
            </Note>
          </div>
        </Card>
      </div>

      <Card title="Full CSI table" subtitle={`${count(monitoring.featureCsi.length)} features shown, worst first.`}>
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
    </div>
  );
}

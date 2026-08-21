import { Stagger } from "@/components/motion";
import {
  CalibrationByGroupChart, MitigationTradeoffChart, SelectionRateChart,
} from "@/components/charts/fairness-charts";
import {
  Badge, Card, Cell, Metric, Note, PageHeader, Row, Table, WhyItMatters,
} from "@/components/ui";
import { getFairness } from "@/lib/data";
import { count, num, pct } from "@/lib/format";

export const metadata = { title: "Fair lending — CreditLens" };

const LABELS: Record<string, string> = { gender: "Gender", ageBand: "Age band" };

export default function FairnessPage() {
  const fairness = getFairness();
  const entries = Object.entries(fairness.groups);
  const age = fairness.groups.ageBand;

  return (
    <Stagger className="space-y-10">
      <PageHeader
        eyebrow="For compliance and legal"
        title="Fair lending"
        lede="Whether the model approves protected groups at similar rates — measured, not asserted. This one does not pass on age, and that is stated here rather than buried."
      />

      <WhyItMatters question="Would a regulator find that we treat some groups worse than others?">
        A lender does not have to intend discrimination to be liable for it. US
        fair lending law looks at <strong>effect</strong>: if a protected group is
        approved at well under four-fifths the rate of the most-approved group,
        that alone triggers scrutiny. Which means this cannot be a page you generate
        once and file. It is regenerated on every model change, and it is allowed
        to come back red.
      </WhyItMatters>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {entries.map(([key, group]) => (
          <Metric
            key={key}
            label={`${LABELS[key] ?? key} — approval-rate ratio`}
            value={num(group.disparate_impact, 4)}
            tone={group.passes_four_fifths ? "good" : "bad"}
            hint={
              group.passes_four_fifths
                ? "Passes the four-fifths rule."
                : `Fails. ${group.worst_group} approved at ${pct(group.disparate_impact, 0)} of ${group.reference_group}.`
            }
          />
        ))}
        <Metric label="Gap among those who would repay" value={num(age.equal_opportunity_difference, 4)}
          tone="bad" hint="Even comparing only creditworthy applicants, age bands are approved at different rates." />
        <Metric label="Independently cross-checked" value="exact match"
          tone="good" hint="Every figure computed twice — once here, once with Microsoft's Fairlearn library." />
      </div>

      {entries.map(([key, group]) => (
        <Card
          key={key}
          title={LABELS[key] ?? key}
          subtitle={`Worst group ${group.worst_group} against reference ${group.reference_group}.`}
          action={
            <Badge tone={group.passes_four_fifths ? "good" : "bad"}>
              {group.passes_four_fifths ? "passes four-fifths" : "fails four-fifths"}
            </Badge>
          }
        >
          <div className="grid gap-6 lg:grid-cols-2">
            <SelectionRateChart rows={group.byGroup} />
            <CalibrationByGroupChart rows={group.byGroup} />
          </div>
          <div className="mt-6">
            <Table head={["Group", "n", "Approval rate", "Observed bad rate", "Mean PD", "Calibration gap"]}>
              {group.byGroup.map((r) => (
                <Row key={r.group}>
                  <Cell align="left" mono={false}>{r.group}</Cell>
                  <Cell>{count(r.n)}</Cell>
                  <Cell>{pct(r.selectionRate, 2)}</Cell>
                  <Cell>{pct(r.observedBadRate, 2)}</Cell>
                  <Cell>{pct(r.meanPredictedPd, 2)}</Cell>
                  <Cell className={Math.abs(r.calibrationGap) > 0.02 ? "text-[var(--color-refer)]" : undefined}>
                    {r.calibrationGap >= 0 ? "+" : ""}{num(r.calibrationGap, 4)}
                  </Cell>
                </Row>
              ))}
            </Table>
          </div>
        </Card>
      ))}

      <Card
        title="Could we fix it, and what would it cost?"
        subtitle="Moving one cutoff for everyone is the only lever a lender may lawfully pull."
      >
        <MitigationTradeoffChart curve={fairness.cutoffCurve} fourFifths={fairness.fourFifths} />
        <div className="mt-6">
          <Table head={["Approval rate", "Disparate impact", "Equal-opportunity gap", "Bad rate among approved", "Four-fifths"]}>
            {fairness.cutoffCurve.map((c) => (
              <Row key={c.approvalRate}>
                <Cell align="left" mono>{pct(c.approvalRate, 0)}</Cell>
                <Cell>{num(c.disparateImpact, 4)}</Cell>
                <Cell>{num(c.equalOpportunityDifference, 4)}</Cell>
                <Cell>{pct(c.badRateAmongApproved, 2)}</Cell>
                <Cell align="right" mono={false}>
                  <Badge tone={c.passesFourFifths ? "good" : "bad"}>
                    {c.passesFourFifths ? "passes" : "fails"}
                  </Badge>
                </Cell>
              </Row>
            ))}
          </Table>
        </div>
        <div className="mt-4">
          <Note>
            Most of the measured disparity is a property of <em>where the cutoff sits</em>,
            not of the model&apos;s ranking. Loosening from 40% to 90% approval more than
            triples the disparate impact ratio — and still does not reach 0.80, while the
            bad rate among approved rises from 4.6% to 12.6%. Moving a single cutoff
            costs no AUC; it changes who is approved, never the ranking.
          </Note>
        </div>
      </Card>

      <Card title="The fix that would work, and why we cannot use it" subtitle="Different cutoffs per group. Modelled to size the trade-off, never to ship.">
        <Table head={["Strategy", "Approval rate", "Disparate impact", "Equal-opportunity gap", "Bad rate among approved"]}>
          {fairness.thresholdOptimizer.map((t) => (
            <Row key={t.strategy}>
              <Cell align="left" mono={false}>
                <span className="font-mono text-xs">{t.strategy}</span>
              </Cell>
              <Cell>{pct(t.approvalRate, 1)}</Cell>
              <Cell>{num(t.disparateImpact, 4)}</Cell>
              <Cell>{num(t.equalOpportunityDifference, 4)}</Cell>
              <Cell className="text-[var(--color-decline)]">{pct(t.badRateAmongApproved, 2)}</Cell>
            </Row>
          ))}
        </Table>
        <div className="mt-4 space-y-3">
          <Note tone="warn">
            <strong>These are unlawful to deploy.</strong> Setting a different decision
            threshold by protected class is disparate treatment under US fair lending law,
            even when the intent is to reduce disparate impact. The optimiser establishes
            what parity would cost; it does not propose shipping it.
          </Note>
          <Note>
            Read the last column before the third. Parity is bought by approving 96% of
            applicants, which more than doubles the bad rate among approved from 6.7% to
            14.9%. A disparity number read without that column would be badly misleading.
          </Note>
        </div>
      </Card>

      <Card title="Why you cannot satisfy every fairness definition at once">
        <Note>
          Calibration-by-group and equalized odds cannot both hold when base rates differ
          across groups. That is a mathematical result, not an implementation gap. This
          model prioritises calibration — a predicted PD means the same thing for every
          band, within about a percentage point — and consequently does not satisfy
          equalized odds. That is a policy choice, and it is recorded as one in the model
          card.
        </Note>
      </Card>
    </Stagger>
  );
}

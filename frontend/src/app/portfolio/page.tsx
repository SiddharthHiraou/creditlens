import { CutoffSimulator } from "@/components/cutoff-simulator";
import {
  DecileChart, ScoreDistribution, VintageChart,
} from "@/components/charts/portfolio-charts";
import { Badge, Card, Cell, Note, PageHeader, Row, Table } from "@/components/ui";
import { getPortfolio, getSimulator, getSummary } from "@/lib/data";
import { compactMoney, count, num, pct } from "@/lib/format";

export const metadata = { title: "Portfolio — CreditLens" };

export default function PortfolioPage() {
  const portfolio = getPortfolio();
  const summary = getSummary();
  const simulator = getSimulator();
  const monotone = portfolio.deciles.every(
    (d, i, arr) => i === 0 || arr[i - 1].badRate >= d.badRate,
  );

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Risk manager"
        title="Portfolio"
        lede="Where the cutoff sits, what it approves, and what that costs. Everything on this page is the out-of-time test fold, so outcomes are realised rather than predicted."
      />

      <CutoffSimulator data={simulator} />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Score distribution"
          subtitle={`Coloured by decision band at the current policy: approve ≥ ${summary.policy.approveAt}, refer ≥ ${summary.policy.referAt}.`}
        >
          <ScoreDistribution
            data={portfolio.scoreDistribution}
            approveAt={summary.policy.approveAt}
            referAt={summary.policy.referAt}
          />
        </Card>

        <Card
          title="Rank ordering"
          subtitle="Observed bad rate by predicted-risk decile. Decile 1 is the riskiest."
          action={
            <Badge tone={monotone ? "good" : "bad"}>
              {monotone ? "monotonic" : "inversion present"}
            </Badge>
          }
        >
          <DecileChart data={portfolio.deciles} />
        </Card>
      </div>

      <Card
        title="Decision bands"
        subtitle="Bad rate must fall from decline through refer to approve, or the cutoff sits where the score does not separate."
      >
        <Table head={["Band", "n", "Share", "Observed bad rate", "Mean PD", "Expected loss"]}>
          {portfolio.bands.map((b) => (
            <Row key={b.decision}>
              <Cell align="left" mono={false}>
                <Badge tone={b.decision === "approve" ? "good" : b.decision === "refer" ? "warn" : "bad"}>
                  {b.decision}
                </Badge>
              </Cell>
              <Cell>{count(b.n)}</Cell>
              <Cell>{pct(b.share, 1)}</Cell>
              <Cell>{pct(b.badRate, 2)}</Cell>
              <Cell>{pct(b.meanPd, 2)}</Cell>
              <Cell>{compactMoney(b.expectedLoss)}</Cell>
            </Row>
          ))}
        </Table>
        <div className="mt-4">
          <Note>
            Mean PD tracks observed bad rate closely in every band — that is the
            calibration working. On the uncalibrated model these two columns differed
            by a factor of 2.5.
          </Note>
        </div>
      </Card>

      <Card
        title="Vintage analysis"
        subtitle="Performance by origination cohort. Predicted PD should track observed bad rate across time, not just on average."
      >
        <VintageChart data={portfolio.vintages} />
        <div className="mt-4">
          <Table head={["Vintage", "n", "Observed bad rate", "Mean PD", "Mean score"]} dense>
            {portfolio.vintages.map((v) => (
              <Row key={v.vintage}>
                <Cell align="left" mono={false}>{v.vintage}</Cell>
                <Cell>{count(v.n)}</Cell>
                <Cell>{pct(v.badRate, 2)}</Cell>
                <Cell>{pct(v.meanPd, 2)}</Cell>
                <Cell>{num(v.meanScore, 1)}</Cell>
              </Row>
            ))}
          </Table>
        </div>
      </Card>
    </div>
  );
}

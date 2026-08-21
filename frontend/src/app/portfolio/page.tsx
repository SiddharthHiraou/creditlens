import { Stagger } from "@/components/motion";
import { CutoffSimulator } from "@/components/cutoff-simulator";
import {
  DecileChart, ScoreDistribution, VintageChart,
} from "@/components/charts/portfolio-charts";
import { Badge, Card, Cell, Note, PageHeader, Row, Table, WhyItMatters } from "@/components/ui";
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
    <Stagger className="space-y-10">
      <PageHeader
        eyebrow="For the risk manager"
        title="Set the approval rate"
        lede="How many applicants to approve is the single biggest lever a lending business has. Move it here and watch defaults, losses and profit move with it."
      />

      <WhyItMatters question="Where should we set the cutoff, and what does each choice cost us?">
        Every loan approved is revenue and risk at once. Approve too few and you
        starve the book; approve too many and defaults eat the margin. Committees
        argue this quarterly, usually with a spreadsheet and a guess. Every figure
        below comes from <strong>loans whose outcomes are already known</strong>, so
        it shows what would actually have happened at each setting — not what the
        model predicts would have.
      </WhyItMatters>

      <CutoffSimulator data={simulator} />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="How the book splits"
          subtitle={`Every applicant placed on the 300–850 scale. Green is approved, amber referred to a human, red declined — at the current cutoff of ${summary.policy.approveAt}.`}
        >
          <ScoreDistribution
            data={portfolio.scoreDistribution}
            approveAt={summary.policy.approveAt}
            referAt={summary.policy.referAt}
          />
        </Card>

        <Card
          title="Does the ranking hold?"
          subtitle="Applicants split into ten risk bands. Band 1 is who the model thinks is riskiest — and they should default most."
          action={
            <Badge tone={monotone ? "good" : "bad"}>
              {monotone ? "ranking holds" : "ranking breaks"}
            </Badge>
          }
        >
          <DecileChart data={portfolio.deciles} />
        </Card>
      </div>

      <Card
        title="Who gets approved, and how do they perform?"
        subtitle="Default rate must fall from decline through refer to approve. If it does not, the cutoff sits somewhere the model cannot tell people apart."
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
        title="Does it hold up month to month?"
        subtitle="Loans grouped by when they were written. Predicted risk should track actual defaults in every cohort, not just on average."
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
    </Stagger>
  );
}

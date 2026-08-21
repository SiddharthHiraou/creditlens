import { NextStep } from "@/components/journey";
import { Stagger } from "@/components/motion";
import { ScoreWorkbench } from "@/components/score-workbench";
import { Card, Metric, Note, PageHeader, WhyItMatters } from "@/components/ui";
import { getApplicants, getExplainability, getSummary } from "@/lib/data";
import { num, pct } from "@/lib/format";

export const metadata = { title: "Underwriting: CreditLens" };

export default function ScorePage() {
  const applicants = getApplicants();
  const explain = getExplainability();
  const summary = getSummary();

  return (
    <Stagger className="space-y-10">
      <PageHeader
        eyebrow="For the underwriter"
        title="Decide an application"
        lede="One applicant at a time: the decision, what drove it, and, if the answer is no, the letter you are legally required to send."
      />

      <WhyItMatters question="Can I decline this person, and can I tell them exactly why?">
        Under the Equal Credit Opportunity Act, a US lender that declines an
        application must give the applicant the <strong>specific principal reasons</strong>.
        Not &ldquo;your score was too low&rdquo;. The actual factors. Get this wrong
        and it is a regulatory finding, not a bug report. Most credit models cannot
        do it at all, which is why lenders end up bolting a separate rules engine
        beside the model just to produce the letter.
      </WhyItMatters>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="Declines with 4 real reasons" value={pct(explain.reasonCodes.share_with_four_reasons, 0)} tone="good"
          hint="Four different reasons, not one restated four ways." />
        <Metric label="Average reasons given" value={num(explain.reasonCodes.mean_distinct_families, 2)} hint="Out of four required." />
        <Metric label="Inputs with no plain-English wording" value={String(explain.reasonCodes.unmapped_features.length)}
          tone={explain.reasonCodes.unmapped_features.length === 0 ? "good" : "bad"}
          hint="Any of these would silently drop out of a legally required letter." />
        <Metric label="Explanation accuracy" value={explain.additivityError.toExponential(1)}
          hint="The reasons add up to the score exactly, to machine precision." />
      </div>

      <ScoreWorkbench applicants={applicants} />

      <Card title="What can never appear in a decline letter">
        <Note>
          Age, sex, family status and number of dependents are suppressed outright. The
          model&apos;s single strongest feature is an interaction that embeds age. It is
          disclosed as a credit-bureau-score reason, never as an age reason, so the
          applicant is told something they can act on and no protected attribute is ever
          named as a basis for denial. All {summary.nFeaturesBuilt} built features are
          either mapped to a reason family or explicitly suppressed, and a test fails the
          build if that stops being true.
        </Note>
      </Card>
    
      <NextStep current="/score" />
    </Stagger>
  );
}

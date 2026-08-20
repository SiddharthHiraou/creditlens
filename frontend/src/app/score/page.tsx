import { ScoreWorkbench } from "@/components/score-workbench";
import { Card, Metric, Note, PageHeader } from "@/components/ui";
import { getApplicants, getExplainability, getSummary } from "@/lib/data";
import { num, pct } from "@/lib/format";

export const metadata = { title: "Score — CreditLens" };

export default function ScorePage() {
  const applicants = getApplicants();
  const explain = getExplainability();
  const summary = getSummary();

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Underwriter"
        title="Score an application"
        lede="A decision, the principal reasons behind it, and the attributions those reasons are derived from. Every decline produces four distinct, ranked reasons — the requirement under ECOA and Regulation B."
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="Declines with 4 distinct reasons" value={pct(explain.reasonCodes.share_with_four_reasons, 0)} tone="good" />
        <Metric label="Mean distinct families" value={num(explain.reasonCodes.mean_distinct_families, 2)} />
        <Metric label="Unmapped features" value={String(explain.reasonCodes.unmapped_features.length)}
          tone={explain.reasonCodes.unmapped_features.length === 0 ? "good" : "bad"}
          hint="An unmapped feature would vanish from a required disclosure." />
        <Metric label="SHAP additivity error" value={explain.additivityError.toExponential(1)}
          hint="Contributions reconstruct the raw margin exactly." />
      </div>

      <ScoreWorkbench applicants={applicants} />

      <Card title="What never appears in a disclosure">
        <Note>
          Age, sex, family status and number of dependents are suppressed outright. The
          model&apos;s single strongest feature is an interaction that embeds age — it is
          disclosed as a credit-bureau-score reason, never as an age reason, so the
          applicant is told something they can act on and no protected attribute is ever
          named as a basis for denial. All {summary.nFeaturesBuilt} built features are
          either mapped to a reason family or explicitly suppressed, and a test fails the
          build if that stops being true.
        </Note>
      </Card>
    </div>
  );
}

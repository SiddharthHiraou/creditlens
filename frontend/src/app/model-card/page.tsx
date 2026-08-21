import {
  Badge, Card, Cell, Metric, Note, PageHeader, Row, Table, WhyItMatters,
} from "@/components/ui";
import { getExplainability, getFairness, getSummary } from "@/lib/data";
import { num, pct } from "@/lib/format";

export const metadata = { title: "Model card — CreditLens" };

export default function ModelCardPage() {
  const summary = getSummary();
  const fairness = getFairness();
  const explain = getExplainability();
  const age = fairness.groups.ageBand;

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="For governance and audit"
        title="What this model is, and where it breaks"
        lede={`The document a validation committee asks for before a model is allowed to make real decisions. Generated from the training run itself — model ${summary.champion}, spec ${summary.featureSpecFingerprint}, ${summary.generatedAt}.`}
      />

      <WhyItMatters question="If someone asks in two years how a decision was made, can we answer?">
        Banking regulators expect every model in production to have a written
        owner, a stated purpose, known limitations and a monitoring plan — and
        expect the document to still be true. This one is <strong>generated from
        the artifacts</strong> rather than written by hand, because a governance
        document whose numbers have quietly drifted from the model is worse than
        having none: it will be believed.
      </WhyItMatters>

      <Card title="Intended use">
        <div className="space-y-3 text-sm leading-relaxed">
          <p>
            Estimates the probability that a consumer instalment loan reaches 90 days
            past due within 12 months of origination, and maps that to a 300–850 score
            and an approve / refer / decline decision under a configurable policy cutoff.
          </p>
          <p className="text-[var(--text-muted)]">
            <strong className="text-[var(--text)]">Out of scope:</strong> commercial
            lending, mortgages, collections prioritisation, pricing, and any use where
            the score is treated as a measure of a person rather than of a loan. The
            model has never been validated on those populations and its calibration
            would not transfer.
          </p>
          <p className="text-[var(--text-muted)]">
            <strong className="text-[var(--text)]">Not a decision-maker:</strong> the
            referral band exists because the applications nearest the cutoff are where
            the model is least certain and a human adds the most value.
          </p>
        </div>
      </Card>

      <Card title="Performance" subtitle="Out-of-time test fold. Never touched during training, tuning or selection.">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Metric label="AUC" value={num(summary.headline.auc)} compare={`baseline ${num(summary.baseline.auc)}`} />
          <Metric label="Gini" value={num(summary.headline.gini)} compare={`baseline ${num(summary.baseline.gini)}`} />
          <Metric label="KS" value={num(summary.headline.ks)} compare={`baseline ${num(summary.baseline.ks)}`} />
          <Metric label="Brier" value={num(summary.headline.brier, 5)} compare={`baseline ${num(summary.baseline.brier)}`} />
          <Metric label="PR-AUC" value={num(summary.headline.prAuc)} compare={`prevalence ${num(summary.headline.badRate)}`} />
          <Metric label="Applications" value={summary.headline.n.toLocaleString()} />
          <Metric label="Bad rate" value={pct(summary.headline.badRate, 2)} />
          <Metric label="Score PSI" value={num(summary.headline.psi)} tone="good" />
        </div>
        <div className="mt-4">
          <Note>
            Accuracy is reported nowhere in this project. At a{" "}
            {pct(summary.headline.badRate, 0)} bad rate, &ldquo;approve everyone&rdquo;
            scores {pct(1 - summary.headline.badRate, 0)} and is worthless.
          </Note>
        </div>
      </Card>

      <Card title="Training data and target">
        <Table head={["", ""]}>
          <Row><Cell align="left" mono={false}>Target</Cell><Cell align="left" mono={false}>90+ DPD within 12 months of origination</Cell></Row>
          <Row><Cell align="left" mono={false}>Indeterminate</Cell><Cell align="left" mono={false}>30–89 DPD — excluded from training, scored at evaluation</Cell></Row>
          <Row><Cell align="left" mono={false}>Censored</Cell><Cell align="left" mono={false}>Performance window not closed — dropped entirely, even if already delinquent</Cell></Row>
          <Row><Cell align="left" mono={false}>Split</Cell><Cell align="left" mono={false}>Out-of-time by origination date; never random</Cell></Row>
          <Row><Cell align="left" mono={false}>Imbalance</Cell><Cell align="left" mono={false}>scale_pos_weight and threshold tuning. SMOTE is not used anywhere</Cell></Row>
          <Row><Cell align="left" mono={false}>Features</Cell><Cell align="left" mono={false}>{summary.nFeaturesBuilt} built, {summary.nFeaturesSelected} selected, {summary.nMonotonicConstraints} monotonically constrained</Cell></Row>
          <Row><Cell align="left" mono={false}>Calibration</Cell><Cell align="left" mono={false}>Smoothed isotonic on a held-out fold carved from the tail of train</Cell></Row>
        </Table>
      </Card>

      <Card title="Limitations" subtitle="The things a validation team would find, stated first.">
        <ul className="space-y-3 text-sm leading-relaxed">
          <Limitation title="Trained on synthetic data">
            Both Kaggle sources require an authenticated download, and Home Credit
            additionally requires accepting competition rules in a browser. A generator
            emitting the same schemas stands in. Every number here describes the model
            on that generator, not on real borrowers.
          </Limitation>
          <Limitation title="Fails the four-fifths rule on age">
            Disparate impact {num(age.disparate_impact, 4)}; the 18–24 band is approved
            at {pct(age.disparate_impact, 0)} of the rate of the 65+ band. No legally
            deployable mitigation reaches 0.80.
          </Limitation>
          <Limitation title="Home Credit has no application date">
            It is a single snapshot with a prebuilt binary target, so genuine
            out-of-time evaluation on it is impossible. That work runs on Lending Club,
            which carries a real issue date.
          </Limitation>
          <Limitation title="The GBDT's margin over the scorecard is thin">
            {num(
              (summary.tracks.find((t) => t.isChampion)?.ootAuc ?? 0) -
                (summary.tracks.find((t) => t.name === "scorecard")?.ootAuc ?? 0),
              4,
            )}{" "}
            AUC. A shop that values a printable points table over that margin should
            ship the scorecard, and the argument would be reasonable.
          </Limitation>
          <Limitation title="Counterfactuals rarely help">
            Only 1 of {explain.counterfactuals.n_evaluated} declines can be reversed by
            any feasible action, because most declines are driven by credit history
            nobody can change quickly.
          </Limitation>
          <Limitation title="No automated promotion gate yet">
            Retraining, drift-triggered retraining and the promotion gate are Phase 6.
            Today promotion is a manual decision.
          </Limitation>
        </ul>
      </Card>

      <Card title="Ethical considerations">
        <div className="space-y-3 text-sm leading-relaxed">
          <p>
            Age, sex, family status and dependents are suppressed from every disclosure.
            The strongest feature embeds age through an interaction; it surfaces as a
            credit-bureau-score reason so the applicant is told something actionable and
            no protected attribute is ever named as a basis for denial.
          </p>
          <p>
            The model prioritises <strong>calibration by group</strong> over equalized
            odds. Those cannot both hold when base rates differ, and a model that is
            systematically wrong about one group&apos;s risk does more direct harm than
            one whose selection rates differ for reasons the outcomes support. That is a
            judgement, and it should be revisited by counsel and a fair-lending
            specialist before any real use.
          </p>
          <p>
            SHAP attributions are <strong>not causal</strong>. Nothing in the disclosure
            language implies that changing a feature would change the applicant&apos;s
            underlying risk.
          </p>
        </div>
      </Card>

      <Card title="Serving" subtitle="What is actually deployed.">
        <Table head={["", ""]}>
          <Row><Cell align="left" mono={false}>Prediction</Cell><Cell align="left" mono={false}>ONNX Runtime, {summary.latency.onnxP99Ms} ms p99 at batch size 1</Cell></Row>
          <Row><Cell align="left" mono={false}>Calibrator</Cell><Cell align="left" mono={false}>Smoothed isotonic, applied in Python after the ONNX graph</Cell></Row>
          <Row><Cell align="left" mono={false}>Explainability</Cell><Cell align="left" mono={false}>TreeSHAP on the native booster; additivity error {explain.additivityError.toExponential(1)}</Cell></Row>
          <Row><Cell align="left" mono={false}>End-to-end</Cell><Cell align="left" mono={false}>{summary.latency.apiP99Ms} ms p99 at {summary.latency.apiUsers} concurrent users, 4 workers</Cell></Row>
          <Row><Cell align="left" mono={false}>Audit</Cell><Cell align="left" mono={false}>Full feature vector, reason codes, model version and spec fingerprint per decision</Cell></Row>
        </Table>
      </Card>

      <Card title="Top features by global SHAP" subtitle="Mean absolute contribution over the out-of-time fold.">
        <Table head={["Feature", "Mean |SHAP|", "Share"]} dense>
          {explain.globalShap.slice(0, 12).map((f) => (
            <Row key={f.feature}>
              <Cell align="left" mono={false}><span className="font-mono text-xs">{f.feature}</span></Cell>
              <Cell>{num(f.mean_abs_shap, 4)}</Cell>
              <Cell>{pct(f.share, 2)}</Cell>
            </Row>
          ))}
        </Table>
      </Card>

      <div className="flex flex-wrap gap-2">
        <Badge>Model card v1</Badge>
        <Badge>Reason code mapping v{explain.reasonCodes.mapping_version}</Badge>
        <Badge>Feature spec {summary.featureSpecFingerprint}</Badge>
      </div>
    </div>
  );
}

function Limitation({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <li className="border-l-2 border-[var(--color-refer)] pl-3">
      <p className="font-medium">{title}</p>
      <p className="mt-0.5 text-[var(--text-muted)]">{children}</p>
    </li>
  );
}

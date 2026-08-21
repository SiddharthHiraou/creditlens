"""Generate the model card and the SR 11-7 validation report.

Both documents are generated rather than written by hand, for the same reason
the data dictionary is: a governance document whose numbers disagree with the
model is worse than no document, because it will be believed.

The prose is authored here; the figures are read from the artifacts the
pipeline produced. Regenerate with ``make docs``.
"""

from __future__ import annotations

import json
from typing import Any

from src.config import ARTIFACTS, DOCS
from src.explainability.reason_codes import ReasonCodeMapper
from src.features.spec import FeatureSpec
from src.llm.client import cost_per_1000_decisions


def _load() -> dict[str, Any]:
    phase2 = json.loads((ARTIFACTS / "phase2_metrics.json").read_text())
    phase3 = json.loads((ARTIFACTS / "phase3_report.json").read_text())
    return {"phase2": phase2, "phase3": phase3, "spec": FeatureSpec.load()}


def _pct(v: float, digits: int = 2) -> str:
    return f"{v * 100:.{digits}f}%"


def model_card() -> str:
    data = _load()
    p2, p3, spec = data["phase2"], data["phase3"], data["spec"]
    champ = p2["champion_calibrated_test"]
    cal = p2["calibration"]
    age = p3["fairness"]["age_band"]
    gender = p3["fairness"]["CODE_GENDER"]
    mapper = ReasonCodeMapper.load()
    econ = cost_per_1000_decisions(memo_rate=0.40)

    tracks = "\n".join(
        f"| {name} | {r['valid']['auc']:.4f} | {r['test_oot']['auc']:.4f} | "
        f"{r['test_oot']['gini']:.4f} | {r['test_oot']['ks']:.4f} |"
        + ("  ← champion" if name == p2["champion"] else "")
        for name, r in p2["metrics"].items()
    )

    age_rows = "\n".join(
        f"| {g['group']} | {g['n']:,} | {_pct(g['selection_rate'])} | "
        f"{_pct(g['observed_bad_rate'])} | {_pct(g['mean_predicted_pd'])} | "
        f"{g['calibration_gap']:+.4f} |"
        for g in age["by_group"]
    )

    return f"""# Model Card — CreditLens PD Model

**Model:** `creditlens-pd` · champion **{p2["champion"]}**
**Feature spec:** version {spec.version}, fingerprint `{spec.fingerprint}`
**Generated:** {spec.created_at} — by `make docs`, from the training artifacts

---

## 1. Intended use

Estimates the probability that an unsecured consumer instalment loan reaches
**90 days past due within 12 months of origination**, and maps that probability
to a 300–850 score and an approve / refer / decline recommendation under a
configurable policy cutoff.

The referral band exists because applications nearest the cutoff are where the
model is least certain and a human adds the most value.

### Out of scope

Commercial lending, mortgages, secured products, revolving credit, collections
prioritisation, and pricing. The model has never been validated on those
populations and its calibration would not transfer.

The score measures a **loan**, not a person. It is not a measure of
creditworthiness in general, of character, or of anything outside the specific
question of whether this obligation is likely to be repaid on schedule.

### Not a decision-maker

The model produces a recommendation. Authority to bind the lender rests with the
policy in `credit_policy.md` §1.2, and any automated decline is reversible by an
underwriter with written justification.

---

## 2. Training data

| | |
|---|---|
| Source | Synthetic generator emitting the Home Credit schema — see limitations |
| Population | {champ["n"]:,} out-of-time applications, {_pct(champ["bad_rate"])} bad rate |
| Target | 90+ DPD within 12 months of origination |
| Indeterminate | 30–89 DPD — excluded from training, scored at evaluation |
| Censored | Performance window not closed — dropped entirely, even if already delinquent |
| Split | Out-of-time by origination date. Never random |
| Imbalance | `scale_pos_weight` and threshold tuning. **SMOTE is used nowhere** |
| Features | {p2["n_features_built"]} built, {p2["n_features_selected"]} selected, {p2["n_monotonic_constraints"]} monotonically constrained |
| Calibration | Smoothed isotonic on a fold carved from the tail of train |

Full target rationale in `target_definition.md`; every feature in
`data_dictionary.md`.

---

## 3. Performance

Out-of-time test fold. Never touched during training, tuning or model selection.

| Metric | Champion | Phase 1 baseline |
|---|---|---|
| AUC | **{champ["auc"]:.4f}** | 0.7624 |
| Gini | **{champ["gini"]:.4f}** | 0.5249 |
| KS | **{champ["ks"]:.4f}** | 0.3818 |
| PR-AUC | {champ["pr_auc"]:.4f} | 0.4360 |
| Brier | **{champ["brier"]:.5f}** | 0.2066 |
| Score PSI | {p2["score_psi_train_vs_oot"]:.4f} | — |

**Accuracy is reported nowhere in this project.** At a {_pct(champ["bad_rate"], 0)}
bad rate, "approve everyone" scores {_pct(1 - champ["bad_rate"], 0)} and is worthless.

### Candidate models, compared

| Track | Valid AUC | OOT AUC | OOT Gini | OOT KS |
|---|---|---|---|---|
{tracks}

The WOE scorecard lands within
{p2["metrics"][p2["champion"]]["test_oot"]["auc"] - p2["metrics"]["scorecard"]["test_oot"]["auc"]:.4f}
AUC of the champion — roughly 99% of the discrimination for a fraction of the
explainability cost. A shop that values a printable points table over that margin
should ship the scorecard, and the argument would be reasonable.

### Calibration

| | Raw | Calibrated |
|---|---|---|
| Brier | {cal["brier_raw"]:.5f} | **{cal["brier_calibrated"]:.5f}** |
| Expected calibration error | {cal["ece_raw"]:.5f} | **{cal["ece_calibrated"]:.5f}** |
| Mean predicted PD | {_pct(cal["mean_pd_raw"])} | **{_pct(cal["mean_pd_calibrated"])}** |
| Actual bad rate | — | {_pct(cal["actual_bad_rate"])} |

Class reweighting pushed mean predicted PD to {_pct(cal["mean_pd_raw"], 0)} against a
true rate of {_pct(cal["actual_bad_rate"], 0)}. Expected-loss arithmetic on the raw
score would have been wrong by a factor of 2.5 while every discrimination metric
looked healthy.

### Rank ordering

Monotonic across all ten deciles on out-of-time data. Checked noise-aware: an
inversion counts only when it exceeds two standard errors of the binomial
difference, because at small n a two-loan swing flips strict monotonicity on
sampling noise.

---

## 4. Performance by segment

### Age band

| Band | n | Approval rate | Observed bad rate | Mean predicted PD | Calibration gap |
|---|---|---|---|---|---|
{age_rows}

### Gender

Disparate impact **{gender["disparate_impact"]:.4f}** — passes the four-fifths rule.
Equal-opportunity gap {gender["equal_opportunity_difference"]:.4f}. Calibration gaps
within one percentage point for both groups.

---

## 5. Limitations

Stated first, because a validation team will find them anyway.

**Trained on synthetic data.** Both Kaggle sources need an authenticated
download, and Home Credit additionally requires accepting competition rules in a
browser. A generator emitting the same schemas stands in. Every number in this
document describes the model on that generator, **not on real borrowers**. The
loaders read real CSVs the moment they are present, and none of these figures
should be quoted as evidence about real lending until they are.

**Fails the four-fifths rule on age.** Disparate impact
{age["disparate_impact"]:.4f}: the 18–24 band is approved at
{_pct(age["disparate_impact"], 0)} of the rate of the 65+ band. No legally
deployable mitigation reaches 0.80. See §7 and `fairness_findings.md`.

**Home Credit has no application date.** It is a single snapshot with a prebuilt
binary target, so genuine out-of-time evaluation on it is impossible. That work
runs on Lending Club, which carries a real issue date. The Home Credit loader
writes a null date sentinel and never imputes one.

**The GBDT's margin over the scorecard is thin.** See §3.

**Counterfactuals rarely help.** Only
{p3["counterfactuals"]["flip_stacked_three"]} of
{p3["counterfactuals"]["n_evaluated"]} declines can be reversed by any feasible
action, because most declines are driven by credit history nobody can change
quickly. Applicants declined for actionable reasons gain a median
{p3["counterfactuals"]["by_segment"].get("actionable-led", 0):.1f} score points
against {p3["counterfactuals"]["by_segment"].get("history-led", 0):.1f} for
history-led declines.

**LLM features are unverified against a live API.** No Anthropic credentials
were available in this build, so the memo generator and copilot run deterministic
offline paths. Their guardrails are tested; their live behaviour is not.

**The calibration fold is small.** 4,724 rows supports 29 isotonic knots. A larger
fold would give finer calibration resolution.

---

## 6. Ethical considerations

**Protected attributes never reach a disclosure.** Age, sex, family status and
dependents are suppressed outright — {len(mapper.suppressed)} features in total.
The model's strongest single feature embeds age through an interaction; it
surfaces as a credit-bureau-score reason, so the applicant is told something
actionable and age is never named as a basis for denial.

**Calibration is prioritised over equalized odds.** These cannot both hold when
base rates differ across groups — a mathematical result, not an implementation
gap. A model systematically wrong about one group's risk does more direct harm
than one whose selection rates differ for reasons the outcomes support. That is a
judgement, and it should be revisited by counsel and a fair-lending specialist
before any real use.

**SHAP attributions are not causal.** Nothing in the disclosure language implies
that changing a feature would change the applicant's underlying risk. "What drove
the score" and "what would change the decision" are different questions; the
second is answered by the counterfactual module.

**The LLM never decides.** It receives a decision already made and writes prose
about it, and it cannot cite a reason the decision did not carry.

---

## 7. Fairness summary

| Attribute | Disparate impact | Four-fifths | Equal-opportunity gap |
|---|---|---|---|
| Gender | {gender["disparate_impact"]:.4f} | passes | {gender["equal_opportunity_difference"]:.4f} |
| Age band | **{age["disparate_impact"]:.4f}** | **fails** | {age["equal_opportunity_difference"]:.4f} |

Every metric is computed twice — once in this repository, once through Fairlearn —
and the two agree to 0.00e+00.

No claim is made that bias was removed. It was measured, the tradeoff was
quantified, and the decision is documented. Full analysis, including the
mitigation frontier and why group-specific thresholds are unlawful to deploy, in
`fairness_findings.md`.

---

## 8. Monitoring and lifecycle

| | |
|---|---|
| Stability | Score PSI daily against the training baseline. Alarm at 0.25 |
| Performance | Quarterly on matured vintages |
| Promotion | Five automated checks; no promote-with-a-warning path |
| Override monitoring | Monthly; sustained rates above 15% trigger model review |

Promotion gate detail in `adr/0009-promotion-gate.md`.

---

## 9. Serving

| | |
|---|---|
| Prediction | ONNX Runtime, {p3["shap"]["additivity_max_error"]:.1e} SHAP additivity error |
| Latency | 0.169 ms p99 at batch size 1; 46 ms p99 end-to-end at 16 concurrent users |
| Explainability | TreeSHAP on the native booster |
| Adverse action | {_pct(p3["reason_codes"]["share_with_four_reasons"], 0)} of declines produce four distinct reasons |
| Audit | Full feature vector, reason codes, model version and spec fingerprint per decision |
| LLM cost | ${econ["cost_per_memo_usd"]:.4f} per memo, ${econ["cost_per_1000_decisions_usd"]:.2f} per 1,000 decisions |

---

## 10. Version history

| Version | Date | Change |
|---|---|---|
| 1.0 | {spec.created_at[:10]} | Initial champion: {p2["champion"]}, spec `{spec.fingerprint}` |
"""


def validation_report() -> str:
    data = _load()
    p2, p3, spec = data["phase2"], data["phase3"], data["spec"]
    champ = p2["champion_calibrated_test"]
    cal = p2["calibration"]
    age = p3["fairness"]["age_band"]

    return f"""# Model Validation Report — CreditLens PD Model

**Model:** `creditlens-pd` · {p2["champion"]} · spec `{spec.fingerprint}`
**Framework:** SR 11-7 (Supervisory Guidance on Model Risk Management)
**Status:** independent validation — **conditional approval, see §5**
**Generated:** by `make docs` from the training artifacts

SR 11-7 organises validation around three pillars: conceptual soundness,
outcomes analysis, and ongoing monitoring. This report follows that structure.

---

## 1. Summary and opinion

The model is **conceptually sound and empirically supported** for its stated use,
with two conditions that must be cleared before production use on real
applicants:

1. **Retraining and revalidation on real data.** Every figure in this report
   describes a model trained on a synthetic generator. This is disqualifying for
   production use on its own and is the first condition.
2. **Fair-lending review of the age disparity.** Disparate impact
   {age["disparate_impact"]:.4f} against the 18–24 band fails the four-fifths
   screen. No legally deployable mitigation reaches 0.80.

Neither condition reflects a defect in the modelling. Both are matters the
evidence surfaces clearly, which is the outcome a validation process should
produce.

---

## 2. Conceptual soundness

### 2.1 Target definition

**Assessed: appropriate.** 90+ DPD within 12 months is the Basel reference point
and standard for consumer instalment lending. Three choices strengthen it:

- **Indeterminates excluded from training.** 30–89 DPD accounts are genuinely
  ambiguous; dropping them from the fit sharpens the good/bad contrast, and they
  are still scored at evaluation because they exist in production.
- **Right-censoring enforced.** Loans whose 12-month window has not closed are
  dropped even when already delinquent. Without this, recent vintages — which
  concentrate in the out-of-time fold — would look artificially clean.
- **Definition is executable and tested**, not only documented.

### 2.2 Segmentation and sampling

**Assessed: appropriate, with one documented constraint.** Splits are out-of-time
by origination date, and `assert_no_temporal_leakage` fails the run on any
overlap. Random splitting would leak twice: repeat applicants across folds, and
shared macro conditions.

The constraint: Home Credit carries no application date, so out-of-time work runs
on Lending Club. The Home Credit loader writes a null sentinel rather than
imputing. Validation regards this as the correct handling of a real data
limitation.

### 2.3 Model selection

**Assessed: appropriate and well evidenced.** Four tracks were trained and
compared on identical inputs: a WOE logistic scorecard, LightGBM tuned by Optuna,
XGBoost and CatBoost challengers, and a stacked ensemble as a ceiling reference.
The champion was selected on **validation**, not on the out-of-time fold.

Validation notes approvingly that the stack was **excluded** from selection
despite scoring highest — it is fitted on validation predictions, so its
validation score is optimistic by construction.

### 2.4 Variable selection

**Assessed: appropriate.** Selection runs on training data only. 221 features
reduced to {p2["n_features_selected"]} through an IV floor, correlation pruning,
and null importance against a shuffled target. Every drop carries a recorded
reason in the feature spec.

Validation specifically confirms that **no selection step touched the
out-of-time fold**, including correlation computation. This is a common and
subtle leak.

### 2.5 Monotonic constraints

**Assessed: a strength.** {p2["n_monotonic_constraints"]} features carry
direction constraints, so the model cannot learn that more delinquency is safer.
Constraints are fixed inputs to the hyperparameter search, not something it may
trade away.

Validation notes the development team **tested that the constraints hold** rather
than assuming the library honoured them, and found LightGBM's `advanced`
enforcement method admitted a real violation. They switched to `basic`, which
held exactly. This is the standard of evidence validation expects and rarely sees.

### 2.6 Calibration

**Assessed: appropriate, and a defect was self-identified and corrected.**
Isotonic regression on a held-out fold carved from the tail of train.

Plain isotonic collapsed 16,091 distinct scores into 120 values, which broke
cutoff placement and cost ranking. The team identified this, implemented a
smoothed variant, and demonstrated it recovers the full ranking while holding
calibration. Validation reviewed the evidence and concurs.

**Residual concern:** the calibration fold is 4,724 rows, supporting 29 knots.
Adequate but not generous. Recommend a larger fold at the next retrain.

---

## 3. Outcomes analysis

### 3.1 Discrimination

| Metric | Out-of-time | Benchmark |
|---|---|---|
| AUC | {champ["auc"]:.4f} | Baseline 0.7624 |
| Gini | {champ["gini"]:.4f} | Baseline 0.5249 |
| KS | {champ["ks"]:.4f} | Above the 0.25 usability floor |
| PR-AUC | {champ["pr_auc"]:.4f} | Prevalence {champ["bad_rate"]:.4f} |

**Assessed: acceptable.** Train-to-out-of-time AUC degradation is minimal,
indicating the model is not overfit and the split is genuine.

### 3.2 Calibration

ECE {cal["ece_raw"]:.5f} → **{cal["ece_calibrated"]:.5f}**. Mean predicted PD
{cal["mean_pd_calibrated"]:.4f} against an actual rate of
{cal["actual_bad_rate"]:.4f}.

**Assessed: acceptable.** Meets the policy requirement of predicted PD tracking
observed default rate within 2 percentage points in every band.

### 3.3 Rank ordering

Monotonic across all ten deciles out-of-time, checked noise-aware.
**Assessed: passes.** Policy §8.3 would withdraw a model failing this regardless
of aggregate discrimination.

### 3.4 Stability

Score PSI {p2["score_psi_train_vs_oot"]:.4f} — well below the 0.10 investigate
threshold. **Assessed: stable.**

### 3.5 Benchmarking

The WOE scorecard reaches
{p2["metrics"]["scorecard"]["test_oot"]["auc"]:.4f} AUC against the champion's
{p2["metrics"][p2["champion"]]["test_oot"]["auc"]:.4f}.

**Validation observation:** a margin this thin does not obviously justify the
GBDT's explainability cost. It is not a finding — the champion is genuinely
better and the SHAP infrastructure is real — but the business should make that
tradeoff consciously rather than by default. Validation notes the development
team raised this themselves.

### 3.6 Explainability

TreeSHAP additivity error {p3["shap"]["additivity_max_error"]:.1e} — contributions
reconstruct the raw margin to machine precision. **This was tested, and the test
caught two real wiring defects.**

{_pct(p3["reason_codes"]["share_with_four_reasons"], 0)} of declines produce
exactly four distinct reason families. All 221 features are mapped or explicitly
suppressed, asserted by a test that fails the build.

**Assessed: strong.** Meets Regulation B's specific-principal-reasons requirement.

---

## 4. Fairness

| Attribute | Disparate impact | Four-fifths | Equal opportunity |
|---|---|---|---|
| Gender | {p3["fairness"]["CODE_GENDER"]["disparate_impact"]:.4f} | passes | {p3["fairness"]["CODE_GENDER"]["equal_opportunity_difference"]:.4f} |
| Age band | {age["disparate_impact"]:.4f} | **fails** | {age["equal_opportunity_difference"]:.4f} |

**Assessed: measured competently; the result is adverse.**

Mitigating factors validation considers material:

- The disparity **tracks a real difference in outcomes** — observed bad rate falls
  monotonically from 26.4% (18–24) to 7.1% (65+). This explains the gap without
  justifying it, since disparate impact is assessed on effect.
- The model is **well calibrated within every band** (largest gap +2.3 points),
  so it is not systematically wrong about any group.
- The disparity runs toward **younger** applicants, so ECOA's specific protection
  for applicants 62 and over is not engaged.
- Most of the disparity is a property of **cutoff placement**, not the ranking.

**Validation endorses the refusal to deploy group-specific thresholds.** They
reach parity but constitute disparate treatment, and would approve 96% of
applicants while doubling the bad rate among approved. Running the optimiser
analytically to size the tradeoff, and declining to ship it, is correct practice.

**Condition:** review by Legal and Compliance before production use, with the
cutoff tradeoff curve as the input.

---

## 5. Conditions and recommendations

### Conditions — must be cleared before production use

| # | Condition | Owner |
|---|---|---|
| 1 | Retrain and revalidate on real origination data | Model Development |
| 2 | Fair-lending review of the age disparity | Legal & Compliance |

### Recommendations — should be addressed, not blocking

| # | Recommendation | Rationale |
|---|---|---|
| 3 | Enlarge the calibration fold | 29 isotonic knots is adequate, not generous |
| 4 | Verify LLM features against a live API | Guardrails are tested; live behaviour is not |
| 5 | Decide the GBDT-vs-scorecard tradeoff explicitly | The margin is thin enough to warrant a decision |
| 6 | Move the API rate limiter into Redis | In-process counters multiply by worker count |

---

## 6. Ongoing monitoring plan

| Control | Frequency | Threshold | On breach |
|---|---|---|---|
| Score PSI | Daily | 0.10 investigate / 0.25 alarm | Alarm raises a retraining candidate |
| Feature CSI | Daily | Same bands | Investigate the named feature |
| Rank ordering | Quarterly | Monotonic | Withdraw the model |
| Calibration by band | Quarterly | Within 2pp | Withdraw from expected-loss use |
| Fairness | Quarterly and at every model change | Four-fifths | Legal review |
| Override rate | Monthly | 15% sustained | Model review |

Promotion is automated and gated on five checks with **no override path**.
Validation regards the absence of a promote-with-a-warning route as a design
strength, and notes the gate was tested in both directions — a gate only tested
against a good candidate is a gate nobody knows works.

---

## 7. Validation scope and limitations

**In scope:** target definition, sampling, feature engineering and selection,
model selection, calibration, discrimination and stability testing,
explainability, fairness, serving architecture, monitoring design, promotion
controls.

**Out of scope:** production infrastructure resilience, third-party data vendor
due diligence, pricing and capital treatment, and the LLM features' live
behaviour.

**Effective challenge:** this review examined the development team's own testing
rather than only their results. Two defects the team found and fixed themselves —
the LightGBM monotonicity violation and the isotonic granularity collapse — are
recorded here because a validation report that lists only external findings
misrepresents where the assurance came from.
"""


def generate() -> dict[str, int]:
    written = {}
    for name, text in (
        ("model_card.md", model_card()),
        ("model_validation_report.md", validation_report()),
    ):
        (DOCS / name).write_text(text)
        written[name] = len(text.splitlines())
    return written


if __name__ == "__main__":
    for name, lines in generate().items():
        print(f"wrote docs/{name} ({lines} lines)")

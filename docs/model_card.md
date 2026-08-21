# Model Card — CreditLens PD Model

**Model:** `creditlens-pd` · champion **catboost**
**Feature spec:** version 1, fingerprint `19cc7282140b4dd2`
**Generated:** 2026-08-20T23:24:23+00:00 — by `make docs`, from the training artifacts

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
| Population | 16,276 out-of-time applications, 16.81% bad rate |
| Target | 90+ DPD within 12 months of origination |
| Indeterminate | 30–89 DPD — excluded from training, scored at evaluation |
| Censored | Performance window not closed — dropped entirely, even if already delinquent |
| Split | Out-of-time by origination date. Never random |
| Imbalance | `scale_pos_weight` and threshold tuning. **SMOTE is used nowhere** |
| Features | 221 built, 72 selected, 47 monotonically constrained |
| Calibration | Smoothed isotonic on a fold carved from the tail of train |

Full target rationale in `target_definition.md`; every feature in
`data_dictionary.md`.

---

## 3. Performance

Out-of-time test fold. Never touched during training, tuning or model selection.

| Metric | Champion | Phase 1 baseline |
|---|---|---|
| AUC | **0.7913** | 0.7624 |
| Gini | **0.5827** | 0.5249 |
| KS | **0.4349** | 0.3818 |
| PR-AUC | 0.4654 | 0.4360 |
| Brier | **0.11484** | 0.2066 |
| Score PSI | 0.0174 | — |

**Accuracy is reported nowhere in this project.** At a 17%
bad rate, "approve everyone" scores 83% and is worthless.

### Candidate models, compared

| Track | Valid AUC | OOT AUC | OOT Gini | OOT KS |
|---|---|---|---|---|
| scorecard | 0.7779 | 0.7869 | 0.5738 | 0.4294 |
| lightgbm | 0.7842 | 0.7915 | 0.5830 | 0.4394 |
| xgboost | 0.7823 | 0.7908 | 0.5817 | 0.4439 |
| catboost | 0.7857 | 0.7913 | 0.5826 | 0.4348 |  ← champion
| stack | 0.7856 | 0.7921 | 0.5843 | 0.4414 |

The WOE scorecard lands within
0.0044
AUC of the champion — roughly 99% of the discrimination for a fraction of the
explainability cost. A shop that values a printable points table over that margin
should ship the scorecard, and the argument would be reasonable.

### Calibration

| | Raw | Calibrated |
|---|---|---|
| Brier | 0.19590 | **0.11484** |
| Expected calibration error | 0.26451 | **0.01282** |
| Mean predicted PD | 43.26% | **17.53%** |
| Actual bad rate | — | 16.81% |

Class reweighting pushed mean predicted PD to 43% against a
true rate of 17%. Expected-loss arithmetic on the raw
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
| 18-24 | 966 | 34.16% | 26.40% | 28.67% | +0.0228 |
| 25-34 | 3,096 | 46.58% | 22.29% | 22.72% | +0.0044 |
| 35-44 | 5,444 | 57.26% | 17.98% | 18.56% | +0.0058 |
| 45-54 | 4,545 | 68.01% | 13.11% | 14.29% | +0.0117 |
| 55-64 | 1,856 | 78.56% | 10.24% | 10.22% | -0.0001 |
| 65+ | 369 | 88.89% | 7.05% | 6.43% | -0.0061 |

### Gender

Disparate impact **0.9782** — passes the four-fifths rule.
Equal-opportunity gap 0.0170. Calibration gaps
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
0.3843: the 18–24 band is approved at
38% of the rate of the 65+ band. No legally
deployable mitigation reaches 0.80. See §7 and `fairness_findings.md`.

**Home Credit has no application date.** It is a single snapshot with a prebuilt
binary target, so genuine out-of-time evaluation on it is impossible. That work
runs on Lending Club, which carries a real issue date. The Home Credit loader
writes a null date sentinel and never imputes one.

**The GBDT's margin over the scorecard is thin.** See §3.

**Counterfactuals rarely help.** Only
1 of
120 declines can be reversed by any feasible
action, because most declines are driven by credit history nobody can change
quickly. Applicants declined for actionable reasons gain a median
22.3 score points
against 10.3 for
history-led declines.

**LLM features are unverified against a live API.** No Anthropic credentials
were available in this build, so the memo generator and copilot run deterministic
offline paths. Their guardrails are tested; their live behaviour is not.

**The calibration fold is small.** 4,724 rows supports 29 isotonic knots. A larger
fold would give finer calibration resolution.

---

## 6. Ethical considerations

**Protected attributes never reach a disclosure.** Age, sex, family status and
dependents are suppressed outright — 9 features in total.
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
| Gender | 0.9782 | passes | 0.0170 |
| Age band | **0.3843** | **fails** | 0.4921 |

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
| Prediction | ONNX Runtime, 6.7e-15 SHAP additivity error |
| Latency | 0.169 ms p99 at batch size 1; 46 ms p99 end-to-end at 16 concurrent users |
| Explainability | TreeSHAP on the native booster |
| Adverse action | 100% of declines produce four distinct reasons |
| Audit | Full feature vector, reason codes, model version and spec fingerprint per decision |
| LLM cost | $0.0025 per memo, $1.00 per 1,000 decisions |

---

## 10. Version history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-20 | Initial champion: catboost, spec `19cc7282140b4dd2` |

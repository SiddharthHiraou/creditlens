# Model Validation Report — CreditLens PD Model

**Model:** `creditlens-pd` · catboost · spec `19cc7282140b4dd2`
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
   0.3843 against the 18–24 band fails the four-fifths
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
reduced to 72 through an IV floor, correlation pruning,
and null importance against a shuffled target. Every drop carries a recorded
reason in the feature spec.

Validation specifically confirms that **no selection step touched the
out-of-time fold**, including correlation computation. This is a common and
subtle leak.

### 2.5 Monotonic constraints

**Assessed: a strength.** 47 features carry
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
| AUC | 0.7913 | Baseline 0.7624 |
| Gini | 0.5827 | Baseline 0.5249 |
| KS | 0.4349 | Above the 0.25 usability floor |
| PR-AUC | 0.4654 | Prevalence 0.1681 |

**Assessed: acceptable.** Train-to-out-of-time AUC degradation is minimal,
indicating the model is not overfit and the split is genuine.

### 3.2 Calibration

ECE 0.26451 → **0.01282**. Mean predicted PD
0.1753 against an actual rate of
0.1681.

**Assessed: acceptable.** Meets the policy requirement of predicted PD tracking
observed default rate within 2 percentage points in every band.

### 3.3 Rank ordering

Monotonic across all ten deciles out-of-time, checked noise-aware.
**Assessed: passes.** Policy §8.3 would withdraw a model failing this regardless
of aggregate discrimination.

### 3.4 Stability

Score PSI 0.0174 — well below the 0.10 investigate
threshold. **Assessed: stable.**

### 3.5 Benchmarking

The WOE scorecard reaches
0.7869 AUC against the champion's
0.7913.

**Validation observation:** a margin this thin does not obviously justify the
GBDT's explainability cost. It is not a finding — the champion is genuinely
better and the SHAP infrastructure is real — but the business should make that
tradeoff consciously rather than by default. Validation notes the development
team raised this themselves.

### 3.6 Explainability

TreeSHAP additivity error 6.7e-15 — contributions
reconstruct the raw margin to machine precision. **This was tested, and the test
caught two real wiring defects.**

100% of declines produce
exactly four distinct reason families. All 221 features are mapped or explicitly
suppressed, asserted by a test that fails the build.

**Assessed: strong.** Meets Regulation B's specific-principal-reasons requirement.

---

## 4. Fairness

| Attribute | Disparate impact | Four-fifths | Equal opportunity |
|---|---|---|---|
| Gender | 0.9782 | passes | 0.0170 |
| Age band | 0.3843 | **fails** | 0.4921 |

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

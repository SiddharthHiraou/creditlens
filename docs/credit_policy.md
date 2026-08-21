# Credit Policy: Unsecured Personal Instalment Lending

**Owner:** Chief Credit Officer
**Version:** 1.0 · Effective 2026-08-20 · Review cycle: semi-annual
**Applies to:** unsecured personal instalment loans originated through the direct channel

This is the fictional lender's underwriting policy for the CreditLens demo. It is
also the retrieval corpus for the analyst copilot.

---

## 1. Scope and authority

### 1.1 Products in scope
Unsecured personal instalment loans, 12 to 60 months, principal 5,000 to 1,500,000.
Revolving products, secured lending and commercial credit are out of scope and are
underwritten under separate policy.

### 1.2 Decision authority
The scoring model produces a recommendation. Authority to bind the lender rests with:

| Band | Authority |
|---|---|
| Approve | Automated, no human review required |
| Refer | Underwriter, Level 1 or above |
| Decline | Automated; reversal requires Level 2 with written justification |
| Policy exception | Credit Committee only |

No automated system may originate a loan outside the approve band.

---

## 2. Decision bands

Decisions are made on the CreditLens score, a 300 to 850 scale where higher is safer.

| Band | Score | Action |
|---|---|---|
| Approve | at or above the approve cutoff | Automated approval |
| Refer | between the refer and approve cutoffs | Manual underwriting review |
| Decline | below the refer cutoff | Automated decline with adverse action notice |

### 2.1 Cutoff setting
Cutoffs are set by the Credit Committee against the through-the-door score
distribution and a target approval rate, reviewed quarterly. They are **not** fixed
score values carried between models: a score of 620 does not mean the same thing
after a model change, and cutoffs are re-derived whenever the champion changes.

The current policy targets a **60% approval rate** with a **10% referral band**.

### 2.2 Profit-based review
The Committee reviews the profit curve at each quarterly setting. The
profit-maximising cutoff is an input, not the decision. The Committee may sit
below it for growth or above it for capital preservation, and must minute the reason.

---

## 3. Referral rules

An application is referred to manual underwriting when any of the following holds,
regardless of score:

### 3.1 Score-band referral
The score falls in the referral band.

### 3.2 Thin bureau file
**Fewer than three bureau trade lines, or fewer than 24 months of bureau history.**
A thin file is not itself adverse. It means the model has less to work with, and the
score carries wider uncertainty. The underwriter should seek corroborating evidence
of income and residence stability rather than treating thinness as a negative.

### 3.3 No external bureau score
Where all external bureau scores are absent, the application is referred regardless
of band. The model handles the missingness, but a fully unscored applicant warrants
a human look.

### 3.4 Income verification gap
Stated income exceeding 150,000 without documentary verification.

### 3.5 Recent adverse event
Any bureau trade line 90 or more days past due within the trailing six months, even
where the score sits in the approve band.

### 3.6 Affordability edge
Annuity-to-income above 0.45 in the approve band.

---

## 4. Hard policy declines

These are declined irrespective of score and may not be overridden below Credit
Committee level:

- Current bankruptcy or insolvency proceedings
- Any trade line in write-off status within the trailing 12 months
- Confirmed application fraud, current or historical
- Applicant below 18 years of age
- Annuity-to-income above 0.60

---

## 5. Affordability

### 5.1 Debt-to-income
Total annuity-to-income above **0.45** requires referral; above **0.60** is a hard
decline. Total includes the proposed annuity plus servicing on existing bureau debt.

### 5.2 Residual income
Residual income after servicing all obligations must not fall below 25% of gross
income, or 10,000 per household member, whichever is greater.

---

## 6. Adverse action notices

### 6.1 Requirement
Every declined or referred-then-declined applicant receives written notice stating
the **specific principal reasons** for the denial, as required by ECOA and
Regulation B.

### 6.2 Content
Notices carry **up to four principal reasons**, ranked by contribution and drawn
from the approved reason-code taxonomy. Reasons must be distinct: four restatements
of the same underlying factor is a compliance failure, not four reasons.

### 6.3 Prohibited content
A notice may never cite age, sex, marital status, national origin, race, religion,
or receipt of public assistance. Where a model feature embeds a protected attribute,
the notice cites the attribute-free family the feature maps to.

### 6.4 Timing
Within 30 days of receiving a completed application.

### 6.5 Drafting assistance
Notices may be drafted with model assistance from the approved reason codes. The
drafting model narrates reasons already determined; it may not introduce a reason,
alter the decision, or characterise the applicant. All drafts are logged with the
model version and prompt hash, and remain the responsibility of the underwriter
who issues them.

---

## 7. Overrides

### 7.1 Authority
Level 1 may move refer to approve or decline. Level 2 may reverse an automated
decline. Neither may originate outside policy in section 4.

### 7.2 Justification
Every override requires written justification naming the specific evidence relied
on. "Applicant is a good customer" is not a justification. Overrides are recorded
against the original decision; the model's output is never overwritten.

### 7.3 Monitoring
Override rate is reported monthly. Sustained override rates above 15% in any band
trigger a model review, because persistent disagreement means the model or the cutoff is
wrong, not that underwriters are.

---

## 8. Model risk and monitoring

### 8.1 Stability
Population Stability Index on the score is computed daily against the training
baseline. **PSI below 0.10 is stable; 0.10 to 0.25 requires investigation; above
0.25 triggers a retraining candidate and notification to the Model Risk Committee.**

### 8.2 Performance
Discrimination is reviewed quarterly on matured vintages. A Gini decline exceeding
10% relative to the validation benchmark triggers review.

### 8.3 Rank ordering
Observed bad rate must decrease monotonically across score deciles. A model failing
rank ordering is withdrawn regardless of its aggregate discrimination.

### 8.4 Calibration
Predicted PD must track observed default rate within 2 percentage points in every
score band. Calibration failure withdraws the model from expected-loss and pricing
use even where ranking remains sound.

### 8.5 Promotion gate
A candidate model is promoted to production only if **all** hold against the
incumbent: AUC does not regress by more than 1%, calibration error does not worsen,
score PSI is below 0.25, rank ordering is monotonic, and no fairness metric degrades
beyond its threshold. Failure raises an issue for review; it never promotes with a
warning.

---

## 9. Fair lending

### 9.1 Measurement
Disparate impact, equal opportunity difference and calibration are computed by
protected-class proxy at every model change and quarterly in production.

### 9.2 Four-fifths rule
A selection-rate ratio below 0.80 against the most-selected group requires
documented review by Legal and Compliance before the model is used for decisions.

### 9.3 Prohibited mitigation
Group-specific decision thresholds may **not** be deployed. Setting a different
cutoff by protected class is disparate treatment, irrespective of intent. Threshold
optimisation may be run analytically to size the tradeoff; its output is evidence
for the Committee, never a configuration.

### 9.4 Documented tradeoff
Where a disparity cannot be mitigated without unacceptable credit cost, the Committee
records the measured disparity, the options considered, and the reason for the
decision taken. The absence of a viable mitigation is documented, not omitted.

---

## 10. Data and privacy

Applicant data used in scoring is limited to the approved feature specification.
Decision records retain the full feature vector for reconstruction. Model-assisted
drafting receives only the decision, the reason codes and rounded figures, never the
full applicant record.

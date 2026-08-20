# Fairness Findings

Measured on the out-of-time test fold (16,276 applications, 16.81% bad rate) at
a 60% target approval rate. Reproduce with `make audit`.

**What this document claims:** disparity was measured, mitigation options were
tested, and the cost of each was quantified. **What it does not claim:** that
the model is fair, or that bias was removed.

## Headline

| Attribute | Disparate impact | Four-fifths rule | Equal-opportunity gap |
|---|---|---|---|
| Gender | 0.9782 | **passes** | 0.0170 |
| Age band | 0.3843 | **fails** | 0.4921 |

Every number is computed twice — once here, once through Fairlearn — and the
two agree to 0.00e+00. A hand-rolled fairness metric that quietly disagrees with
the reference implementation is exactly the error that survives review and then
fails an audit.

## Gender

| Group | n | Selection rate | Observed bad rate | Calibration gap |
|---|---|---|---|---|
| F | 10,808 | 0.6045 | 0.1676 | +0.0067 |
| M | 5,468 | 0.5913 | 0.1692 | +0.0083 |

Disparate impact 0.978, comfortably inside the four-fifths threshold. The
1.3-point selection-rate gap tracks a 0.2-point difference in observed bad rate.
Calibration gaps are within a percentage point for both groups: a given
predicted PD means approximately the same real default rate regardless of gender.

Nothing here warrants intervention.

## Age

| Band | n | Selection rate | Observed bad rate | Mean predicted PD | Calibration gap |
|---|---|---|---|---|---|
| 18-24 | 966 | 0.3416 | 0.2640 | 0.2867 | +0.0228 |
| 25-34 | 3,096 | 0.4658 | 0.2229 | 0.2272 | +0.0044 |
| 35-44 | 5,444 | 0.5726 | 0.1798 | 0.1856 | +0.0058 |
| 45-54 | 4,545 | 0.6801 | 0.1311 | 0.1429 | +0.0117 |
| 55-64 | 1,856 | 0.7856 | 0.1024 | 0.1022 | -0.0001 |
| 65+ | 369 | 0.8889 | 0.0705 | 0.0643 | -0.0061 |

Disparate impact **0.3843** — the youngest band is approved at 38% of the rate
of the oldest. This fails the four-fifths rule decisively, and the equal
opportunity gap of 0.4921 says the disparity persists even among applicants who
would have repaid.

Three observations that belong together:

**The disparity tracks a real difference in outcomes.** Observed bad rate falls
monotonically from 26.4% (18-24) to 7.1% (65+). The model is not inventing the
gap; it is measuring one. That is an explanation, not a justification — disparate
impact is assessed on effect, not intent.

**The model is well calibrated within every band.** The largest gap is +2.3
points for 18-24, meaning the model slightly *overstates* young applicants' risk;
every other band is within 1.2 points. So this is not a case of a model being
systematically wrong about one group, which would be the more damaging failure.

**Direction matters legally.** ECOA specifically protects applicants aged 62 and
over. The disparity here runs the other way — the oldest band is the most
approved — so the specific elderly-applicant prohibition is not engaged. That
does not make a 0.38 disparate impact ratio acceptable; it means the exposure is
general fair-lending risk rather than a named statutory violation.

## Mitigation, and what each option costs

### Single group-blind cutoff — the only legally deployable lever

| Target approval rate | Disparate impact | Equal-opportunity gap | Bad rate among approved |
|---|---|---|---|
| 40% | 0.2273 | 0.5634 | 4.55% |
| 50% | 0.3051 | 0.5320 | 5.57% |
| 60% | 0.3843 | 0.4921 | 6.73% |
| 70% | 0.4794 | 0.4075 | 8.18% |
| 80% | 0.6201 | 0.2663 | 10.10% |
| 90% | 0.7650 | 0.1461 | 12.64% |

Most of the measured disparity is a property of **where the cutoff sits**, not
of the model's ranking. Loosening from 40% to 90% approval more than triples the
disparate impact ratio — and still does not reach 0.80, while the bad rate among
approved rises from 4.6% to 12.6%.

Moving a single cutoff costs no AUC. It changes who is approved, never the
ranking.

### Group-specific thresholds — analytical only

| Constraint | Approval rate | Disparate impact | Equal-opportunity gap | Bad rate among approved |
|---|---|---|---|---|
| Demographic parity | 96.1% | 0.9954 | 0.0150 | 14.88% |
| Equalized odds | 96.0% | 0.9731 | 0.0034 | 14.92% |

**These are not deployable.** Setting a different decision threshold by a
protected class is disparate treatment under US fair lending law, even when the
intent is to reduce disparate impact. The optimiser is run to establish the
frontier, not to propose shipping it.

And read the last column before reading the third: parity is bought by approving
**96% of applicants**, which more than doubles the bad rate among approved from
6.7% to 14.9%. "Disparate impact 0.995" in isolation would be a deeply
misleading number.

## The tension that cannot be resolved

Calibration-by-group and equalized odds cannot both hold when base rates differ
across groups. That is a mathematical result, not an implementation gap. This
model prioritises **calibration** — a predicted PD means the same thing for
every band, within about a percentage point — and consequently does not satisfy
equalized odds.

That is a policy choice, and it belongs in the model card as one.

## Recommendation

Nothing here supports a claim that the model is fair on age. What it supports:

1. The disparity is real, measured, and driven substantially by cutoff placement.
2. No legally deployable mitigation reaches the four-fifths threshold.
3. Calibration by group is sound, so the model is not systematically wrong about
   any band.
4. This should be reviewed by counsel and a fair-lending specialist before the
   model is used for real decisions, with the cutoff curve above as the input to
   that conversation.

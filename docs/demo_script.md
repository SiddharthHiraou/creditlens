# Demo Script — three minutes

The brief asks for a recorded video. This is the script for it; the recording
itself is not something this build can produce, so the walkthrough is written
out instead and every command is real.

Setup, once: `make setup && make data && make train && make up`

---

## 0:00 — What it is (20s)

> "CreditLens decides consumer loan applications. An application goes in; a
> probability of default, a 300–850 score, an approve/refer/decline decision and
> four ECOA-compliant reasons come back in under 150 milliseconds. Around that
> sits what a bank needs before a model is allowed to decide anything —
> out-of-time validation, calibrated probabilities, drift monitoring, fairness
> measurement, an audit log, and an automated promotion gate."

Open `http://localhost:3000`. Headline numbers on screen: Gini 0.5827, KS 0.4349,
Brier 0.1148, PSI 0.0174.

---

## 0:20 — Score a real decline (40s)

```bash
curl -s -X POST localhost:8000/v1/score \
  -H "X-API-Key: demo-key-underwriter" -H "Content-Type: application/json" \
  -d @artifacts/one_payload.json | jq '{pd, score, decision, latency_ms}'
```

> "PD 29.8%, score 512, declined, nine milliseconds."

Switch to `/score` in the dashboard.

> "Four reasons, and they're four *different* reasons — not four ways of saying
> the debt-to-income is high. Contributions are summed into ECOA families first,
> so the applicant gets four real reasons instead of one reason four times.
> That holds for 100% of declines.
>
> The model's single strongest feature embeds age. It's disclosed as a credit
> bureau score reason, never as age — age is a prohibited basis. All 221
> features are either mapped to a reason family or explicitly suppressed, and a
> test fails the build if that ever stops being true."

---

## 1:00 — The cutoff simulator (35s)

Open `/portfolio`. Drag the approval-rate slider.

> "This is the conversation a credit committee actually has. Move the cutoff and
> the approval rate, bad rate, expected loss and profit all move. The profit curve
> marks its own maximum at 61%.
>
> The important part: the bad rate here is **observed**, not predicted. It runs
> over the out-of-time fold with realised outcomes, so it's showing what would
> actually have happened at each cutoff — not what the model thinks would have."

---

## 1:35 — Calibration (25s)

> "Class reweighting fixes discrimination and wrecks the probability. Mean
> predicted PD came out at 43% against a true rate of 17% — expected-loss
> arithmetic would have been wrong by two and a half times while every AUC-style
> metric looked fine.
>
> Isotonic on a held-out fold takes calibration error from 0.265 to 0.013. And
> the plain version turned out to be unusable as a *score*: it collapsed 16,000
> distinct values into 120, which broke cutoff placement. That's ADR 6."

---

## 2:00 — Fairness (30s)

Open `/fairness`.

> "Gender passes the four-fifths rule at 0.978. Age **fails**, at 0.384 — the
> 18-to-24 band is approved at 38% of the rate of the over-65s.
>
> I'm showing you a failure because a fairness section that only reports passes
> isn't worth reading. No legally deployable cutoff reaches 0.80. Group-specific
> thresholds do — and they're unlawful, and they get there by approving 96% of
> applicants and doubling the bad rate among approvals. Measured, quantified,
> documented. Not solved."

---

## 2:30 — The promotion gate (30s)

```bash
make drift      # PSI 0.0174 — stable
make promote
```

> "Five checks against the incumbent: AUC within 1%, calibration not worsening,
> PSI under 0.25, rank ordering monotonic, disparate impact not degrading.
>
> There's no promote-with-a-warning path. A failing gate writes an issue and
> leaves the incumbent serving — running the current model another month costs a
> known amount, promoting a quietly worse one doesn't.
>
> I tested it in both directions. A candidate identical to the incumbent passes
> all five. A deliberately crippled one gets refused on discrimination and
> fairness. A gate you've only tested against a good candidate is a gate you
> don't know works."

---

## 3:00 — Close (15s)

> "Two things I'd want stated up front rather than found later. It's trained on a
> synthetic generator, because neither Kaggle source downloads
> non-interactively — the loaders read real CSVs the moment they're there. And
> the two LLM features have never made a live API call, because there were no
> credentials on the machine; their guardrails are tested, their live behaviour
> isn't. Both are in the model card limitations.
>
> Everything else is reproducible: `make train && make audit && make docs`
> regenerates every number in the README and the governance documents."

---

## If asked: the questions this project is built to answer

| Question | Answer |
|---|---|
| Why not a random split? | Repeat applicants leak across folds and macro conditions are shared. Home Credit has no application date at all, so out-of-time work runs on Lending Club — ADR 2 |
| Why no SMOTE? | It interpolates between real defaulters to invent applicants who never applied, and it wrecks calibration. `scale_pos_weight` instead |
| Why not report accuracy? | At a 17% bad rate, "approve everyone" scores 83% |
| Is SHAP causal? | No, and the counterfactual module exists because "what drove the score" and "what would change the decision" are different questions |
| Does the GBDT earn its keep? | Barely — the WOE scorecard is within 0.0044 AUC. That's in the model card as an open question, not hidden |
| What if the model drifts? | PSI daily, alarm at 0.25, which stages a retraining candidate that must clear the gate |

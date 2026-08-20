# ADR 0008: The frontend is a static export over pre-computed snapshots

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 5

## Context

The frontend's job is to be a link someone can click. A portfolio dashboard that
returns a 502 because a free-tier container is asleep, or that needs a local
Postgres before it renders, has failed at the only thing it was for.

But the interesting parts of this project are interactive: the cutoff simulator
has to recompute approval rate, bad rate, expected loss and profit on every
slider tick.

## Decision

**Static export** (`output: "export"`), driven by JSON snapshots that
`src/api/export_demo.py` writes from the same artifacts the model pipeline
produces. Nothing on any page is hand-authored; regenerate with `make demo-data`.

**The simulator computes client-side** over a seeded 4,000-row sample of the
out-of-time fold, stratified by score so the tails survive. A round trip per
slider tick would make it feel broken, and 4,000 rows recompute instantly.

Crucially the sample carries **realised outcomes**, so the bad rate the
simulator shows at each cutoff is observed, not predicted. That is what makes it
an honest instrument rather than a plot of the model's own opinion.

**Server Components by default**, client components only where interaction
requires them: the simulator, the applicant picker, the theme toggle and the
nav. The chart-free pages ship no chart JavaScript.

## Consequences

**Accepted cost.** The scoring page shows pre-scored applicants rather than
accepting a live application. The full form-to-decision path exists at
`POST /v1/score`; the static page demonstrates the output rather than the round
trip.

**The snapshots are a build input, so CI builds the site.** If a page reaches
for data the export does not produce, that fails in CI rather than on the
deployed link.

**Total payload is ~140KB of JSON**, dominated by the 101KB simulator sample.
Small enough to ship in the page; large enough that it should not grow without
a reason.

**Charts never animate on mount** (`isAnimationActive: false`). Recharts drives
its entry animation with `requestAnimationFrame`, which does not advance in a
background tab — so a chart mounted in a hidden tab stays at frame zero with
axes drawn and no data, and stays that way after the tab is foregrounded. This
was found by rendering the page in a background tab and is a real user-facing
bug, not a test artifact. Static dashboards gain nothing from the animation.

**Theme colours live on `<html>`, not just `<body>`.** Painting only the body
leaves the canvas transparent above and below it, which flashes white during
overscroll in dark mode.

## Rejected alternatives

**Server-rendered against the live API.** Would make every page depend on the
container being awake, for no gain — the data is a snapshot of a trained model,
not live traffic.

**Mocking the copilot chat.** A chat panel returning scripted answers would demo
well and mean nothing. `/copilot` says what it will do, what tools it will call
and what guardrails it will carry, and states plainly that it is Phase 6 and
does not exist. An empty honest page beats a convincing fake one.

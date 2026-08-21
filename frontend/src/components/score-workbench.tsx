"use client";

import { useState } from "react";
import {
  Bar, BarChart, Cell as RCell, CartesianGrid, ReferenceLine, Tooltip, XAxis, YAxis,
} from "recharts";

import { AXIS, ChartFrame, NO_ANIM, PALETTE, TooltipCard } from "@/components/charts/primitives";
import { Badge, Card, Cell, Metric, Note, Row, Table } from "@/components/ui";
import { compactMoney, money, num, pct } from "@/lib/format";
import type { Applicant } from "@/lib/types";

/**
 * Underwriter view. Pick an applicant, see the decision, the reason codes and
 * the attributions behind them.
 *
 * Applicants are pre-scored snapshots rather than a live API call, so the
 * public demo works with no backend. The same payloads drive `POST /v1/score`
 * when the API is running.
 */
export function ScoreWorkbench({ applicants }: { applicants: Applicant[] }) {
  const [selectedId, setSelectedId] = useState(applicants[0].skIdCurr);
  const applicant = applicants.find((a) => a.skIdCurr === selectedId) ?? applicants[0];

  const tone =
    applicant.decision === "approve" ? "good" : applicant.decision === "refer" ? "warn" : "bad";

  const waterfall = [...applicant.shap]
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    .slice(0, 10)
    .reverse()
    .map((c) => ({ ...c, label: c.feature.length > 26 ? `${c.feature.slice(0, 24)}…` : c.feature }));

  return (
    <div className="space-y-6">
      <Card
        title="Pick an applicant"
        subtitle="A spread of real outcomes — approved, referred and declined. Colour shows the decision."
      >
        <div className="flex flex-wrap gap-2">
          {applicants.map((a) => (
            <button
              key={a.skIdCurr}
              type="button"
              onClick={() => setSelectedId(a.skIdCurr)}
              aria-pressed={a.skIdCurr === selectedId}
              className={`rounded-md border px-2.5 py-1.5 text-xs tabular-nums transition-colors ${
                a.skIdCurr === selectedId
                  ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 font-medium text-[var(--color-accent)]"
                  : "border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)]"
              }`}
            >
              {a.skIdCurr}
              <span
                className="ml-2 inline-block h-1.5 w-1.5 rounded-full align-middle"
                style={{
                  background:
                    a.decision === "approve" ? PALETTE.approve
                      : a.decision === "refer" ? PALETTE.refer : PALETTE.decline,
                }}
              />
            </button>
          ))}
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
        <div className="space-y-6">
          <Card title="The decision" action={<Badge tone={tone}>{applicant.decision}</Badge>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Metric label="Credit score" value={num(applicant.score, 0)} tone={tone} />
              <Metric label="Chance of default" value={pct(applicant.pd, 2)} tone={tone} />
              <Metric label="Amount requested" value={compactMoney(applicant.exposure)} />
              <Metric label="Expected loss" value={compactMoney(applicant.expectedLoss)}
                hint="What we lose on average: risk × 65% unrecovered × amount." />
            </div>
          </Card>

          <Card title="Applicant profile">
            <Table head={["Attribute", "Value"]} dense>
              <Row><Cell align="left" mono={false}>Income</Cell><Cell>{money(applicant.profile.income)}</Cell></Row>
              <Row><Cell align="left" mono={false}>Requested credit</Cell><Cell>{money(applicant.profile.credit)}</Cell></Row>
              <Row><Cell align="left" mono={false}>Annuity</Cell><Cell>{money(applicant.profile.annuity)}</Cell></Row>
              <Row><Cell align="left" mono={false}>Age</Cell><Cell>{applicant.profile.ageYears} yrs</Cell></Row>
              <Row><Cell align="left" mono={false}>Employment tenure</Cell><Cell>{applicant.profile.employedYears} yrs</Cell></Row>
              <Row><Cell align="left" mono={false}>Education</Cell><Cell mono={false}>{applicant.profile.education}</Cell></Row>
              <Row><Cell align="left" mono={false}>Occupation</Cell><Cell mono={false}>{applicant.profile.occupation}</Cell></Row>
              <Row><Cell align="left" mono={false}>Contract</Cell><Cell mono={false}>{applicant.profile.contractType}</Cell></Row>
            </Table>
          </Card>
        </div>

        <div className="space-y-6">
          <Card
            title="Adverse action reasons"
            subtitle={
              applicant.reasonCodes.length
                ? "Four distinct principal reasons, ranked by contribution, as required under ECOA and Regulation B."
                : undefined
            }
          >
            {applicant.reasonCodes.length === 0 ? (
              <Note>
                This applicant was approved. Reason codes are an adverse action
                artifact — an approved applicant has not been denied anything, so
                issuing them &ldquo;reasons&rdquo; would be meaningless and, in a
                letter, misleading.
              </Note>
            ) : (
              <ol className="space-y-3">
                {applicant.reasonCodes.map((rc) => (
                  <li key={rc.rank} className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3">
                    <div className="flex items-start gap-3">
                      <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded bg-[var(--surface-raised)] text-[11px] font-semibold">
                        {rc.rank}
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-xs font-semibold">{rc.label}</span>
                          {rc.actionable && <Badge tone="accent">actionable</Badge>}
                        </div>
                        <p className="mt-1 text-sm leading-snug">{rc.phrase}</p>
                        <p className="mt-1.5 font-mono text-[10px] text-[var(--text-muted)]">
                          {rc.driving_features.join(" · ")}
                        </p>
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </Card>

          <Card
            title="What moved the decision"
            subtitle="The ten strongest factors. Red pushed toward declining, green toward approving."
          >
            <ChartFrame height={320}>
              <BarChart data={waterfall} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
                <CartesianGrid strokeDasharray="2 4" horizontal={false} />
                <XAxis type="number" {...AXIS} />
                <YAxis type="category" dataKey="label" {...AXIS} width={150} interval={0} />
                <Tooltip content={<TooltipCard formatter={(_n, v) => v.toFixed(4)} />} />
                <ReferenceLine x={0} stroke="var(--border)" />
                <Bar {...NO_ANIM} dataKey="shap" name="Contribution" radius={[0, 2, 2, 0]}>
                  {waterfall.map((c, i) => (
                    <RCell key={i} fill={c.shap > 0 ? PALETTE.decline : PALETTE.approve} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ChartFrame>
            <div className="mt-4">
              <Note>
                These are attributions, not causes. A SHAP value says how much a feature
                moved this model&apos;s output relative to a baseline of{" "}
                {num(applicant.baseValue, 4)}, given the other features. It does not say
                that changing the feature would change the applicant&apos;s real risk.
              </Note>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

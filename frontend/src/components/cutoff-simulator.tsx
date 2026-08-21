"use client";

import { useMemo, useState } from "react";
import {
  Area, AreaChart, CartesianGrid, Line, ReferenceLine, Tooltip, XAxis, YAxis,
} from "recharts";

import { Badge, Card, Metric, Note } from "@/components/ui";
import { AXIS, ChartFrame, NO_ANIM, PALETTE, TooltipCard } from "@/components/charts/primitives";
import { compactMoney, count, pct } from "@/lib/format";
import type { SimulatorData } from "@/lib/types";

/**
 * The demo moment: a cutoff slider that recomputes approval rate, bad rate,
 * expected loss and profit live.
 *
 * Everything runs client-side over a seeded 4,000-row sample of the
 * out-of-time fold. A round trip per slider tick would make it feel broken,
 * and the sample is small enough that a full recompute is instant.
 *
 * Realised outcomes are what make this honest: bad rate is *observed*, not
 * predicted, so the curve shows what would actually have happened at each
 * cutoff rather than what the model thinks would have.
 */
export function CutoffSimulator({ data }: { data: SimulatorData }) {
  const [approvalRate, setApprovalRate] = useState(0.6);
  const [margin, setMargin] = useState(0.12);

  // Sort once; every slider tick is then a slice rather than a re-sort.
  const sorted = useMemo(() => {
    const rows = data.score.map((score, i) => ({
      score, pd: data.pd[i], y: data.y[i], exposure: data.exposure[i],
    }));
    rows.sort((a, b) => b.score - a.score);
    return rows;
  }, [data]);

  const evaluate = useMemo(
    () => (rate: number, interestMargin: number) => {
      const n = Math.max(1, Math.round(sorted.length * rate));
      const approved = sorted.slice(0, n);
      let bad = 0, expectedLoss = 0, income = 0, exposure = 0;
      for (const row of approved) {
        bad += row.y;
        expectedLoss += row.pd * data.lgd * row.exposure;
        income += row.exposure * interestMargin * (1 - row.pd);
        exposure += row.exposure;
      }
      return {
        approvalRate: n / sorted.length,
        cutoff: approved[approved.length - 1]?.score ?? 0,
        badRate: bad / n,
        expectedLoss,
        income,
        profit: income - expectedLoss,
        exposure,
        n,
      };
    },
    [sorted, data.lgd],
  );

  const current = useMemo(() => evaluate(approvalRate, margin), [evaluate, approvalRate, margin]);

  const curve = useMemo(() => {
    const points = [];
    for (let rate = 0.05; rate <= 0.98; rate += 0.01) {
      const r = evaluate(rate, margin);
      points.push({
        approvalRate: Math.round(r.approvalRate * 1000) / 10,
        badRate: r.badRate * 100,
        profit: r.profit,
        expectedLoss: r.expectedLoss,
        cutoff: r.cutoff,
      });
    }
    return points;
  }, [evaluate, margin]);

  const best = useMemo(
    () => curve.reduce((a, b) => (b.profit > a.profit ? b : a), curve[0]),
    [curve],
  );

  return (
    <div className="space-y-6">
      <Card
        title="Move the approval rate"
        subtitle={`A ${count(data.score.length)}-loan sample of the ${count(data.sampledFrom)} test applications, all with known outcomes. The default rate below is what happened, not a forecast.`}
        action={
          <Badge tone={current.profit >= best.profit * 0.98 ? "good" : "neutral"}>
            {current.profit >= best.profit * 0.98 ? "near profit-optimal" : "off optimum"}
          </Badge>
        }
      >
        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          <div className="space-y-5">
            <div>
              <div className="mb-2 flex items-baseline justify-between">
                <label htmlFor="approval" className="text-xs font-medium">
                  Approve this share of applicants
                </label>
                <span className="text-sm font-semibold tabular-nums">
                  {pct(current.approvalRate, 1)}
                </span>
              </div>
              <input
                id="approval"
                type="range"
                min={5}
                max={98}
                step={1}
                value={Math.round(approvalRate * 100)}
                onChange={(e) => setApprovalRate(Number(e.target.value) / 100)}
                className="w-full accent-[var(--color-accent)]"
              />
              <p className="mt-1.5 text-xs tabular-nums text-[var(--text-muted)]">
                cutoff score {current.cutoff.toFixed(0)} · {count(current.n)} of {count(sorted.length)} approved
              </p>
            </div>

            <div>
              <div className="mb-2 flex items-baseline justify-between">
                <label htmlFor="margin" className="text-xs font-medium">
                  Interest margin we earn
                </label>
                <span className="text-sm font-semibold tabular-nums">{pct(margin, 0)}</span>
              </div>
              <input
                id="margin"
                type="range"
                min={4}
                max={30}
                step={1}
                value={Math.round(margin * 100)}
                onChange={(e) => setMargin(Number(e.target.value) / 100)}
                className="w-full accent-[var(--color-accent)]"
              />
              <p className="mt-1.5 text-xs text-[var(--text-muted)]">
                Annual margin on a performing loan. LGD fixed at {pct(data.lgd, 0)}.
              </p>
            </div>

            <button
              type="button"
              onClick={() => setApprovalRate(best.approvalRate / 100)}
              className="w-full rounded-md border border-[var(--color-accent)]/40 bg-[var(--color-accent)]/10 px-3 py-2 text-xs font-medium text-[var(--color-accent)] transition-colors hover:bg-[var(--color-accent)]/20"
            >
              Jump to profit-maximising cutoff ({best.approvalRate.toFixed(0)}%)
            </button>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <Metric label="Of those approved, this share defaults" value={pct(current.badRate, 2)}
              tone={current.badRate > 0.12 ? "bad" : current.badRate > 0.08 ? "warn" : "good"} />
            <Metric label="Money we expect to lose" value={compactMoney(current.expectedLoss)}
              hint="Across every loan approved at this setting." />
            <Metric label="Interest we expect to earn" value={compactMoney(current.income)}
              hint="From the loans expected to be repaid." />
            <Metric label="Profit" value={compactMoney(current.profit)}
              tone={current.profit > 0 ? "good" : "bad"} compare={`optimum ${compactMoney(best.profit)}`} />
          </div>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Where profit peaks" subtitle="Interest earned, minus what we expect to lose, at every approval rate.">
          <ChartFrame>
            <AreaChart data={curve} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="profitFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={PALETTE.accent} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={PALETTE.accent} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="approvalRate" {...AXIS} unit="%" />
              <YAxis {...AXIS} tickFormatter={(v: number) => compactMoney(v)} width={52} />
              <Tooltip
                content={<TooltipCard formatter={(_n, v) => compactMoney(v)} />}
                labelFormatter={(l) => `Approval rate ${l}%`}
              />
              <Area {...NO_ANIM} type="monotone" dataKey="profit" name="Profit" stroke={PALETTE.accent}
                strokeWidth={2} fill="url(#profitFill)" />
              <ReferenceLine x={best.approvalRate} stroke={PALETTE.approve} strokeDasharray="4 4"
                label={{ value: "optimum", position: "top", fontSize: 10, fill: PALETTE.approve }} />
              <ReferenceLine x={Math.round(current.approvalRate * 1000) / 10}
                stroke={PALETTE.muted} strokeWidth={1.5} />
            </AreaChart>
          </ChartFrame>
        </Card>

        <Card title="The cost of approving more" subtitle="Every additional approval brings in more defaults. This is the trade.">
          <ChartFrame>
            <AreaChart data={curve} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="badFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={PALETTE.decline} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={PALETTE.decline} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="approvalRate" {...AXIS} unit="%" />
              <YAxis {...AXIS} unit="%" width={40} />
              <Tooltip
                content={<TooltipCard formatter={(_n, v) => `${v.toFixed(2)}%`} />}
                labelFormatter={(l) => `Approval rate ${l}%`}
              />
              <Area {...NO_ANIM} type="monotone" dataKey="badRate" name="Bad rate" stroke={PALETTE.decline}
                strokeWidth={2} fill="url(#badFill)" />
              <ReferenceLine x={Math.round(current.approvalRate * 1000) / 10}
                stroke={PALETTE.muted} strokeWidth={1.5} />
            </AreaChart>
          </ChartFrame>
        </Card>
      </div>

      <Note>
        Profit here assumes every approved loan earns the margin on its full exposure for
        a year, weighted by the probability it performs. It is a comparison device for
        moving the cutoff, not a P&amp;L: no funding cost, no operational cost, no
        prepayment, and recoveries only through the {pct(data.lgd, 0)} LGD assumption.
      </Note>
    </div>
  );
}

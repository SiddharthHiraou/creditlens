"use client";

import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line,
  ReferenceLine, Tooltip, XAxis, YAxis,
} from "recharts";

import { AXIS, ChartFrame, NO_ANIM, PALETTE, TooltipCard } from "./primitives";
import type { Fairness, GroupRow } from "@/lib/types";

export function SelectionRateChart({ rows }: { rows: GroupRow[] }) {
  const data = rows.map((r) => ({
    group: r.group,
    selection: r.selectionRate * 100,
    badRate: r.observedBadRate * 100,
  }));
  return (
    <ChartFrame height={260}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="group" {...AXIS} />
        <YAxis {...AXIS} unit="%" width={44} />
        <Tooltip content={<TooltipCard formatter={(_n, v) => `${v.toFixed(1)}%`} />} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar {...NO_ANIM} dataKey="selection" name="Approval rate" fill={PALETTE.accent}
          fillOpacity={0.85} radius={[3, 3, 0, 0]} />
        <Line {...NO_ANIM} type="monotone" dataKey="badRate" name="Observed bad rate"
          stroke={PALETTE.decline} strokeWidth={2} dot={{ r: 3 }} />
      </ComposedChart>
    </ChartFrame>
  );
}

export function CalibrationByGroupChart({ rows }: { rows: GroupRow[] }) {
  const data = rows.map((r) => ({
    group: r.group,
    predicted: r.meanPredictedPd * 100,
    observed: r.observedBadRate * 100,
  }));
  return (
    <ChartFrame height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="group" {...AXIS} />
        <YAxis {...AXIS} unit="%" width={44} />
        <Tooltip content={<TooltipCard formatter={(_n, v) => `${v.toFixed(2)}%`} />} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar {...NO_ANIM} dataKey="predicted" name="Mean predicted PD" fill={PALETTE.accent}
          fillOpacity={0.8} radius={[3, 3, 0, 0]} />
        <Bar {...NO_ANIM} dataKey="observed" name="Observed bad rate" fill={PALETTE.violet}
          fillOpacity={0.8} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ChartFrame>
  );
}

export function MitigationTradeoffChart({ curve, fourFifths }: {
  curve: Fairness["cutoffCurve"]; fourFifths: number;
}) {
  const data = curve.map((c) => ({
    approvalRate: Math.round(c.approvalRate * 100),
    disparateImpact: c.disparateImpact,
    badRate: c.badRateAmongApproved * 100,
  }));
  return (
    <ChartFrame height={280}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="approvalRate" {...AXIS} unit="%" />
        <YAxis yAxisId="di" {...AXIS} domain={[0, 1]} width={40} />
        <YAxis yAxisId="bad" orientation="right" {...AXIS} unit="%" width={44} />
        <Tooltip content={<TooltipCard />} labelFormatter={(l) => `Approval rate ${l}%`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine yAxisId="di" y={fourFifths} stroke={PALETTE.approve} strokeDasharray="4 4"
          label={{ value: "four-fifths 0.80", position: "insideTopLeft", fontSize: 9, fill: PALETTE.approve }} />
        <Bar {...NO_ANIM} yAxisId="di" dataKey="disparateImpact" name="Disparate impact" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.disparateImpact >= fourFifths ? PALETTE.approve : PALETTE.decline}
              fillOpacity={0.7} />
          ))}
        </Bar>
        <Line {...NO_ANIM} yAxisId="bad" type="monotone" dataKey="badRate" name="Bad rate among approved"
          stroke={PALETTE.refer} strokeWidth={2} dot={{ r: 3 }} />
      </ComposedChart>
    </ChartFrame>
  );
}

"use client";

import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Tooltip, XAxis, YAxis,
} from "recharts";

import { AXIS, ChartFrame, NO_ANIM, PALETTE, TooltipCard } from "./primitives";
import type { Portfolio } from "@/lib/types";

export function ScoreDistribution({ data, approveAt, referAt }: {
  data: Portfolio["scoreDistribution"]; approveAt: number; referAt: number;
}) {
  return (
    <ChartFrame height={240}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="score" {...AXIS} />
        <YAxis {...AXIS} width={44} />
        <Tooltip content={<TooltipCard formatter={(_n, v) => v.toLocaleString()} />}
          labelFormatter={(l) => `Score ≈ ${l}`} />
        <Bar {...NO_ANIM} dataKey="count" name="Applications" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => (
            <Cell
              key={i}
              fill={
                d.score >= approveAt ? PALETTE.approve
                  : d.score >= referAt ? PALETTE.refer
                    : PALETTE.decline
              }
              fillOpacity={0.75}
            />
          ))}
        </Bar>
      </BarChart>
    </ChartFrame>
  );
}

export function DecileChart({ data }: { data: Portfolio["deciles"] }) {
  const rows = data.map((d) => ({ ...d, badRatePct: d.badRate * 100 }));
  return (
    <ChartFrame height={240}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="decile" {...AXIS} />
        <YAxis {...AXIS} unit="%" width={44} />
        <Tooltip content={<TooltipCard formatter={(_n, v) => `${v.toFixed(2)}%`} />}
          labelFormatter={(l) => `Decile ${l} (1 = riskiest)`} />
        <Bar {...NO_ANIM} dataKey="badRatePct" name="Observed bad rate" fill={PALETTE.accent}
          fillOpacity={0.85} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ChartFrame>
  );
}

export function VintageChart({ data }: { data: Portfolio["vintages"] }) {
  const rows = data.map((v) => ({
    ...v, badRatePct: v.badRate * 100, meanPdPct: v.meanPd * 100,
  }));
  return (
    <ChartFrame height={240}>
      <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="vintage" {...AXIS} />
        <YAxis {...AXIS} unit="%" width={44} />
        <Tooltip content={<TooltipCard formatter={(_n, v) => `${v.toFixed(2)}%`} />} />
        <Line {...NO_ANIM} type="monotone" dataKey="badRatePct" name="Observed bad rate"
          stroke={PALETTE.decline} strokeWidth={2} dot={{ r: 2.5 }} />
        <Line {...NO_ANIM} type="monotone" dataKey="meanPdPct" name="Mean predicted PD"
          stroke={PALETTE.accent} strokeWidth={2} strokeDasharray="4 3" dot={{ r: 2.5 }} />
      </LineChart>
    </ChartFrame>
  );
}

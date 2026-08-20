"use client";

import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ReferenceLine, Tooltip, XAxis, YAxis,
} from "recharts";

import { AXIS, ChartFrame, NO_ANIM, PALETTE, TooltipCard } from "./primitives";
import type { Monitoring } from "@/lib/types";

export function VintagePsiChart({ data, thresholds }: {
  data: Monitoring["vintagePsi"]; thresholds: { investigate: number; alarm: number };
}) {
  return (
    <ChartFrame height={240}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="vintage" {...AXIS} />
        <YAxis {...AXIS} width={48} />
        <Tooltip content={<TooltipCard formatter={(_n, v) => v.toFixed(4)} />} />
        <ReferenceLine y={thresholds.investigate} stroke={PALETTE.refer} strokeDasharray="4 4"
          label={{ value: "investigate 0.10", position: "right", fontSize: 9, fill: PALETTE.refer }} />
        <ReferenceLine y={thresholds.alarm} stroke={PALETTE.decline} strokeDasharray="4 4"
          label={{ value: "alarm 0.25", position: "right", fontSize: 9, fill: PALETTE.decline }} />
        <Line {...NO_ANIM} type="monotone" dataKey="psi" name="PSI" stroke={PALETTE.accent}
          strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ChartFrame>
  );
}

export function PsiBinsChart({ data }: { data: Monitoring["psiBins"] }) {
  const rows = data.map((b) => ({
    bin: `b${b.bin}`, expected: b.expected * 100, actual: b.actual * 100,
  }));
  return (
    <ChartFrame height={240}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="bin" {...AXIS} />
        <YAxis {...AXIS} unit="%" width={40} />
        <Tooltip content={<TooltipCard formatter={(_n, v) => `${v.toFixed(2)}%`} />} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar {...NO_ANIM} dataKey="expected" name="Training baseline" fill={PALETTE.muted} fillOpacity={0.6} radius={[2, 2, 0, 0]} />
        <Bar {...NO_ANIM} dataKey="actual" name="Out-of-time" fill={PALETTE.accent} fillOpacity={0.85} radius={[2, 2, 0, 0]} />
      </BarChart>
    </ChartFrame>
  );
}

export function CsiChart({ data }: { data: Monitoring["featureCsi"] }) {
  const rows = [...data].slice(0, 12).reverse().map((f) => ({
    ...f, label: f.feature.length > 24 ? `${f.feature.slice(0, 22)}…` : f.feature,
  }));
  return (
    <ChartFrame height={320}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="2 4" horizontal={false} />
        <XAxis type="number" {...AXIS} />
        <YAxis type="category" dataKey="label" {...AXIS} width={150} interval={0} />
        <Tooltip content={<TooltipCard formatter={(_n, v) => v.toFixed(5)} />} />
        <Bar {...NO_ANIM} dataKey="csi" name="CSI" radius={[0, 2, 2, 0]}>
          {rows.map((r, i) => (
            <Cell key={i}
              fill={r.csi >= 0.25 ? PALETTE.decline : r.csi >= 0.1 ? PALETTE.refer : PALETTE.approve}
              fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ChartFrame>
  );
}

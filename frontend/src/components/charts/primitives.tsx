"use client";

import type { ReactNode } from "react";
import { ResponsiveContainer } from "recharts";

export const AXIS = {
  tickLine: false,
  axisLine: false,
  tick: { fontSize: 11 },
} as const;

/**
 * Series never animate on mount.
 *
 * Recharts drives its entry animation with requestAnimationFrame, which does
 * not advance while a tab is in the background. A chart mounted in a hidden
 * tab therefore stays at frame zero: axes and gridlines render, the data
 * never appears, and it stays that way after the tab is foregrounded. Anyone
 * who opens the dashboard in a background tab sees empty charts.
 *
 * Static dashboards gain nothing from the animation anyway.
 */
export const NO_ANIM = { isAnimationActive: false } as const;

// Kept in sync by hand with the CSS custom properties in globals.css. Recharts
// renders to SVG attributes and cannot read var(--color-*), so these are the one
// place hex values are duplicated, so change both together.
export const PALETTE = {
  accent: "#7c5cff",
  accent2: "#35e0f2",
  approve: "#2ee6a8",
  refer: "#ffb44d",
  decline: "#ff5d6c",
  muted: "#8d95ab",
  violet: "#a78bfa",
} as const;

/** Recharts needs a sized parent; this keeps every chart on one height scale. */
export function ChartFrame({
  height = 260, children,
}: {
  height?: number; children: ReactNode;
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        {children as never}
      </ResponsiveContainer>
    </div>
  );
}

export function TooltipCard({
  active, payload, label, formatter,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number | string; color?: string }[];
  label?: string | number;
  formatter?: (name: string, value: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs shadow-lg">
      {label !== undefined && (
        <p className="mb-1 font-medium text-[var(--text)]">{label}</p>
      )}
      {payload.map((entry, i) => (
        <p key={i} className="flex items-center gap-2 tabular-nums text-[var(--text-muted)]">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: entry.color }}
          />
          <span>{entry.name}</span>
          <span className="ml-auto font-medium text-[var(--text)]">
            {formatter && typeof entry.value === "number"
              ? formatter(entry.name ?? "", entry.value)
              : String(entry.value)}
          </span>
        </p>
      ))}
    </div>
  );
}

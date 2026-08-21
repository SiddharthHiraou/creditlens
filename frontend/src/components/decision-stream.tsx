"use client";

import { useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui";
import type { SimulatorData } from "@/lib/types";

/**
 * A live view of applications being decided.
 *
 * Every dot is a **real loan from the out-of-time test set**. Its lane comes
 * from the model's actual score against the current cutoff, and the ring marks
 * loans that genuinely went on to default. So the visual is not decoration: the
 * red rings landing in the approve lane are the mistakes the model really makes,
 * and there are honestly some.
 *
 * Canvas rather than DOM because a few hundred moving nodes in React is a
 * dropped-frame machine. One rAF loop, one draw call per frame.
 */

type Dot = {
  x: number;
  y: number;
  targetY: number;
  lane: 0 | 1 | 2;
  defaulted: boolean;
  speed: number;
  scored: boolean;
};

// Matches --color-approve / --color-refer / --color-decline in globals.css.
const LANE_COLORS = ["#2ee6a8", "#ffb44d", "#ff5d6c"] as const;
const LANE_LABELS = ["Approved", "Referred", "Declined"] as const;

export function DecisionStream({
  data, approveAt, referAt,
}: {
  data: SimulatorData; approveAt: number; referAt: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [running, setRunning] = useState(true);
  const [reduced, setReduced] = useState(false);

  // Counters are written straight into the DOM from the animation loop rather
  // than held in React state. Re-rendering a component several times a second
  // to move four numbers is wasted work, and it makes the display depend on
  // React's scheduler keeping up with rAF.
  const laneRefs = [
    useRef<HTMLSpanElement>(null),
    useRef<HTMLSpanElement>(null),
    useRef<HTMLSpanElement>(null),
  ];
  const lanePctRefs = [
    useRef<HTMLSpanElement>(null),
    useRef<HTMLSpanElement>(null),
    useRef<HTMLSpanElement>(null),
  ];
  const missedRef = useRef<HTMLSpanElement>(null);
  const missedPctRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    setReduced(window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }, []);

  useEffect(() => {
    if (reduced) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = 0;
    let height = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const dots: Dot[] = [];

    // The exported sample is sorted ascending by score so the cutoff simulator
    // can slice it cheaply. Walking it in order here would replay a contiguous
    // band, and the stream showed an empty approve lane and a solid wall of
    // declines. Shuffle once so the flow is representative of the whole book.
    const order = Array.from({ length: data.score.length }, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    let cursor = 0;
    let frame = 0;
    let raf = 0;
    const tally: [number, number, number] = [0, 0, 0];
    let missedCount = 0;

    const gateX = () => width * 0.42;

    // Speed is derived from the canvas width so a dot always crosses in about
    // three seconds. A fixed pixels-per-frame looks fine on a phone and leaves
    // a 1700px desktop canvas looking frozen, at twenty seconds per crossing.
    const baseSpeed = () => Math.max(width / (3 * 60), 2.2);

    const spawn = (atX?: number) => {
      const i = order[cursor % order.length];
      cursor += 1;
      const score = data.score[i];
      const lane: 0 | 1 | 2 = score >= approveAt ? 0 : score >= referAt ? 1 : 2;
      const x = atX ?? -8;
      const scored = x >= gateX();
      dots.push({
        x,
        y: scored
          ? [height * 0.22, height * 0.5, height * 0.78][lane]
          : height / 2 + (Math.random() - 0.5) * height * 0.42,
        targetY: [height * 0.22, height * 0.5, height * 0.78][lane],
        lane,
        defaulted: data.y[i] === 1,
        speed: baseSpeed() * (0.85 + Math.random() * 0.4),
        scored,
      });
      if (scored) {
        tally[lane] += 1;
        if (lane === 0 && data.y[i] === 1) missedCount += 1;
      }
    };

    // Seed the canvas so it reads as already-running on first paint rather
    // than filling in awkwardly from the left edge.
    for (let k = 0; k < 90; k++) spawn(Math.random() * width);

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      const gx = gateX();
      const laneY = [height * 0.22, height * 0.5, height * 0.78];

      // Lane rails on the right half.
      for (let l = 0; l < 3; l++) {
        ctx.strokeStyle = LANE_COLORS[l] + "22";
        ctx.lineWidth = 22;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(gx + 24, laneY[l]);
        ctx.lineTo(width - 14, laneY[l]);
        ctx.stroke();
      }

      // The gate: where the model makes the call.
      const pulse = 0.5 + 0.5 * Math.sin(frame / 26);
      ctx.strokeStyle = `rgba(124,92,255,${0.35 + pulse * 0.4})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(gx, height * 0.1);
      ctx.lineTo(gx, height * 0.9);
      ctx.stroke();

      for (let i = dots.length - 1; i >= 0; i--) {
        const d = dots[i];
        d.x += d.speed;

        if (!d.scored && d.x >= gx) {
          d.scored = true;
          d.targetY = laneY[d.lane];
          tally[d.lane] += 1;
          if (d.lane === 0 && d.defaulted) missedCount += 1;
        }
        // Ease into the assigned lane after the gate.
        if (d.scored) d.y += (d.targetY - d.y) * 0.09;

        const colour = d.scored ? LANE_COLORS[d.lane] : "#7c5cff";
        ctx.beginPath();
        ctx.arc(d.x, d.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = colour;
        ctx.globalAlpha = d.scored ? 0.95 : 0.5;
        ctx.fill();
        ctx.globalAlpha = 1;

        // A ring marks a loan that actually defaulted.
        if (d.scored && d.defaulted) {
          ctx.beginPath();
          ctx.arc(d.x, d.y, 6, 0, Math.PI * 2);
          ctx.strokeStyle = colour;
          ctx.lineWidth = 1.2;
          ctx.globalAlpha = 0.75;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }

        if (d.x > width + 12) dots.splice(i, 1);
      }

      frame += 1;
      if (frame % 3 === 0 && dots.length < 260) spawn();
      if (frame % 10 === 0) paintCounters();
      raf = requestAnimationFrame(draw);
    };

    const paintCounters = () => {
      const total = tally[0] + tally[1] + tally[2];
      for (let l = 0; l < 3; l++) {
        const el = laneRefs[l].current;
        if (el) el.textContent = tally[l].toLocaleString();
        const pct = lanePctRefs[l].current;
        if (pct) pct.textContent = total ? `${((tally[l] / total) * 100).toFixed(0)}%` : "";
      }
      if (missedRef.current) missedRef.current.textContent = missedCount.toLocaleString();
      if (missedPctRef.current) {
        missedPctRef.current.textContent = tally[0]
          ? `${((missedCount / tally[0]) * 100).toFixed(1)}%`
          : "";
      }
    };

    paintCounters();
    if (running) raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [data, approveAt, referAt, running, reduced]);

  return (
    <section className="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface-raised)]">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] px-5 py-3.5">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Applications being decided</h2>
          <p className="mt-0.5 text-xs text-[var(--text-muted)]">
            Every dot is a real loan from the test set. A ring means it went on to default.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone="accent">live</Badge>
          {!reduced && (
            <button
              type="button"
              onClick={() => setRunning((v) => !v)}
              className="rounded-md border border-[var(--border)] px-2.5 py-1 text-xs text-[var(--text-muted)] transition-colors hover:text-[var(--text)]"
            >
              {running ? "Pause" : "Play"}
            </button>
          )}
        </div>
      </header>

      {reduced ? (
        <div className="px-5 py-8 text-sm text-[var(--text-muted)]">
          Animation disabled to respect your reduced-motion setting. At the current
          cutoff the model approves 60% of applicants, refers 10%, and declines 30%.
        </div>
      ) : (
        <canvas ref={canvasRef} className="block h-[260px] w-full" aria-hidden />
      )}

      <div className="grid grid-cols-2 gap-px border-t border-[var(--border)] bg-[var(--border)] sm:grid-cols-4">
        {LANE_LABELS.map((label, i) => (
          <div key={label} className="bg-[var(--surface-raised)] px-5 py-3">
            <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: LANE_COLORS[i] }}
              />
              {label}
            </p>
            <p className="mt-1 text-xl font-semibold tabular-nums">
              <span ref={laneRefs[i]}>0</span>
              <span
                ref={lanePctRefs[i]}
                className="ml-1.5 text-xs font-normal text-[var(--text-muted)]"
              />
            </p>
          </div>
        ))}
        <div className="bg-[var(--surface-raised)] px-5 py-3">
          <p className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Approved that defaulted
          </p>
          <p className="mt-1 text-xl font-semibold tabular-nums text-[var(--color-decline)]">
            <span ref={missedRef}>0</span>
            <span
              ref={missedPctRef}
              className="ml-1.5 text-xs font-normal text-[var(--text-muted)]"
            />
          </p>
        </div>
      </div>
    </section>
  );
}

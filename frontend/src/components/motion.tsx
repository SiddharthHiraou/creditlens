"use client";

import clsx from "clsx";
import { useEffect, useRef, useState, type ReactNode } from "react";

/** Respect the OS setting once, on mount. */
function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

/**
 * Fade-and-rise a block the first time it scrolls into view.
 *
 * Deliberately one-shot: re-animating on every scroll past is the thing that
 * makes motion feel cheap. `delay` staggers siblings.
 */
export function Reveal({
  children, delay = 0, className,
}: {
  children: ReactNode; delay?: number; className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    if (reduced) return setShown(true);
    const el = ref.current;
    if (!el) return;

    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 },
    );
    io.observe(el);

    // Fail open. The pre-reveal state is opacity 0, so anything that stops the
    // observer from firing — a background tab suppressing callbacks, an old
    // browser, a script error elsewhere on the page — would leave the content
    // permanently invisible. A blank page is a far worse outcome than a missed
    // animation, so reveal unconditionally after a short grace period.
    const failsafe = window.setTimeout(() => {
      setShown(true);
      io.disconnect();
    }, 1200);

    return () => {
      window.clearTimeout(failsafe);
      io.disconnect();
    };
  }, [reduced]);

  return (
    <div
      ref={ref}
      className={clsx(className, shown && !reduced && "reveal")}
      style={{
        animationDelay: shown && !reduced ? `${delay}ms` : undefined,
        opacity: shown || reduced ? undefined : 0,
      }}
    >
      {children}
    </div>
  );
}

/**
 * Count a number up when it first becomes visible, and re-run smoothly
 * whenever the value changes afterwards.
 *
 * Eased rather than linear, and short — a slow counter on a dashboard reads as
 * a loading bug. Reduced motion skips straight to the value.
 */
export function useCountUp(target: number, { duration = 900 }: { duration?: number } = {}) {
  const [value, setValue] = useState(target);
  const from = useRef(target);
  const raf = useRef<number>(undefined);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    if (reduced) {
      setValue(target);
      from.current = target;
      return;
    }
    const start = performance.now();
    const origin = from.current;
    const delta = target - origin;
    if (delta === 0) return;

    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      // easeOutCubic: fast to begin with, settles gently on the final digit.
      const eased = 1 - (1 - t) ** 3;
      setValue(origin + delta * eased);
      if (t < 1) raf.current = requestAnimationFrame(tick);
      else from.current = target;
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
      from.current = target;
    };
  }, [target, duration, reduced]);

  return value;
}

/** A number that animates to its value, formatted by the caller. */
export function CountUp({
  value, format, className,
}: {
  value: number; format: (v: number) => string; className?: string;
}) {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        setVisible(true);
        io.disconnect();
      }
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const animated = useCountUp(visible ? value : 0);
  return (
    <span ref={ref} className={className}>
      {format(visible ? animated : 0)}
    </span>
  );
}

/**
 * Reveal each direct child in sequence as the page scrolls.
 *
 * Applied at page level so individual pages don't have to wrap every section
 * by hand. The stagger is capped — past about half a second the delay stops
 * reading as choreography and starts reading as slowness.
 */
export function Stagger({
  children, step = 70, max = 420, className,
}: {
  children: ReactNode; step?: number; max?: number; className?: string;
}) {
  const items = Array.isArray(children) ? children : [children];
  return (
    <div className={className}>
      {items.map((child, i) => (
        <Reveal key={i} delay={Math.min(i * step, max)}>
          {child}
        </Reveal>
      ))}
    </div>
  );
}

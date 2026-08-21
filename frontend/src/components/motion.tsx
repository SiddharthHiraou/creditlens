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
 * Fade-and-rise a block on page load.
 *
 * Deliberately **CSS-only**. The previous version hid the element at
 * `opacity: 0` and waited for an IntersectionObserver, with a `setTimeout`
 * failsafe. Both are throttled in a background tab, so a page opened in one
 * rendered completely blank (the nav and nothing else) and stayed that way.
 * The failsafe shared the exact weakness of the thing it was meant to protect.
 *
 * Now the animation is pure decoration: the keyframe runs on its own, browsers
 * resume it on focus, and if the animation never runs at all the content is
 * simply visible. There is no code path where a scripting failure hides the
 * page. The tradeoff is that entrances play on load rather than on scroll,
 * which on pages this length is barely distinguishable and worth the safety.
 */
export function Reveal({
  children, delay = 0, className,
}: {
  children: ReactNode; delay?: number; className?: string;
}) {
  return (
    <div
      className={clsx("reveal", className)}
      style={delay ? { animationDelay: `${delay}ms` } : undefined}
    >
      {children}
    </div>
  );
}

/**
 * Count a number up when it first becomes visible, and re-run smoothly
 * whenever the value changes afterwards.
 *
 * Eased rather than linear, and short, because a slow counter on a dashboard reads as
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
 * by hand. The stagger is capped, because past about half a second the delay stops
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

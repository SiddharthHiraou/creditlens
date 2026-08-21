import clsx from "clsx";
import type { ReactNode } from "react";

export function Card({
  children, className, title, subtitle, action,
}: {
  children: ReactNode; className?: string; title?: string;
  subtitle?: string; action?: ReactNode;
}) {
  return (
    <section
      className={clsx(
        "rounded-xl border p-5",
        "border-[var(--border)] bg-[var(--surface-raised)]",
        className,
      )}
    >
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-sm font-semibold tracking-tight">{title}</h2>}
            {subtitle && (
              <p className="mt-1 text-xs text-[var(--text-muted)]">{subtitle}</p>
            )}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Metric({
  label, value, hint, tone = "neutral", compare,
}: {
  label: string; value: string; hint?: string;
  tone?: "neutral" | "good" | "warn" | "bad"; compare?: string;
}) {
  const toneClass = {
    neutral: "text-[var(--text)]",
    good: "text-[var(--color-approve)]",
    warn: "text-[var(--color-refer)]",
    bad: "text-[var(--color-decline)]",
  }[tone];

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <p className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
        {label}
      </p>
      <p className={clsx("mt-1.5 text-2xl font-semibold tabular-nums", toneClass)}>{value}</p>
      {compare && (
        <p className="mt-0.5 text-xs tabular-nums text-[var(--text-muted)]">{compare}</p>
      )}
      {hint && <p className="mt-1.5 text-xs leading-snug text-[var(--text-muted)]">{hint}</p>}
    </div>
  );
}

export function Badge({
  children, tone = "neutral",
}: {
  children: ReactNode; tone?: "neutral" | "good" | "warn" | "bad" | "accent";
}) {
  const tones = {
    neutral: "bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)]",
    good: "bg-[var(--color-approve)]/12 text-[var(--color-approve)] border-[var(--color-approve)]/30",
    warn: "bg-[var(--color-refer)]/12 text-[var(--color-refer)] border-[var(--color-refer)]/30",
    bad: "bg-[var(--color-decline)]/12 text-[var(--color-decline)] border-[var(--color-decline)]/30",
    accent: "bg-[var(--color-accent)]/12 text-[var(--color-accent)] border-[var(--color-accent)]/30",
  }[tone];
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-[11px] font-medium",
        tones,
      )}
    >
      {children}
    </span>
  );
}

export function Table({
  head, children, dense = false,
}: {
  head: string[]; children: ReactNode; dense?: boolean;
}) {
  return (
    // Wide tables scroll inside their own container so the page body never does.
    <div className="-mx-1 overflow-x-auto px-1">
      <table className="w-full min-w-[520px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-[var(--border)]">
            {head.map((h, i) => (
              <th
                key={h}
                className={clsx(
                  "pb-2 text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]",
                  i === 0 ? "text-left" : "text-right",
                )}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className={dense ? "[&_td]:py-1" : "[&_td]:py-2"}>{children}</tbody>
      </table>
    </div>
  );
}

export function Row({ children }: { children: ReactNode }) {
  return (
    <tr className="border-b border-[var(--border)]/60 last:border-0">{children}</tr>
  );
}

export function Cell({
  children, align = "right", mono = true, className,
}: {
  children: ReactNode; align?: "left" | "right";
  mono?: boolean; className?: string;
}) {
  return (
    <td
      className={clsx(
        align === "left" ? "text-left" : "text-right",
        mono && align === "right" && "tabular-nums",
        className,
      )}
    >
      {children}
    </td>
  );
}

export function Note({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "warn" }) {
  return (
    <p
      className={clsx(
        "rounded-lg border-l-2 py-2 pl-3 text-xs leading-relaxed",
        tone === "warn"
          ? "border-[var(--color-refer)] text-[var(--text-muted)]"
          : "border-[var(--border)] text-[var(--text-muted)]",
      )}
    >
      {children}
    </p>
  );
}

export function WhyItMatters({
  question, children,
}: {
  question: string; children: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-[var(--color-accent)]/25 bg-[var(--color-accent)]/[0.06] p-5">
      <p className="text-[11px] font-medium uppercase tracking-wider text-[var(--color-accent)]">
        The question this page answers
      </p>
      <p className="mt-1.5 text-base font-medium leading-snug">{question}</p>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[var(--text-muted)]">
        {children}
      </p>
    </div>
  );
}

export function Jargon({ term, children }: { term: string; children: ReactNode }) {
  return (
    <span className="inline-flex items-baseline gap-1">
      <span>{term}</span>
      <span
        title={typeof children === "string" ? children : undefined}
        className="cursor-help border-b border-dotted border-[var(--text-muted)] text-[var(--text-muted)]"
      >
        ⓘ
      </span>
    </span>
  );
}

export function PageHeader({
  title, lede, eyebrow,
}: {
  title: string; lede: string; eyebrow?: string;
}) {
  return (
    <header className="mb-8">
      {eyebrow && (
        <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-[var(--color-accent)]">
          {eyebrow}
        </p>
      )}
      <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h1>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[var(--text-muted)]">{lede}</p>
    </header>
  );
}

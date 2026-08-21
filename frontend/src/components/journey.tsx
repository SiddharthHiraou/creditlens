import Link from "next/link";

/**
 * The site is seven dense pages. Without a stated path a visitor lands
 * somewhere in the middle, sees a wall of metrics, and leaves. This gives every
 * page a position in a sequence and an obvious next step.
 */

export const JOURNEY = [
  {
    href: "/",
    label: "Overview",
    question: "What does this thing do?",
    blurb: "The trade a lender makes, and how well the model separates good from bad.",
  },
  {
    href: "/score",
    label: "Underwriting",
    question: "Can I decline someone lawfully?",
    blurb: "One applicant: the decision, and the four reasons the law requires.",
  },
  {
    href: "/portfolio",
    label: "Portfolio",
    question: "How many should we approve?",
    blurb: "Drag the cutoff and watch defaults, losses and profit move.",
  },
  {
    href: "/monitoring",
    label: "Monitoring",
    question: "Is the model still right?",
    blurb: "Whether today's applicants still look like the training data.",
  },
  {
    href: "/fairness",
    label: "Fair lending",
    question: "Are we discriminating?",
    blurb: "Approval rates by protected group. This one fails on age.",
  },
  {
    href: "/model-card",
    label: "Governance",
    question: "Can we defend it in two years?",
    blurb: "Purpose, performance, limitations: the document a regulator asks for.",
  },
  {
    href: "/copilot",
    label: "AI assistance",
    question: "Where does AI help safely?",
    blurb: "Drafting decline letters and answering questions about the book.",
  },
] as const;

/** "Step 3 of 7" plus a link onward. Rendered at the foot of every page. */
export function NextStep({ current }: { current: string }) {
  const i = JOURNEY.findIndex((s) => s.href === current);
  if (i === -1) return null;
  const next = JOURNEY[(i + 1) % JOURNEY.length];
  const isWrap = i === JOURNEY.length - 1;

  return (
    <div className="mt-4">
      <div className="rule mb-6" />
      <div className="flex flex-wrap items-end justify-between gap-4">
        <p className="text-xs text-[var(--text-muted)]">
          Step {i + 1} of {JOURNEY.length}
        </p>
        <Link
          href={next.href}
          className="group flex max-w-md items-center gap-4 rounded-xl border border-[var(--border)] bg-[var(--surface-raised)] px-4 py-3 transition-colors hover:border-[var(--color-accent)]/50"
        >
          <div className="min-w-0">
            <p className="text-[11px] font-medium uppercase tracking-wider text-[var(--color-accent)]">
              {isWrap ? "Back to the start" : "Next"} · {next.label}
            </p>
            <p className="mt-0.5 text-sm font-medium">{next.question}</p>
          </div>
          <span className="text-lg text-[var(--text-muted)] transition-transform group-hover:translate-x-1 group-hover:text-[var(--color-accent)]">
            →
          </span>
        </Link>
      </div>
    </div>
  );
}

/** The full path, shown once on the overview so the shape of the site is visible. */
export function JourneyMap() {
  return (
    <section className="edge-lit rounded-2xl border border-[var(--border)] bg-[var(--surface-raised)] p-6 shadow-[var(--card-shadow)]">
      <h2 className="text-sm font-semibold tracking-tight">Start here</h2>
      <p className="mt-1 text-xs text-[var(--text-muted)]">
        Seven pages, one question each. In order, they walk from &ldquo;does it
        work&rdquo; to &ldquo;can we defend it&rdquo;.
      </p>

      <ol className="mt-5 grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
        {JOURNEY.slice(1).map((step, i) => (
          <li key={step.href}>
            <Link
              href={step.href}
              className="card-hover group flex h-full gap-3 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4"
            >
              <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-[var(--color-accent)]/12 text-[11px] font-semibold text-[var(--color-accent)]">
                {i + 1}
              </span>
              <div className="min-w-0">
                <p className="text-sm font-medium leading-snug group-hover:text-[var(--color-accent)]">
                  {step.question}
                </p>
                <p className="mt-1 text-xs leading-relaxed text-[var(--text-muted)]">
                  {step.blurb}
                </p>
              </div>
            </Link>
          </li>
        ))}
      </ol>
    </section>
  );
}

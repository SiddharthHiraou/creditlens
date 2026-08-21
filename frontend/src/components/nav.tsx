"use client";

import clsx from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { ThemeToggle } from "./theme-toggle";

// Labelled by the job each page does for a lending team, not by the technique
// behind it. "PSI monitoring" means nothing to a credit officer; "is the model
// still right" is the question they actually have.
const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/score", label: "Underwriting" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/monitoring", label: "Monitoring" },
  { href: "/fairness", label: "Fair lending" },
  { href: "/model-card", label: "Governance" },
  { href: "/copilot", label: "AI assistance" },
];

export function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const active = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <nav className="sticky top-0 z-30 border-b border-[var(--border)] bg-[var(--surface)]/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6">
        <Link href="/" className="flex shrink-0 items-center gap-2">
          <span className="accent-ring grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent-2)] text-[11px] font-bold text-white">
            CL
          </span>
          <span className="font-display text-[15px] font-semibold tracking-tight">CreditLens</span>
        </Link>

        <div className="hidden flex-1 items-center gap-1 md:flex">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={clsx(
                "nav-pill rounded-lg px-3 py-1.5 text-[13px]",
                active(link.href)
                  ? "bg-[var(--color-accent)]/12 font-medium text-[var(--color-accent)]"
                  : "text-[var(--text-muted)] hover:bg-[var(--surface-raised)] hover:text-[var(--text)]",
              )}
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <ThemeToggle />
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-label="Toggle navigation"
            className="rounded-md border border-[var(--border)] px-2 py-1 text-xs md:hidden"
          >
            Menu
          </button>
        </div>
      </div>

      {open && (
        <div className="grid grid-cols-2 gap-1 border-t border-[var(--border)] px-4 py-2 md:hidden">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setOpen(false)}
              className={clsx(
                "rounded-md px-2.5 py-2 text-[13px]",
                active(link.href)
                  ? "bg-[var(--surface-raised)] font-medium"
                  : "text-[var(--text-muted)]",
              )}
            >
              {link.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}

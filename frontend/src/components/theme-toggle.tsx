"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // The server does not know the user's theme, so rendering the icon before
  // mount produces a hydration mismatch. A fixed-size placeholder holds the
  // layout instead of shifting it.
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-7 w-7" aria-hidden />;

  const next = resolvedTheme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      aria-label={`Switch to ${next} theme`}
      className="grid h-7 w-7 place-items-center rounded-md border border-[var(--border)] text-[var(--text-muted)] transition-colors hover:text-[var(--text)]"
    >
      {resolvedTheme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
    </button>
  );
}

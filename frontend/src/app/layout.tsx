import type { Metadata } from "next";
import { IBM_Plex_Mono, Sora, Inter } from "next/font/google";
import type { ReactNode } from "react";

import { Nav } from "@/components/nav";
import { Providers } from "@/components/providers";

import "./globals.css";

// Sora for headings — geometric, a little editorial, distinct from the
// system-font look every dashboard defaults to. Inter for body, Plex Mono for
// figures and feature names.
const display = Sora({ subsets: ["latin"], weight: ["600", "700"], variable: "--font-display" });
const sans = Inter({ subsets: ["latin"], variable: "--font-sans" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "CreditLens — automated lending decisions a bank could defend",
  description:
    "Decides consumer loan applications, explains every decline in plain English, proves it is not discriminating, and flags when it has gone stale.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${display.variable} ${sans.variable} ${mono.variable}`}
    >
      <body className="min-h-screen font-[family-name:var(--font-sans)] antialiased">
        <Providers>
          <Nav />
          <main className="mx-auto max-w-6xl px-5 py-10 sm:px-8 sm:py-14">{children}</main>
          <footer className="mx-auto max-w-6xl px-5 pb-14 sm:px-8">
            <div className="rule mb-5" />
            <p className="text-xs leading-relaxed text-[var(--text-muted)]">
              A portfolio project by Siddharth Hiraou. Every figure is generated from the
              model artifacts in the repository and reproducible with{" "}
              <code className="font-mono text-[11px]">make train</code>. Trained on
              synthetic data — not a real lender, and no real applicants.
            </p>
          </footer>
        </Providers>
      </body>
    </html>
  );
}

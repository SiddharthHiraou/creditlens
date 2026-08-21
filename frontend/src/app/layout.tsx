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
          <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 sm:py-10">{children}</main>
          <footer className="mx-auto max-w-7xl px-4 pb-10 text-xs text-[var(--text-muted)] sm:px-6">
            <p>
              Demo data generated from the project&apos;s own model artifacts. Not a
              real lender; no real applicants.
            </p>
          </footer>
        </Providers>
      </body>
    </html>
  );
}

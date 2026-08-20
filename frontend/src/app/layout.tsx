import type { Metadata } from "next";
import type { ReactNode } from "react";

import { Nav } from "@/components/nav";
import { Providers } from "@/components/providers";

import "./globals.css";

export const metadata: Metadata = {
  title: "CreditLens — credit decisioning and model risk",
  description:
    "Probability of default, ECOA reason codes, fairness measurement and drift monitoring for consumer lending.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen antialiased">
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

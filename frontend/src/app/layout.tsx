import type { Metadata } from "next";
import type { ReactNode } from "react";

import { Nav } from "@/components/nav";
import { Providers } from "@/components/providers";

import "./globals.css";

export const metadata: Metadata = {
  title: "CreditLens — automated lending decisions a bank could defend",
  description:
    "Decides consumer loan applications, explains every decline in plain English, proves it is not discriminating, and flags when it has gone stale.",
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

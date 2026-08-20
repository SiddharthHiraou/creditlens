export const pct = (v: number, digits = 1) =>
  `${(v * 100).toFixed(digits)}%`;

export const num = (v: number, digits = 4) => v.toFixed(digits);

export const compactMoney = (v: number) => {
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(v / 1e3).toFixed(0)}k`;
  return v.toFixed(0);
};

export const money = (v: number) =>
  new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(v);

export const count = (v: number) => new Intl.NumberFormat("en-US").format(v);

export const decisionColor = (decision: string) =>
  decision === "approve"
    ? "var(--color-approve)"
    : decision === "refer"
      ? "var(--color-refer)"
      : "var(--color-decline)";

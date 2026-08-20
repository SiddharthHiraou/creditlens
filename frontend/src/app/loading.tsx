export default function Loading() {
  return (
    <div className="space-y-4" role="status" aria-label="Loading">
      <div className="h-8 w-64 animate-pulse rounded bg-[var(--surface-raised)]" />
      <div className="h-4 w-96 max-w-full animate-pulse rounded bg-[var(--surface-raised)]" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-24 animate-pulse rounded-lg bg-[var(--surface-raised)]" />
        ))}
      </div>
      <div className="h-72 animate-pulse rounded-xl bg-[var(--surface-raised)]" />
    </div>
  );
}

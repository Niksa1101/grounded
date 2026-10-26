"use client";

import Link from "next/link";

// Shown only when a render of /metrics throws with no previous render to fall back on (a failed
// background regeneration keeps the old page instead, app/metrics/metrics-page.tsx).
export default function MetricsError() {
  return (
    <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-4 px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-semibold tracking-tight">Metrics</h1>
      <p>
        The metrics are not available right now. The free-tier backend may be asleep or slow; try
        again in a minute.
      </p>
      <p>
        <Link className="underline underline-offset-4" href="/">
          Back to asking a question
        </Link>
      </p>
    </main>
  );
}

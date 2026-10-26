import "server-only";

import { MetricsDashboard } from "@/components/metrics-dashboard";
import type { MetricsWindow } from "@/lib/api";
import { classifyMetrics, type MetricsResult } from "@/lib/metrics";
import { loadMetrics } from "@/lib/metrics-source";

// The server side of /metrics and /metrics/[window] (Tech §19). A Server Component: it reads the
// secret here, on the server, and the browser receives only HTML with the aggregates.

/** DEV ONLY: the `?fake=` name. Reading `searchParams` makes a page dynamic, so the read sits
 * behind the NODE_ENV constant: a production build never reads it and the pages stay static. */
export async function devFakeName(
  searchParams: Promise<Record<string, string | string[] | undefined>>,
): Promise<string | undefined> {
  if (process.env.NODE_ENV === "production") return undefined;
  const value = (await searchParams).fake;
  return typeof value === "string" ? value : undefined;
}

async function load(window: MetricsWindow, fake: string | undefined): Promise<MetricsResult> {
  // DEV ONLY, as in components/ask-form.tsx: the branch and the fixtures module are removed from a
  // production build because Next replaces process.env.NODE_ENV with a constant.
  if (process.env.NODE_ENV !== "production" && fake) {
    const { fakeMetrics } = await import("@/lib/metrics-dev-fixtures");
    const result = fakeMetrics(fake, window);
    if (result) return result;
  }
  return loadMetrics(window, {
    backendUrl: process.env.BACKEND_URL,
    sharedSecret: process.env.PROXY_SHARED_SECRET,
  });
}

/**
 * True while a production server regenerates the page in the background. Throwing then makes Next
 * keep serving the previous render (ISR: "the last successfully generated data will continue to
 * be served"), whose visible timestamp then shows how old it is. During `next build` (no previous
 * render exists, and a sleeping backend must not fail a deploy) and in development the page
 * renders its "unavailable" state instead.
 */
function keepPreviousRender(): boolean {
  return (
    process.env.NODE_ENV === "production" &&
    process.env.NEXT_PHASE !== "phase-production-build"
  );
}

export async function MetricsPage({
  window,
  fake,
}: {
  window: MetricsWindow;
  fake?: string;
}) {
  const result = await load(window, fake);
  if (!result.ok && keepPreviousRender()) {
    throw new Error(`metrics page: ${result.code}, keeping the previous render`);
  }
  return <MetricsDashboard window={window} view={classifyMetrics(result)} />;
}

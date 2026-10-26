import type { Metadata } from "next";

import { DEFAULT_WINDOW } from "@/lib/metrics";

import { devFakeName, MetricsPage } from "./metrics-page";

// ISR (Tech §14): a static page regenerated at most once every 60 s, in the background, when
// someone visits it. Visitors get the cached HTML from the CDN; the backend is not called per view.
export const revalidate = 60;

export const metadata: Metadata = {
  title: "Metrics · Grounded",
  description: "Public aggregates of the Grounded demo: latency, cost, rates and the latest evals.",
};

// The default window (7 days). /metrics/1d and /metrics/30d are the other two.
export default async function Page({ searchParams }: PageProps<"/metrics">) {
  return <MetricsPage window={DEFAULT_WINDOW} fake={await devFakeName(searchParams)} />;
}

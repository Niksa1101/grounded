import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { isWindow, WINDOWS } from "@/lib/metrics";

import { devFakeName, MetricsPage } from "../metrics-page";

// One static page per window, built at `next build` and regenerated like /metrics (60 s). A path
// segment, not `?window=`: a query string can only be read by rendering on every request, which
// would start a function per view instead of serving cached HTML.
export const revalidate = 60;
export const dynamicParams = false; // any other segment is a 404, never a backend call

export function generateStaticParams() {
  return WINDOWS.map((window) => ({ window }));
}

export const metadata: Metadata = {
  title: "Metrics · Grounded",
  description: "Public aggregates of the Grounded demo: latency, cost, rates and the latest evals.",
};

export default async function Page({ params, searchParams }: PageProps<"/metrics/[window]">) {
  const { window } = await params;
  if (!isWindow(window)) notFound();
  return <MetricsPage window={window} fake={await devFakeName(searchParams)} />;
}

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { MetricsWindow } from "@/lib/api";
import { classifyMetrics, NO_DATA } from "@/lib/metrics";
import { fakeMetrics } from "@/lib/metrics-dev-fixtures";

import { MetricsDashboard } from "./metrics-dashboard";

function render(name: string, window: MetricsWindow = "7d"): string {
  const view = classifyMetrics(fakeMetrics(name, window)!);
  return renderToStaticMarkup(<MetricsDashboard window={window} view={view} />);
}

describe("MetricsDashboard", () => {
  it("renders semantic tables: a caption and scoped header cells in each", () => {
    const html = render("full");
    const tables = html.match(/<table/g)?.length ?? 0;
    expect(tables).toBe(6);
    expect(html.match(/<caption/g)?.length).toBe(tables);
    expect(html).toContain('scope="col"');
    expect(html).toContain('scope="row"');
  });

  it("shows p50/p95 per stage with n, and no data for a stage that never ran", () => {
    const html = render("full");
    expect(html).toContain("3,120 ms");
    expect(html).toContain("7,040 ms");
    expect(html).toMatch(/Re-ranking<\/th><td[^>]*>no data<\/td><td[^>]*>no data<\/td><td[^>]*>0<\/td>/);
  });

  it("shows both per-1k costs, each with its explanation", () => {
    const html = render("full");
    expect(html).toContain("$0.328");
    expect(html).toContain("$0.434");
    expect(html).toContain("cache hits included");
    expect(html).toContain("comparable with the eval");
  });

  it("labels days as UTC and says the daily sums can differ from the window", () => {
    const html = render("full", "30d");
    expect(html).toContain("Day (UTC)");
    expect(html).toContain("the last 30 days");
    expect(html).toContain("can differ");
    expect(html).toContain("720 hours");
  });

  it("shows eval status as a word, not colour alone", () => {
    const html = render("full");
    expect(html).toContain("Pass");
    expect(html).toContain("Inconclusive");
    expect(html).toContain("faithfulness</span> 0.912");
  });

  it("marks the shown window in the window links", () => {
    const html = render("full", "30d");
    expect(html).toMatch(/<a aria-current="page"[^>]*href="\/metrics\/30d"/);
    expect(html.match(/aria-current/g)?.length).toBe(1);
    expect(html).toContain('href="/metrics/1d"');
    expect(html).toContain('href="/"');
  });

  it("empty: a notice, explicit no data, and no eval runs yet", () => {
    const html = render("empty");
    expect(html).toContain("No requests in this window yet");
    expect(html).toContain(NO_DATA);
    expect(html).not.toContain("NaN");
    expect(html).toContain("No eval runs recorded yet.");
  });

  it("unavailable: a friendly message and no tables", () => {
    const html = render("error");
    expect(html).toContain("The metrics are not available right now");
    expect(html).toContain("asleep or slow");
    expect(html).not.toContain("<table");
  });
});

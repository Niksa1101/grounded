import { describe, expect, it } from "vitest";

import { fakeMetrics } from "./metrics-dev-fixtures";
import {
  classifyMetrics,
  dataAge,
  formatMetric,
  formatMs,
  formatRate,
  formatUsd,
  formatUtc,
  isMetricsSummary,
  isWindow,
  NO_DATA,
  STALE_AFTER_S,
  WINDOWS,
} from "./metrics";
import openapi from "./openapi.json";

describe("windows", () => {
  it("are exactly the enum of the committed OpenAPI schema", () => {
    const schema = openapi.components.schemas.MetricsSummary.properties.window;
    expect([...WINDOWS]).toEqual(schema.enum);
  });

  it("accept only the enum values", () => {
    expect(isWindow("30d")).toBe(true);
    expect(isWindow("2d")).toBe(false);
    expect(isWindow("7D")).toBe(false);
    expect(isWindow(undefined)).toBe(false);
  });
});

describe("formatting", () => {
  it("says no data for null, never 0 or NaN", () => {
    for (const format of [formatRate, formatMs, formatUsd]) {
      expect(format(null)).toBe(NO_DATA);
      expect(format(Number.NaN)).toBe(NO_DATA);
    }
  });

  it("formats rates as percentages with one decimal", () => {
    expect(formatRate(0)).toBe("0.0%");
    expect(formatRate(9 / 37)).toBe("24.3%");
    expect(formatRate(1)).toBe("100.0%");
  });

  it("formats milliseconds as whole numbers with separators", () => {
    expect(formatMs(95)).toBe("95 ms");
    expect(formatMs(12_345)).toBe("12,345 ms");
  });

  it("keeps three significant digits of a sub-dollar cost instead of rounding it away", () => {
    expect(formatUsd(0)).toBe("$0");
    expect(formatUsd(0.000357)).toBe("$0.000357");
    expect(formatUsd(0.32811)).toBe("$0.328");
    expect(formatUsd(0.00000001)).toBe("$0.00000001");
    expect(formatUsd(12.345)).toBe("$12.35");
  });

  it("shows timestamps in UTC to the minute", () => {
    expect(formatUtc("2026-10-26T14:03:59.123456+00:00")).toBe("2026-10-26 14:03 UTC");
    expect(formatUtc("2026-10-26T01:00:00+02:00")).toBe("2026-10-25 23:00 UTC");
  });

  it("formats eval metrics: integers as they are, fractions to three decimals", () => {
    expect(formatMetric(30)).toBe("30");
    expect(formatMetric(0.91234)).toBe("0.912");
  });
});

describe("classifyMetrics", () => {
  it("is unavailable with the failure code", () => {
    expect(classifyMetrics({ ok: false, code: "upstream_timeout" })).toEqual({
      kind: "unavailable",
      code: "upstream_timeout",
    });
  });

  it("is empty when the window has no request", () => {
    const view = classifyMetrics(fakeMetrics("empty", "7d")!);
    expect(view.kind === "ready" && view.empty).toBe(true);
  });

  it("is ready and not empty with data", () => {
    const view = classifyMetrics(fakeMetrics("full", "30d")!);
    expect(view.kind === "ready" && !view.empty && view.data.daily.length).toBe(30);
  });
});

describe("dataAge", () => {
  const generated = "2026-10-26T12:00:00Z";
  const at = (seconds: number) => Date.parse(generated) + seconds * 1000;

  it("is current up to the stale threshold", () => {
    expect(dataAge(generated, at(30))).toEqual({ stale: false, text: "about a minute ago" });
    expect(dataAge(generated, at(STALE_AFTER_S))?.stale).toBe(false);
  });

  it("is stale past it, in words", () => {
    expect(dataAge(generated, at(STALE_AFTER_S + 1))).toEqual({ stale: true, text: "10 minutes ago" });
    expect(dataAge(generated, at(3 * 3600))?.text).toBe("3 hours ago");
    expect(dataAge(generated, at(3 * 86_400))?.text).toBe("3 days ago");
  });

  it("treats a clock behind the server as no age, and a bad timestamp as unknown", () => {
    expect(dataAge(generated, at(-120))?.stale).toBe(false);
    expect(dataAge("not a date", at(0))).toBeNull();
  });
});

describe("isMetricsSummary", () => {
  it("accepts the fixture and rejects other JSON", () => {
    const result = fakeMetrics("full", "1d")!;
    expect(result.ok && isMetricsSummary(result.data)).toBe(true);
    expect(isMetricsSummary(null)).toBe(false);
    expect(isMetricsSummary({ error: { code: "internal_error" } })).toBe(false);
    expect(isMetricsSummary({ ...(result.ok ? result.data : {}), window: "2d" })).toBe(false);
  });
});

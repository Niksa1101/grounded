// Pure helpers of the /metrics page (Tech §19, ticket 8.03): the window rules, number formatting
// with an explicit "no data" for null, and the classification of one load into what the page
// shows. No React, no fetch, no process.env: everything here is unit-tested in Node.
import type { MetricsSummary, MetricsWindow } from "@/lib/api";

// The windows of GET /v1/metrics/summary (Tech §13). `satisfies` keeps the list in step with the
// generated type: a window the backend drops breaks `npm run typecheck`.
export const WINDOWS = ["1d", "7d", "30d"] as const satisfies readonly MetricsWindow[];
export const DEFAULT_WINDOW: MetricsWindow = "7d";

export const WINDOW_DAYS: Record<MetricsWindow, number> = { "1d": 1, "7d": 7, "30d": 30 };

export function isWindow(value: unknown): value is MetricsWindow {
  return typeof value === "string" && (WINDOWS as readonly string[]).includes(value);
}

/** Shown wherever a value is null: "no data" is not 0, and it is never NaN. */
export const NO_DATA = "no data";

/** A ratio in [0, 1] as a percentage with one decimal. */
export function formatRate(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return NO_DATA;
  return `${(value * 100).toFixed(1)}%`;
}

/** Whole milliseconds with a thousands separator. */
export function formatMs(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return NO_DATA;
  return `${Math.round(value).toLocaleString("en-US")} ms`;
}

export function formatCount(value: number): string {
  return value.toLocaleString("en-US");
}

/**
 * USD. Shadow costs are tiny (a request costs fractions of a cent), so below one dollar the value
 * keeps three significant digits instead of rounding to $0.00; the backend's 8 decimals are the
 * floor. Null (a zero denominator) is "no data", never $0.
 */
export function formatUsd(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return NO_DATA;
  if (value === 0) return "$0";
  if (Math.abs(value) >= 1) return `$${value.toFixed(2)}`;
  const digits = Number(value.toPrecision(3)).toFixed(8).replace(/0+$/, "").replace(/\.$/, "");
  return `$${digits}`;
}

/** An ISO timestamp as "2026-10-26 14:03 UTC" (the page is static, so no local time zone). */
export function formatUtc(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

/** An eval metric: integers as they are, fractions with three decimals. */
export function formatMetric(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

/** Why a load failed. The same codes as the proxy's own errors (lib/api.ts). */
export type MetricsFailure = "upstream_timeout" | "upstream_unavailable" | "proxy_misconfigured";

export type MetricsResult =
  | { ok: true; data: MetricsSummary }
  | { ok: false; code: MetricsFailure };

export type MetricsView =
  | { kind: "unavailable"; code: MetricsFailure }
  | { kind: "ready"; data: MetricsSummary; empty: boolean };

/**
 * What the page shows. `empty` is a window without any user request: the tables still render
 * (every rate and percentile then says "no data"), with a notice above them.
 */
export function classifyMetrics(result: MetricsResult): MetricsView {
  if (!result.ok) return { kind: "unavailable", code: result.code };
  return { kind: "ready", data: result.data, empty: result.data.requests.total === 0 };
}

// The page is regenerated at most once a minute, and a failed regeneration keeps the previous
// render (app/metrics/metrics-page.tsx). Numbers older than this mean the backend has not answered
// for a while, so the page says so instead of passing them off as current.
export const STALE_AFTER_S = 10 * 60;

export type DataAge = { stale: boolean; text: string };

/** How old the numbers are, judged in the browser at view time (the HTML may be minutes old). */
export function dataAge(generatedAtIso: string, nowMs: number): DataAge | null {
  const generated = new Date(generatedAtIso).getTime();
  if (Number.isNaN(generated)) return null;
  const seconds = Math.max(0, Math.round((nowMs - generated) / 1000));
  return { stale: seconds > STALE_AFTER_S, text: describeAge(seconds) };
}

function describeAge(seconds: number): string {
  if (seconds < 90) return "about a minute ago";
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `${minutes} minutes ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 36) return `${hours} hours ago`;
  return `${Math.round(hours / 24)} days ago`;
}

/** A minimal shape check of the backend's JSON before the page trusts it (the backend validates
 * with Pydantic; this only guards against a platform page or a different API answering). */
export function isMetricsSummary(body: unknown): body is MetricsSummary {
  if (typeof body !== "object" || body === null) return false;
  const value = body as Record<string, unknown>;
  return (
    isWindow(value.window) &&
    typeof value.generated_at === "string" &&
    ["requests", "rates", "latency_ms", "cost"].every(
      (key) => typeof value[key] === "object" && value[key] !== null,
    ) &&
    Array.isArray(value.daily) &&
    Array.isArray(value.evals)
  );
}

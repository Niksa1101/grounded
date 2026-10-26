// DEV ONLY. Canned results for the /metrics page, so each of its states can be seen without a
// backend: open `/metrics?fake=<name>` (or `/metrics/30d?fake=<name>`) under `npm run dev`.
//
// It cannot ship: the only importer (app/metrics/metrics-page.tsx) loads it with `import()` behind
// `process.env.NODE_ENV !== "production"`, a constant at build time, so the module is dropped from
// the production bundle. 8.03 checks that by grepping the build output for METRICS_FIXTURE_MARKER.
import type { DailyStats, MetricsSummary, MetricsWindow } from "@/lib/api";
import { WINDOW_DAYS, type MetricsResult } from "@/lib/metrics";

export const METRICS_FIXTURE_MARKER = "metrics-fixtures-only-5d81";

const DAY_MS = 86_400_000;

function days(window: MetricsWindow, now: number, fill: (i: number) => Omit<DailyStats, "day">): DailyStats[] {
  const count = WINDOW_DAYS[window];
  return Array.from({ length: count }, (_, i) => ({
    day: new Date(now - (count - 1 - i) * DAY_MS).toISOString().slice(0, 10),
    ...fill(i),
  }));
}

function full(window: MetricsWindow, now: number): MetricsSummary {
  return {
    window,
    generated_at: new Date(now).toISOString(),
    requests: { total: 42, answered: 30, partial: 4, insufficient_context: 3, throttled: 3, errors: 2, cache_hits: 9 },
    rates: { cache_hit: 9 / 37, fallback: 2 / 27, validation_retry: 3 / 27, validation_failure: 1 / 27 },
    latency_ms: {
      total: { p50: 3120, p95: 7040, n: 30 },
      embed: { p50: 210, p95: 480, n: 30 },
      retrieval: { p50: 95, p95: 440, n: 30 },
      rerank: { p50: null, p95: null, n: 0 },
      llm: { p50: 2740, p95: 6100, n: 28 },
    },
    cost: { shadow_usd_total: 0.01214, per_1k_questions_usd: 0.32811, per_1k_uncached_usd: 0.43357 },
    daily: days(window, now, (i) =>
      i % 3 === 1
        ? { requests: 0, p50_ms: null, p95_ms: null, shadow_usd: 0 }
        : { requests: 4 + i, p50_ms: 2900 + 40 * i, p95_ms: 6800 + 30 * i, shadow_usd: 0.00121 + i / 1e5 },
    ),
    evals: [
      {
        suite: "generation",
        config: "fixture-hybrid",
        created_at: new Date(now - 2 * DAY_MS).toISOString(),
        git_sha: "5d81c0ffee5d81c0ffee5d81c0ffee5d81c0ffee",
        status: "pass",
        case_count: 30,
        metrics: { faithfulness: 0.912, answer_relevance: 0.874, citation_validity: 1, cost_per_1k_usd: 0.4123 },
        report_url: "https://github.com/example/grounded/actions/runs/1",
      },
      {
        suite: "retrieval",
        config: "fixture-hybrid",
        created_at: new Date(now - 3 * DAY_MS).toISOString(),
        git_sha: "0ddba11c0de0ddba11c0de0ddba11c0de0ddba11",
        status: "inconclusive",
        case_count: 30,
        metrics: { "recall@5": 0.833, mrr: 0.71, "ndcg@5": 0.742 },
        report_url: null,
      },
    ],
  };
}

function empty(window: MetricsWindow, now: number): MetricsSummary {
  const none = { p50: null, p95: null, n: 0 };
  return {
    window,
    generated_at: new Date(now).toISOString(),
    requests: { total: 0, answered: 0, partial: 0, insufficient_context: 0, throttled: 0, errors: 0, cache_hits: 0 },
    rates: { cache_hit: null, fallback: null, validation_retry: null, validation_failure: null },
    latency_ms: { total: none, embed: none, retrieval: none, rerank: none, llm: none },
    cost: { shadow_usd_total: 0, per_1k_questions_usd: null, per_1k_uncached_usd: null },
    daily: days(window, now, () => ({ requests: 0, p50_ms: null, p95_ms: null, shadow_usd: 0 })),
    evals: [],
  };
}

const FIXTURES: Record<string, (window: MetricsWindow, now: number) => MetricsResult> = {
  full: (w, now) => ({ ok: true, data: full(w, now) }),
  empty: (w, now) => ({ ok: true, data: empty(w, now) }),
  // A previous render kept while the backend fails: its numbers are three hours old.
  stale: (w, now) => ({ ok: true, data: full(w, now - 3 * 3_600_000) }),
  error: () => ({ ok: false, code: "upstream_timeout" }),
  misconfigured: () => ({ ok: false, code: "proxy_misconfigured" }),
};

/** The canned result for `?fake=<name>`, or null for an unknown name (the real load proceeds). */
export function fakeMetrics(name: string, window: MetricsWindow): MetricsResult | null {
  const fixture = FIXTURES[name];
  return fixture ? fixture(window, Date.now()) : null;
}

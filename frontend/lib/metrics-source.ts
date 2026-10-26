// Server side of the dashboard: one function that reads GET /v1/metrics/summary with the proxy
// secret (used by the /metrics page and by GET /api/metrics), and the JSON the Route Handler
// answers with. Like lib/ask-proxy.ts it never reads process.env (the callers pass the config), so
// a unit test drives it without a Next server. `server-only` makes a client import a build error.
import "server-only";

import type { MetricsWindow, ProxyErrorCode, ProxyErrorResponse } from "./api";
import {
  DEFAULT_WINDOW,
  isMetricsSummary,
  isWindow,
  type MetricsFailure,
  type MetricsResult,
} from "./metrics";

// Tech §14: the dashboard is revalidated every 60 s. Next's Data Cache keys a fetch by its URL and
// init, so the page and the Route Handler share one cached entry per window: while the backend
// answers, it (and the Neon database behind it) is asked at most about once a minute per window,
// whatever the traffic. The Data Cache stores only 200s, so it does not bound the failure case;
// FAILURE_BACKOFF_MS below does.
export const METRICS_REVALIDATE_S = 60;

// After a failed call, this server instance answers the same failure for this long without asking
// the backend again: during an outage every request would otherwise be a backend call (and a
// possible Neon wake). Per instance and per window; a new instance asks once.
export const FAILURE_BACKOFF_MS = 60_000;
const recentFailures = new Map<MetricsWindow, { until: number; code: MetricsFailure }>();

/** Tests only: forget the failures of earlier calls. */
export function resetFailureBackoff(): void {
  recentFailures.clear();
}

// Long enough for a cold backend (function start about 2.6 s plus the Neon wake, Tech §18), short
// enough that a page regeneration or a visitor does not hang on a stuck one. The route's
// `maxDuration` (app/api/metrics/route.ts) is longer.
export const METRICS_TIMEOUT_MS = 15_000;

export interface MetricsConfig {
  backendUrl: string | undefined;
  sharedSecret: string | undefined;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
  now?: () => number;
}

/**
 * The race is the timeout, not only the abort signal: when Next revalidates a stale Data Cache
 * entry while it regenerates a page, it drops the caller's signal from the origin fetch
 * (next/dist/server/lib/patch-fetch.js, "don't pass through signal when revalidating"), so an
 * abort alone would not bound that case.
 */
async function withTimeout<T>(ms: number, run: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      const error = new DOMException("metrics fetch timed out", "TimeoutError");
      controller.abort(error);
      reject(error);
    }, ms);
  });
  try {
    return await Promise.race([run(controller.signal), timeout]);
  } finally {
    clearTimeout(timer);
  }
}

/** One summary, or why there is none. Never throws; the backend's error text is never passed on. */
export async function loadMetrics(
  window: MetricsWindow,
  config: MetricsConfig,
): Promise<MetricsResult> {
  const { backendUrl, sharedSecret } = config;
  if (!backendUrl || !sharedSecret) {
    console.error("metrics: BACKEND_URL and PROXY_SHARED_SECRET must both be set");
    return { ok: false, code: "proxy_misconfigured" };
  }
  const now = (config.now ?? Date.now)();
  const held = recentFailures.get(window);
  if (held && now < held.until) return { ok: false, code: held.code };
  const result = await fetchSummary(window, backendUrl, sharedSecret, config);
  if (result.ok) recentFailures.delete(window);
  else recentFailures.set(window, { until: now + FAILURE_BACKOFF_MS, code: result.code });
  return result;
}

async function fetchSummary(
  window: MetricsWindow,
  backendUrl: string,
  sharedSecret: string,
  config: MetricsConfig,
): Promise<MetricsResult> {
  const url = `${backendUrl.replace(/\/+$/, "")}/v1/metrics/summary?window=${window}`;
  const doFetch = config.fetchImpl ?? fetch;
  try {
    const { status, body } = await withTimeout(
      config.timeoutMs ?? METRICS_TIMEOUT_MS,
      async (signal) => {
        // Only the secret: no client address (the backend's metrics route reads none, and one
        // cached answer serves every visitor), no cookies, nothing of the visitor's.
        const response = await doFetch(url, {
          headers: { accept: "application/json", "x-proxy-secret": sharedSecret },
          signal,
          next: { revalidate: METRICS_REVALIDATE_S },
        });
        return { status: response.status, body: (await response.json().catch(() => null)) as unknown };
      },
    );
    if (status === 200 && isMetricsSummary(body)) return { ok: true, data: body };
    console.error("metrics: unexpected backend response, status", status);
    // 401 means our secret is wrong: a configuration fault, not the backend's.
    if (status === 401) return { ok: false, code: "proxy_misconfigured" };
    return { ok: false, code: status === 504 ? "upstream_timeout" : "upstream_unavailable" };
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") {
      console.error("metrics: backend timed out");
      return { ok: false, code: "upstream_timeout" };
    }
    console.error("metrics: backend unreachable", (error as Error).name);
    return { ok: false, code: "upstream_unavailable" };
  }
}

const FAILURES: Record<Exclude<ProxyErrorCode, "bad_request">, [number, string]> = {
  upstream_timeout: [504, "The backend took too long to answer. It may be waking up: try again."],
  upstream_unavailable: [502, "The metrics could not be loaded."],
  proxy_misconfigured: [500, "The server is not configured."],
};

function failure(status: number, code: ProxyErrorCode, message: string): Response {
  const body: ProxyErrorResponse = { error: { code, message }, request_id: null };
  return Response.json(body, { status, headers: { "cache-control": "no-store" } });
}

/**
 * The query string: `window` alone, at most once (absent is the default). Anything else is refused,
 * so only the three canonical URLs exist: a junk parameter would otherwise make a new CDN cache key,
 * and so a new function invocation, per request.
 */
export function parseWindowParam(params: URLSearchParams): MetricsWindow | null {
  const keys = [...params.keys()];
  if (keys.some((key) => key !== "window") || keys.length > 1) return null;
  const raw = params.get("window");
  if (raw === null) return DEFAULT_WINDOW;
  return isWindow(raw) ? raw : null;
}

/** GET /api/metrics?window=: validate, load (Data Cache, 60 s), answer JSON. */
export async function serveMetrics(request: Request, config: MetricsConfig): Promise<Response> {
  const window = parseWindowParam(new URL(request.url).searchParams);
  if (window === null) {
    // Nothing of the query is echoed (the backend's 422 rule, Tech §13).
    return failure(422, "bad_request", "The only query parameter is window: 1d, 7d or 30d.");
  }
  const result = await loadMetrics(window, config);
  if (!result.ok) {
    const [status, message] = FAILURES[result.code];
    return failure(status, result.code, message);
  }
  // Aggregates only, the same for every visitor: a shared cache (Vercel's CDN) may keep it for the
  // same minute as the Data Cache, so most requests never start the function.
  return Response.json(result.data, {
    headers: { "cache-control": `public, max-age=0, s-maxage=${METRICS_REVALIDATE_S}` },
  });
}

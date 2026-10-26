// Server side of the dashboard: one function that reads GET /v1/metrics/summary with the proxy
// secret (used by the /metrics page and by GET /api/metrics), and the JSON the Route Handler
// answers with. Like lib/ask-proxy.ts it never reads process.env (the callers pass the config), so
// a unit test drives it without a Next server, and no client component imports it.
import type { MetricsWindow, ProxyErrorCode, ProxyErrorResponse } from "./api";
import { DEFAULT_WINDOW, isMetricsSummary, isWindow, type MetricsResult } from "./metrics";

// Tech §14: the dashboard is revalidated every 60 s. Next's Data Cache keys a fetch by its URL and
// init, so the page and the Route Handler share one cached entry per window: the backend (and the
// Neon database behind it) is asked at most about once a minute per window, whatever the traffic.
export const METRICS_REVALIDATE_S = 60;

// Long enough for a cold backend (function start about 2.6 s plus the Neon wake, Tech §18), short
// enough that a page regeneration or a visitor does not hang on a stuck one. The route's
// `maxDuration` (app/api/metrics/route.ts) is longer.
export const METRICS_TIMEOUT_MS = 15_000;

export interface MetricsConfig {
  backendUrl: string | undefined;
  sharedSecret: string | undefined;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
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

/** The `window` query parameter: absent is the default, anything outside the enum is refused. */
export function parseWindowParam(raw: string | null): MetricsWindow | null {
  if (raw === null) return DEFAULT_WINDOW;
  return isWindow(raw) ? raw : null;
}

/** GET /api/metrics?window=: validate, load (Data Cache, 60 s), answer JSON. */
export async function serveMetrics(request: Request, config: MetricsConfig): Promise<Response> {
  const window = parseWindowParam(new URL(request.url).searchParams.get("window"));
  if (window === null) {
    // The value is not echoed (the backend's 422 rule, Tech §13).
    return failure(422, "bad_request", "window must be one of 1d, 7d or 30d.");
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

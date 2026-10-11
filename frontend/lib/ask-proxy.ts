// The logic behind POST /api/ask: forward the question to the backend with the proxy secret and the
// visitor's address, and turn every way the trip can fail into the error JSON the UI understands
// (Tech §12, §18, §19). It lives in lib/ so a unit test can drive it without a Next server; the
// Route Handler (app/api/ask/route.ts) is the only caller and the only place that reads
// process.env, so the secret never reaches a module that a client component imports.
import type { ProxyErrorCode, ProxyErrorResponse } from "./api";

// The timeout chain of Tech §18 (D53): the backend deadline (25 s) and its platform limit (30 s)
// are shorter than this, and the route's `maxDuration` (40 s) is longer, so the layer closest to the
// problem answers first. The rest of the 35 s is the budget for a backend cold start.
export const UPSTREAM_TIMEOUT_MS = 35_000;

// POST /v1/ask refuses a larger body (MAX_BODY_BYTES in schemas/api.py). Refusing here first keeps
// the proxy from buffering something the backend would reject anyway.
export const MAX_BODY_BYTES = 4096;

export interface ProxyConfig {
  backendUrl: string | undefined;
  sharedSecret: string | undefined;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

/**
 * The visitor's address for the backend's rate limiter. The backend wants ONE address in
 * `X-Client-IP` and treats anything else (a comma list included) as "unknown", which puts every
 * visitor into one shared bucket. Vercel *overwrites* `X-Forwarded-For` with the public address of
 * the client and does not forward addresses a caller supplied (Vercel docs, "Request headers",
 * read 2026-10-11), so its first entry is the client and cannot be forged by the visitor. Behind
 * a proxy that appends instead, the first entry would be spoofable: that is not this deployment.
 * `X-Real-IP` is documented as identical on Vercel and is the fallback.
 */
export function clientIp(headers: Headers): string | null {
  const first = headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  if (first) return first;
  const real = headers.get("x-real-ip")?.trim();
  return real || null;
}

function failure(
  status: number,
  code: ProxyErrorCode,
  message: string,
): Response {
  const body: ProxyErrorResponse = {
    error: { code, message },
    request_id: null,
  };
  return Response.json(body, {
    status,
    headers: { "cache-control": "no-store" },
  });
}

export async function forwardAsk(
  request: Request,
  config: ProxyConfig,
): Promise<Response> {
  const { backendUrl, sharedSecret } = config;
  if (!backendUrl || !sharedSecret) {
    // Say what is missing in the server log, never in the response or with a value.
    console.error("ask proxy: BACKEND_URL and PROXY_SHARED_SECRET must both be set");
    return failure(500, "proxy_misconfigured", "The server is not configured.");
  }

  const raw = await request.text();
  if (new TextEncoder().encode(raw).length > MAX_BODY_BYTES) {
    return failure(422, "bad_request", "The request is too large.");
  }
  let question: unknown;
  try {
    question = (JSON.parse(raw) as { question?: unknown } | null)?.question;
  } catch {
    question = undefined;
  }
  if (typeof question !== "string") {
    return failure(422, "bad_request", "The request needs a text question.");
  }

  // Only what the backend needs: no cookies, no user agent, nothing else of the visitor's.
  const headers: Record<string, string> = {
    "content-type": "application/json",
    accept: "application/json",
    "x-proxy-secret": sharedSecret,
  };
  const ip = clientIp(request.headers);
  if (ip) headers["x-client-ip"] = ip;

  const timeoutMs = config.timeoutMs ?? UPSTREAM_TIMEOUT_MS;
  const doFetch = config.fetchImpl ?? fetch;
  let status: number;
  let text: string;
  let retryAfter: string | null;
  try {
    // The visitor leaving (request.signal) frees the function early; the timeout bounds the rest.
    // The body is read under the same signal: a backend that sends headers and then stalls is
    // still a timeout.
    const response = await doFetch(`${backendUrl.replace(/\/+$/, "")}/v1/ask`, {
      method: "POST",
      headers,
      body: JSON.stringify({ question }),
      cache: "no-store",
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(timeoutMs)]),
    });
    status = response.status;
    retryAfter = response.headers.get("retry-after");
    text = await response.text();
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") {
      return failure(
        504,
        "upstream_timeout",
        "The backend took too long to answer. It may be waking up: try again.",
      );
    }
    console.error("ask proxy: backend unreachable", (error as Error).name);
    return failure(502, "upstream_unavailable", "The backend could not be reached.");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = undefined;
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    // Not our API's JSON: a platform page, typically the 504 of a function that hit its limit.
    console.error("ask proxy: non-JSON backend response, status", status);
    return status === 504
      ? failure(
          504,
          "upstream_timeout",
          "The backend took too long to answer. It may be waking up: try again.",
        )
      : failure(502, "upstream_unavailable", "The backend returned an unexpected response.");
  }

  const out: Record<string, string> = {
    "content-type": "application/json",
    "cache-control": "no-store",
  };
  if (retryAfter) out["retry-after"] = retryAfter;
  return new Response(text, { status, headers: out });
}

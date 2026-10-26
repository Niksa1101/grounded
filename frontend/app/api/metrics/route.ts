import { serveMetrics } from "@/lib/metrics-source";

// Longer than the fetch timeout (METRICS_TIMEOUT_MS, 15 s). A literal: Next reads it at build time.
export const maxDuration = 20;

// GET /api/metrics?window=1d|7d|30d: the dashboard's aggregates as JSON (Tech §19). Server-only:
// a Route Handler never ships to the browser. The backend fetch is in Next's Data Cache for 60 s
// (lib/metrics-source.ts), shared with the /metrics page, and the response may be kept by the CDN
// for the same minute.
export async function GET(request: Request): Promise<Response> {
  return serveMetrics(request, {
    backendUrl: process.env.BACKEND_URL,
    sharedSecret: process.env.PROXY_SHARED_SECRET,
  });
}

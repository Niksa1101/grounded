import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fakeMetrics } from "./metrics-dev-fixtures";
import {
  FAILURE_BACKOFF_MS,
  loadMetrics,
  METRICS_REVALIDATE_S,
  parseWindowParam,
  resetFailureBackoff,
  serveMetrics,
  type MetricsConfig,
} from "./metrics-source";

const SECRET = "test-secret-value";
const SUMMARY = (() => {
  const result = fakeMetrics("full", "7d")!;
  if (!result.ok) throw new Error("fixture");
  return result.data;
})();

function config(fetchImpl: unknown, extra: Partial<MetricsConfig> = {}): MetricsConfig {
  return { backendUrl: "http://backend.test/", sharedSecret: SECRET, fetchImpl: fetchImpl as typeof fetch, ...extra };
}

function get(query = ""): Request {
  return new Request(`http://localhost/api/metrics${query}`);
}

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
  resetFailureBackoff();
});
afterEach(() => {
  vi.restoreAllMocks();
});

describe("loadMetrics", () => {
  it("asks the backend with the secret only, cached for 60 s", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(Response.json(SUMMARY));
    expect(await loadMetrics("30d", config(fetchImpl))).toEqual({ ok: true, data: SUMMARY });

    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit & { next?: { revalidate?: number } }];
    expect(url).toBe("http://backend.test/v1/metrics/summary?window=30d");
    expect(init.headers).toEqual({ accept: "application/json", "x-proxy-secret": SECRET });
    expect(init.next).toEqual({ revalidate: METRICS_REVALIDATE_S });
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it("is misconfigured without a URL or a secret, and calls nothing", async () => {
    const fetchImpl = vi.fn();
    expect(await loadMetrics("7d", config(fetchImpl, { sharedSecret: "" }))).toEqual({
      ok: false,
      code: "proxy_misconfigured",
    });
    expect(await loadMetrics("7d", config(fetchImpl, { backendUrl: undefined }))).toMatchObject({ ok: false });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("maps a 401 to misconfigured and other failures to unavailable", async () => {
    const fresh = (cfg: MetricsConfig) => {
      resetFailureBackoff(); // each case on its own
      return loadMetrics("7d", cfg);
    };
    const answer = (status: number, body: unknown) => vi.fn().mockResolvedValue(Response.json(body, { status }));
    const err = { error: { code: "internal_error", message: "secret backend detail" }, request_id: "x" };
    expect(await fresh(config(answer(401, err)))).toEqual({ ok: false, code: "proxy_misconfigured" });
    expect(await fresh(config(answer(500, err)))).toEqual({ ok: false, code: "upstream_unavailable" });
    expect(await fresh(config(answer(504, err)))).toEqual({ ok: false, code: "upstream_timeout" });
    // A 200 that is not the summary (a platform page, another API) is not trusted.
    expect(await fresh(config(answer(200, { hello: 1 })))).toEqual({
      ok: false,
      code: "upstream_unavailable",
    });
  });

  it("is a timeout when the backend never answers, even if the abort signal is ignored", async () => {
    const fetchImpl = vi.fn(() => new Promise<Response>(() => {}));
    expect(await loadMetrics("7d", config(fetchImpl, { timeoutMs: 5 }))).toEqual({
      ok: false,
      code: "upstream_timeout",
    });
  });

  it("after a failure, answers it again without a backend call for the backoff, per window", async () => {
    let now = 1_000_000;
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError("fetch failed"));
    const cfg = config(fetchImpl, { now: () => now });

    expect(await loadMetrics("7d", cfg)).toEqual({ ok: false, code: "upstream_unavailable" });
    now += FAILURE_BACKOFF_MS - 1;
    expect(await loadMetrics("7d", cfg)).toEqual({ ok: false, code: "upstream_unavailable" });
    expect(fetchImpl).toHaveBeenCalledTimes(1);

    await loadMetrics("1d", cfg); // another window is asked
    expect(fetchImpl).toHaveBeenCalledTimes(2);

    now += 1; // the 7d backoff is over: ask again, and a success clears it
    fetchImpl.mockImplementation(async () => Response.json(SUMMARY));
    expect(await loadMetrics("7d", cfg)).toEqual({ ok: true, data: SUMMARY });
    expect(await loadMetrics("7d", cfg)).toEqual({ ok: true, data: SUMMARY });
    expect(fetchImpl).toHaveBeenCalledTimes(4);
  });

  it("is unavailable when the network fails", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError("fetch failed"));
    expect(await loadMetrics("7d", config(fetchImpl))).toEqual({ ok: false, code: "upstream_unavailable" });
  });
});

describe("serveMetrics", () => {
  it("defaults to 7d and answers the summary with a shared-cache header", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(Response.json(SUMMARY));
    const response = await serveMetrics(get(), config(fetchImpl));

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe("public, max-age=0, s-maxage=60");
    expect(await response.json()).toEqual(SUMMARY);
    expect(fetchImpl.mock.calls[0][0]).toMatch(/window=7d$/);
  });

  it("refuses a window outside the enum before any backend call, without echoing it", async () => {
    const fetchImpl = vi.fn();
    const response = await serveMetrics(get("?window=%3Cscript%3E"), config(fetchImpl));

    expect(response.status).toBe(422);
    expect(response.headers.get("cache-control")).toBe("no-store");
    const text = await response.text();
    expect(JSON.parse(text)).toEqual({
      error: { code: "bad_request", message: "The only query parameter is window: 1d, 7d or 30d." },
      request_id: null,
    });
    expect(text).not.toContain("script");
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("refuses any other query parameter, so only the canonical URLs reach the backend", async () => {
    const fetchImpl = vi.fn();
    for (const query of ["?window=1d&n=1", "?n=1", "?window=1d&window=7d", "?window=7d&junk"]) {
      const response = await serveMetrics(get(query), config(fetchImpl));
      expect(response.status).toBe(422);
      expect(await response.text()).not.toContain("n=1");
    }
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("answers its own error JSON, never the backend's text or the secret", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      Response.json({ error: { code: "internal_error", message: "secret backend detail" } }, { status: 500 }),
    );
    const response = await serveMetrics(get("?window=1d"), config(fetchImpl));

    expect(response.status).toBe(502);
    expect(response.headers.get("cache-control")).toBe("no-store");
    const text = await response.text();
    expect(JSON.parse(text)).toEqual({
      error: { code: "upstream_unavailable", message: "The metrics could not be loaded." },
      request_id: null,
    });
    expect(text).not.toContain("secret backend detail");
    expect(text).not.toContain(SECRET);
  });

  it("answers 500 proxy_misconfigured without saying what is missing", async () => {
    const response = await serveMetrics(get(), config(vi.fn(), { backendUrl: "" }));
    expect(response.status).toBe(500);
    expect((await response.json()).error.code).toBe("proxy_misconfigured");
  });
});

describe("parseWindowParam", () => {
  it("defaults when absent and refuses anything outside the enum", () => {
    const parse = (query: string) => parseWindowParam(new URLSearchParams(query));
    expect(parse("")).toBe("7d");
    expect(parse("window=1d")).toBe("1d");
    expect(parse("window=")).toBeNull();
    expect(parse("window=365d")).toBeNull();
    expect(parse("window=1d&x=1")).toBeNull();
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  clientIp,
  forwardAsk,
  MAX_BODY_BYTES,
  UPSTREAM_TIMEOUT_MS,
  type ProxyConfig,
} from "./ask-proxy";

const SECRET = "test-secret-value";
const CONFIG: ProxyConfig = {
  backendUrl: "http://backend.test/",
  sharedSecret: SECRET,
};

function askRequest(
  body: unknown = { question: "How do I use Depends?" },
  headers: Record<string, string> = {},
): Request {
  return new Request("http://localhost/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return Response.json(body, init);
}

// What fetch was called with, typed: the proxy always passes a URL string and an init object.
function lastCall(fetchImpl: ReturnType<typeof vi.fn>) {
  const [url, init] = fetchImpl.mock.calls.at(-1) as [string, RequestInit];
  return { url, init, headers: init.headers as Record<string, string> };
}

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(() => {
  vi.restoreAllMocks();
});

describe("clientIp", () => {
  it("takes the first entry of X-Forwarded-For, not the whole list", () => {
    const headers = new Headers({ "x-forwarded-for": "203.0.113.7, 10.0.0.1, 10.0.0.2" });
    expect(clientIp(headers)).toBe("203.0.113.7");
  });

  it("falls back to X-Real-IP, then to nothing", () => {
    expect(clientIp(new Headers({ "x-real-ip": "198.51.100.4" }))).toBe("198.51.100.4");
    expect(clientIp(new Headers())).toBeNull();
  });
});

describe("forwardAsk", () => {
  it("forwards the question with the secret and one client address", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ status: "answered" }));
    const res = await forwardAsk(
      askRequest({ question: "How do I use Depends?", extra: "dropped" }, {
        "x-forwarded-for": "203.0.113.7, 10.0.0.1",
        cookie: "session=abc",
      }),
      { ...CONFIG, fetchImpl },
    );

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ status: "answered" });
    const { url, init, headers } = lastCall(fetchImpl);
    expect(url).toBe("http://backend.test/v1/ask");
    expect(init.method).toBe("POST");
    expect(headers["x-proxy-secret"]).toBe(SECRET);
    expect(headers["x-client-ip"]).toBe("203.0.113.7");
    expect(headers.cookie).toBeUndefined();
    expect(JSON.parse(init.body as string)).toEqual({ question: "How do I use Depends?" });
  });

  it("sends no X-Client-IP when the platform gave no address", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({}));
    await forwardAsk(askRequest(), { ...CONFIG, fetchImpl });
    expect(lastCall(fetchImpl).headers["x-client-ip"]).toBeUndefined();
  });

  it("passes a backend error through with its status and Retry-After", async () => {
    const body = {
      error: { code: "rate_limited", message: "Too many requests.", retry_after_s: 20 },
      request_id: "3f1c7a52-0a6b-4f6e-9a39-2d3c1c1f9d11",
    };
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(jsonResponse(body, { status: 429, headers: { "retry-after": "20" } }));
    const res = await forwardAsk(askRequest(), { ...CONFIG, fetchImpl });

    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("20");
    expect(await res.json()).toEqual(body);
  });

  it("answers with a timeout error when the backend does not answer in time", async () => {
    // A fetch that never resolves but honors the abort signal, like the real one.
    const fetchImpl = vi.fn(
      (_url: unknown, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
        }),
    );
    const res = await forwardAsk(askRequest(), { ...CONFIG, timeoutMs: 20, fetchImpl });

    expect(res.status).toBe(504);
    expect(await res.json()).toMatchObject({
      error: { code: "upstream_timeout" },
      request_id: null,
    });
  });

  it("treats the platform's non-JSON 504 as a timeout too", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(new Response("An error occurred with your deployment", { status: 504 }));
    const res = await forwardAsk(askRequest(), { ...CONFIG, fetchImpl });

    expect(res.status).toBe(504);
    expect(await res.json()).toMatchObject({ error: { code: "upstream_timeout" } });
  });

  it("answers 502 when the backend cannot be reached or sends something else", async () => {
    const down = vi.fn().mockRejectedValue(new TypeError("fetch failed"));
    const res = await forwardAsk(askRequest(), { ...CONFIG, fetchImpl: down });
    expect(res.status).toBe(502);
    expect(await res.json()).toMatchObject({ error: { code: "upstream_unavailable" } });

    const html = vi.fn().mockResolvedValue(new Response("<html>", { status: 200 }));
    const res2 = await forwardAsk(askRequest(), { ...CONFIG, fetchImpl: html });
    expect(res2.status).toBe(502);
  });

  it("refuses to run without its configuration and never forwards", async () => {
    const fetchImpl = vi.fn();
    for (const config of [
      { backendUrl: undefined, sharedSecret: SECRET },
      { backendUrl: "http://backend.test", sharedSecret: undefined },
      { backendUrl: "http://backend.test", sharedSecret: "" },
    ]) {
      const res = await forwardAsk(askRequest(), { ...config, fetchImpl });
      expect(res.status).toBe(500);
      expect(await res.json()).toMatchObject({ error: { code: "proxy_misconfigured" } });
    }
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("rejects an oversized or malformed body without forwarding", async () => {
    const fetchImpl = vi.fn();
    const big = await forwardAsk(askRequest({ question: "x".repeat(MAX_BODY_BYTES) }), {
      ...CONFIG,
      fetchImpl,
    });
    const notJson = await forwardAsk(askRequest("not json"), { ...CONFIG, fetchImpl });
    const noQuestion = await forwardAsk(askRequest({ question: 42 }), { ...CONFIG, fetchImpl });

    for (const res of [big, notJson, noQuestion]) {
      expect(res.status).toBe(422);
      expect(await res.json()).toMatchObject({ error: { code: "bad_request" } });
    }
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("never puts the secret in a response", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error(`boom ${SECRET}`));
    const res = await forwardAsk(askRequest(), { ...CONFIG, fetchImpl });
    expect(await res.text()).not.toContain(SECRET);
  });

  it("uses the 35 s timeout of the chain by default", () => {
    expect(UPSTREAM_TIMEOUT_MS).toBe(35_000);
  });
});

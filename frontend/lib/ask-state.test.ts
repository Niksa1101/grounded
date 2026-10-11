import { describe, expect, it } from "vitest";

import { announcement, askReducer, classifyResponse, type View } from "./ask-state";
import { fakeFetch } from "./dev-fixtures";

const REQUEST_ID = "3f0c9d3e-6a1b-4c52-9d0e-5b7a2e8f1c44";

function failure(code: string, extra: object = {}, requestId: string | null = REQUEST_ID) {
  return { error: { code, message: `msg ${code}`, ...extra }, request_id: requestId };
}

async function viewOf(name: string): Promise<View> {
  const fake = await fakeFetch(name);
  if (!fake) throw new Error(`no fixture ${name}`);
  return classifyResponse(fake.status, fake.body);
}

describe("askReducer", () => {
  const loading: View = { kind: "loading", waking: false };
  const errored: View = { kind: "error", code: "x", message: "m", requestId: null };

  it("submit starts loading from any view", () => {
    expect(askReducer({ kind: "idle" }, { type: "submit" })).toEqual(loading);
    expect(askReducer(errored, { type: "submit" })).toEqual(loading);
  });

  it("wake only changes a loading view", () => {
    expect(askReducer(loading, { type: "wake" })).toEqual({ kind: "loading", waking: true });
    expect(askReducer(errored, { type: "wake" })).toBe(errored);
    expect(askReducer({ kind: "idle" }, { type: "wake" })).toEqual({ kind: "idle" });
  });

  it("a result is accepted only while loading", () => {
    expect(askReducer(loading, { type: "result", view: errored })).toBe(errored);
    expect(askReducer({ kind: "idle" }, { type: "result", view: errored })).toEqual({ kind: "idle" });
  });
});

describe("classifyResponse", () => {
  it("maps rate_limited with its wait", () => {
    const view = classifyResponse(429, failure("rate_limited", { retry_after_s: 20 }));
    expect(view).toEqual({
      kind: "rate_limited",
      message: "msg rate_limited",
      retryAfterS: 20,
      requestId: REQUEST_ID,
    });
  });

  it("keeps a rate_limited view without a usable wait (no countdown, still friendly)", () => {
    for (const retry of [null, 0, -3, "20"]) {
      const view = classifyResponse(429, failure("rate_limited", { retry_after_s: retry }));
      expect(view).toMatchObject({ kind: "rate_limited", retryAfterS: null });
    }
  });

  it("maps budget_exhausted", () => {
    expect(classifyResponse(503, failure("budget_exhausted", { retry_after_s: null }))).toMatchObject({
      kind: "budget_exhausted",
      requestId: REQUEST_ID,
    });
  });

  it("maps other backend codes to an error with the request id", () => {
    expect(classifyResponse(502, failure("validation_failed"))).toEqual({
      kind: "error",
      code: "validation_failed",
      message: "msg validation_failed",
      requestId: REQUEST_ID,
    });
  });

  it("handles proxy-generated errors, which have no request id", () => {
    expect(classifyResponse(504, failure("upstream_timeout", {}, null))).toMatchObject({
      kind: "error",
      code: "upstream_timeout",
      requestId: null,
    });
  });

  it("turns anything else into an unexpected_response error instead of crashing", () => {
    const badStatus = {
      status: "weird",
      answer_markdown: "",
      claims: [],
      citations: [],
      follow_up_questions: [],
      meta: {},
    };
    for (const [status, body] of [
      [200, null],
      [200, { status: "answered" }],
      [200, badStatus],
      [500, "<html>"],
      [502, {}],
    ] as const) {
      expect(classifyResponse(status, body)).toMatchObject({
        kind: "error",
        code: "unexpected_response",
      });
    }
  });
});

describe("every page state is reachable from the dev fixtures", () => {
  // "slow" resolves after 12 s on purpose (to see the waking-up message), so it is not awaited here.
  it.each([
    ["answered", "answer"],
    ["partial", "answer"],
    ["insufficient", "answer"],
    ["rate_limited", "rate_limited"],
    ["rate_limited_day", "rate_limited"],
    ["budget", "budget_exhausted"],
    ["error", "error"],
    ["proxy_timeout", "error"],
  ])("%s -> %s", async (name, kind) => {
    expect((await viewOf(name)).kind).toBe(kind);
  });

  it("the network fixture throws, like a failed fetch", async () => {
    await expect(fakeFetch("network")).rejects.toThrow();
  });

  it("an unknown name falls through to the real call", async () => {
    expect(await fakeFetch("nope")).toBeNull();
  });
});

describe("announcement", () => {
  it("is short, and says waking up after the threshold", () => {
    expect(announcement({ kind: "idle" })).toBe("");
    expect(announcement({ kind: "loading", waking: false })).toMatch(/Looking/);
    expect(announcement({ kind: "loading", waking: true })).toMatch(/Waking up/);
  });

  it("counts sources and distinguishes the three statuses", async () => {
    expect(announcement(await viewOf("answered"))).toBe("Answer ready with 2 sources.");
    expect(announcement(await viewOf("partial"))).toBe("Partial answer ready with 1 source.");
    expect(announcement(await viewOf("insufficient"))).toMatch(/do not cover/);
  });
});

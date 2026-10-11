// The ask page's state machine, kept free of React and the network so it is unit-tested in Node
// (lib/ask-state.test.ts). `classifyResponse` turns one HTTP outcome into a View; `askReducer`
// moves between views. Tech §19 lists the states.
import type { AskFailure, AskResponse } from "@/lib/api";

/** After this long without an answer the loading view says the free-tier backend is waking up. */
export const WAKING_AFTER_MS = 5_000;

export type View =
  | { kind: "idle" }
  | { kind: "loading"; waking: boolean }
  // answered, partial and insufficient_context share one renderer: `response.status` tells which.
  | { kind: "answer"; response: AskResponse }
  | { kind: "rate_limited"; message: string; retryAfterS: number | null; requestId: string | null }
  | { kind: "budget_exhausted"; message: string; requestId: string | null }
  | { kind: "error"; code: string; message: string; requestId: string | null };

export type AskAction = { type: "submit" } | { type: "wake" } | { type: "result"; view: View };

export function askReducer(state: View, action: AskAction): View {
  switch (action.type) {
    case "submit":
      return { kind: "loading", waking: false };
    case "wake":
      // A late timer must not turn an answer back into a loading view.
      return state.kind === "loading" ? { ...state, waking: true } : state;
    case "result":
      // Only a request that is still pending may deliver: a stale result must never overwrite a
      // newer view.
      return state.kind === "loading" ? action.view : state;
  }
}

const STATUSES: ReadonlySet<unknown> = new Set(["answered", "partial", "insufficient_context"]);

function isFailure(body: unknown): body is AskFailure {
  const error = (body as { error?: { message?: unknown } } | null)?.error;
  return typeof error?.message === "string";
}

// The generated types are the contract, but a body that is not even an object of the right shape
// (a platform error page, a half-written response) must become an error view, not a crash.
function isAskResponse(body: unknown): body is AskResponse {
  const r = body as Partial<AskResponse> | null;
  return (
    typeof r === "object" &&
    r !== null &&
    STATUSES.has(r.status) &&
    typeof r.answer_markdown === "string" &&
    Array.isArray(r.claims) &&
    Array.isArray(r.citations) &&
    Array.isArray(r.follow_up_questions) &&
    typeof r.meta === "object" &&
    r.meta !== null
  );
}

export function classifyResponse(httpStatus: number, body: unknown): View {
  if (isFailure(body)) {
    const { code, message, retry_after_s } = body.error;
    const requestId = typeof body.request_id === "string" ? body.request_id : null;
    if (code === "rate_limited") {
      const retryAfterS =
        typeof retry_after_s === "number" && retry_after_s > 0 ? retry_after_s : null;
      return { kind: "rate_limited", message, retryAfterS, requestId };
    }
    if (code === "budget_exhausted") return { kind: "budget_exhausted", message, requestId };
    return { kind: "error", code, message, requestId };
  }
  if (httpStatus >= 200 && httpStatus < 300 && isAskResponse(body)) {
    return { kind: "answer", response: body };
  }
  return {
    kind: "error",
    code: "unexpected_response",
    message: `Unexpected response (HTTP ${httpStatus}). Try again.`,
    requestId: null,
  };
}

export const NETWORK_ERROR_VIEW: View = {
  kind: "error",
  code: "network",
  message: "The request could not be sent. Check your connection and try again.",
  requestId: null,
};

/** The short sentence a screen reader hears when the view changes (the answer itself is not read). */
export function announcement(view: View): string {
  switch (view.kind) {
    case "idle":
      return "";
    case "loading":
      return view.waking ? "Waking up the free-tier backend." : "Looking through the docs.";
    case "answer": {
      const { status, citations } = view.response;
      if (status === "insufficient_context") return "The docs do not cover this question.";
      const label = status === "partial" ? "Partial answer" : "Answer";
      return `${label} ready with ${citations.length} ${citations.length === 1 ? "source" : "sources"}.`;
    }
    case "rate_limited":
      return "Rate limit reached.";
    case "budget_exhausted":
      return "The demo's daily budget is used up.";
    case "error":
      return "The request failed.";
  }
}

"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { AnswerView } from "@/components/answer-view";
import {
  BudgetNotice,
  ErrorNotice,
  LoadingNotice,
  RateLimitedNotice,
} from "@/components/state-notices";
import { Button } from "@/components/ui/button";
import {
  NETWORK_ERROR_VIEW,
  WAKING_AFTER_MS,
  announcement,
  askReducer,
  classifyResponse,
  type View,
} from "@/lib/ask-state";
import { retryDisplay } from "@/lib/countdown";
import {
  MAX_QUESTION_CHARS,
  MIN_QUESTION_CHARS,
  normalizeQuestion,
  questionLength,
} from "@/lib/question";

async function post(question: string): Promise<{ status: number; body: unknown }> {
  // DEV ONLY: `/?fake=<name>` answers from canned fixtures instead of the network (lib/dev-fixtures.ts).
  // `process.env.NODE_ENV` is a build-time constant, so this block and the fixtures module are
  // removed from a production build.
  if (process.env.NODE_ENV !== "production") {
    const name = new URLSearchParams(window.location.search).get("fake");
    if (name) {
      const { fakeFetch } = await import("@/lib/dev-fixtures");
      const fake = await fakeFetch(name);
      if (fake) return fake;
    }
  }
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return { status: res.status, body: await res.json().catch(() => null) };
}

async function ask(question: string): Promise<View> {
  try {
    const { status, body } = await post(question);
    return classifyResponse(status, body);
  } catch {
    return NETWORK_ERROR_VIEW;
  }
}

export function AskForm() {
  const [text, setText] = useState("");
  const [view, dispatch] = useReducer(askReducer, { kind: "idle" });
  // Set when a rate-limit countdown reaches zero, so the Ask button unlocks.
  const [waitOver, setWaitOver] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  const onWaitDone = useCallback(() => setWaitOver(true), []);

  const question = normalizeQuestion(text);
  const length = questionLength(question);
  const tooShort = length < MIN_QUESTION_CHARS;
  const tooLong = length > MAX_QUESTION_CHARS;
  const loading = view.kind === "loading";
  // A ticking rate-limit countdown blocks sending; "try again tomorrow" does not (it never ends
  // before a reload, and the server decides anyway).
  const waiting =
    view.kind === "rate_limited" &&
    retryDisplay(view.retryAfterS).mode === "countdown" &&
    !waitOver;

  useEffect(() => {
    if (view.kind !== "loading" || view.waking) return;
    const timer = setTimeout(() => dispatch({ type: "wake" }), WAKING_AFTER_MS);
    return () => clearTimeout(timer);
  }, [view]);

  async function submit(event: React.SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    if (tooShort || tooLong || loading || waiting) return;
    setWaitOver(false);
    dispatch({ type: "submit" });
    dispatch({ type: "result", view: await ask(question) });
  }

  function pickFollowUp(suggestion: string) {
    setText(suggestion);
    input.current?.focus();
  }

  return (
    <div className="flex flex-col gap-6">
      <form onSubmit={submit} className="flex flex-col gap-3">
        <label htmlFor="question" className="text-sm font-medium">
          Your question about the FastAPI docs
        </label>
        <textarea
          ref={input}
          id="question"
          name="question"
          rows={3}
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            // Enter sends, Shift+Enter is a newline (it becomes a space on send).
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          aria-describedby="question-help question-count"
          className="w-full resize-y rounded-lg border border-input bg-background px-3 py-2 text-base outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/70"
        />
        <p id="question-help" className="text-xs text-muted-foreground">
          Privacy: this demo calls free AI APIs, so do not enter personal data. Questions are kept
          for at most 30 days to measure quality, and never shown publicly.
        </p>
        <div className="flex items-center justify-between gap-3">
          <p id="question-count" className="text-xs text-muted-foreground">
            {length}/{MAX_QUESTION_CHARS} characters
            {length > 0 && tooShort ? ` (at least ${MIN_QUESTION_CHARS})` : ""}
            {tooLong ? " (too long)" : ""}
          </p>
          <Button type="submit" disabled={tooShort || tooLong || loading || waiting}>
            {loading ? "Asking…" : "Ask"}
          </Button>
        </div>
      </form>

      {/* Short spoken updates only: the answer itself is not read out. */}
      <p role="status" className="sr-only">
        {announcement(view)}
      </p>
      <div aria-busy={loading}>
        {view.kind === "loading" && <LoadingNotice waking={view.waking} />}
        {view.kind === "rate_limited" && (
          <RateLimitedNotice
            message={view.message}
            retryAfterS={view.retryAfterS}
            requestId={view.requestId}
            onDone={onWaitDone}
          />
        )}
        {view.kind === "budget_exhausted" && <BudgetNotice requestId={view.requestId} />}
        {view.kind === "error" && (
          <ErrorNotice code={view.code} message={view.message} requestId={view.requestId} />
        )}
        {view.kind === "answer" && (
          <AnswerView response={view.response} onPickFollowUp={pickFollowUp} />
        )}
      </div>
    </div>
  );
}

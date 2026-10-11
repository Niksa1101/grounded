"use client";

import { useState } from "react";

import { AnswerMarkdown } from "@/components/answer-markdown";
import { Button } from "@/components/ui/button";
import type { AskFailure, AskResponse } from "@/lib/api";
import {
  MAX_QUESTION_CHARS,
  MIN_QUESTION_CHARS,
  normalizeQuestion,
  questionLength,
} from "@/lib/question";
import { safeHref } from "@/lib/safe-url";

// The tracer's states; 5.07 adds the waking-up, rate-limited and budget views (Tech §19).
type View =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "answered"; response: AskResponse }
  | { kind: "error"; message: string; code: string; requestId: string | null };

function isFailure(body: unknown): body is AskFailure {
  const error = (body as { error?: { message?: unknown } } | null)?.error;
  return typeof error?.message === "string";
}

async function ask(question: string): Promise<View> {
  let res: Response;
  try {
    res = await fetch("/api/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    return {
      kind: "error",
      code: "network",
      message: "The request could not be sent. Check your connection and try again.",
      requestId: null,
    };
  }
  const body: unknown = await res.json().catch(() => null);
  if (res.ok && !isFailure(body)) {
    return { kind: "answered", response: body as AskResponse };
  }
  if (isFailure(body)) {
    return {
      kind: "error",
      code: body.error.code,
      message: body.error.message,
      requestId: body.request_id,
    };
  }
  return {
    kind: "error",
    code: "unexpected_response",
    message: `Unexpected response (HTTP ${res.status}). Try again.`,
    requestId: null,
  };
}

export function AskForm() {
  const [text, setText] = useState("");
  const [view, setView] = useState<View>({ kind: "idle" });

  const question = normalizeQuestion(text);
  const length = questionLength(question);
  const tooShort = length < MIN_QUESTION_CHARS;
  const tooLong = length > MAX_QUESTION_CHARS;
  const loading = view.kind === "loading";

  async function submit(event: React.SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    if (tooShort || tooLong || loading) return;
    setView({ kind: "loading" });
    setView(await ask(question));
  }

  return (
    <div className="flex flex-col gap-6">
      <form onSubmit={submit} className="flex flex-col gap-3">
        <label htmlFor="question" className="text-sm font-medium">
          Your question about the FastAPI docs
        </label>
        <textarea
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
          className="w-full resize-y rounded-lg border border-input bg-background px-3 py-2 text-base outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        />
        <p id="question-help" className="text-xs text-muted-foreground">
          Free AI APIs are used: do not enter personal data.
        </p>
        <div className="flex items-center justify-between gap-3">
          <p id="question-count" className="text-xs text-muted-foreground">
            {length}/{MAX_QUESTION_CHARS} characters
            {length > 0 && tooShort ? ` (at least ${MIN_QUESTION_CHARS})` : ""}
            {tooLong ? " (too long)" : ""}
          </p>
          <Button type="submit" disabled={tooShort || tooLong || loading}>
            {loading ? "Asking…" : "Ask"}
          </Button>
        </div>
      </form>

      <div aria-live="polite" aria-busy={loading}>
        {view.kind === "loading" && (
          <p className="text-sm text-muted-foreground">Looking through the docs…</p>
        )}
        {view.kind === "error" && (
          <div
            role="alert"
            className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm"
          >
            <p>{view.message}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              {view.code}
              {view.requestId ? ` · request ${view.requestId}` : ""}
            </p>
          </div>
        )}
        {view.kind === "answered" && <Answer response={view.response} />}
      </div>
    </div>
  );
}

function Answer({ response }: { response: AskResponse }) {
  return (
    <article className="flex flex-col gap-4">
      <p className="text-xs text-muted-foreground">
        Status: {response.status} · request {response.meta.request_id}
      </p>
      <AnswerMarkdown markdown={response.answer_markdown} />
      {response.citations.length > 0 && (
        <section aria-labelledby="sources-heading" className="flex flex-col gap-2">
          <h2 id="sources-heading" className="text-sm font-semibold">
            Sources
          </h2>
          <ol className="flex flex-col gap-2">
            {response.citations.map((citation) => {
              const href = safeHref(citation.url);
              return (
                <li
                  key={citation.n}
                  className="rounded-lg border border-border p-3 text-sm"
                >
                  <p className="font-medium">
                    [{citation.n}]{" "}
                    {href ? (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline underline-offset-4"
                      >
                        {citation.title}
                      </a>
                    ) : (
                      citation.title
                    )}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {citation.breadcrumb}
                  </p>
                  <p className="mt-1 text-muted-foreground">{citation.snippet}</p>
                </li>
              );
            })}
          </ol>
        </section>
      )}
    </article>
  );
}

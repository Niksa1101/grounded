import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { AskResponse } from "@/lib/api";
import { fakeFetch } from "@/lib/dev-fixtures";

import { AnswerView } from "./answer-view";
import { BudgetNotice, ErrorNotice, LoadingNotice, RateLimitedNotice } from "./state-notices";

async function render(name: string): Promise<string> {
  const response = (await fakeFetch(name))!.body as AskResponse;
  return renderToStaticMarkup(<AnswerView response={response} onPickFollowUp={() => {}} />);
}

describe("AnswerView", () => {
  it("source cards show breadcrumb and snippet, with a safe external link", async () => {
    const html = await render("answered");
    expect(html).toContain('id="source-1"');
    expect(html).toContain("Tutorial &gt; Background Tasks &gt; Using BackgroundTasks");
    expect(html).toContain("First, import BackgroundTasks");
    expect(html).toContain(
      'href="https://fastapi.tiangolo.com/tutorial/background-tasks/#using-backgroundtasks"',
    );
    expect(html).toContain('rel="noopener noreferrer"');
    expect(html).toContain('target="_blank"');
  });

  it("labels the answered, partial and insufficient_context statuses", async () => {
    expect(await render("answered")).toContain("Answered");
    const partial = await render("partial");
    expect(partial).toContain("Partial answer");
    expect(partial).toContain("the answer says what is missing");
    expect(await render("insufficient")).toContain("Not enough context");
  });

  it("offers follow-up questions, and shows no Sources section without citations", async () => {
    const insufficient = await render("insufficient");
    expect(insufficient).toContain("How do I deploy FastAPI with Docker?");
    expect(insufficient).not.toContain("Sources");
  });
});

describe("state notices", () => {
  const noop = () => {};

  it("shows the waking-up message only after the threshold", () => {
    expect(renderToStaticMarkup(<LoadingNotice waking={false} />)).not.toContain("Waking up");
    expect(renderToStaticMarkup(<LoadingNotice waking />)).toContain(
      "Waking up the free-tier backend",
    );
  });

  it("starts a countdown for a short wait", () => {
    const html = renderToStaticMarkup(
      <RateLimitedNotice
        message="Too many requests."
        retryAfterS={20}
        requestId="r1"
        onDone={noop}
      />,
    );
    expect(html).toContain("0:20");
    expect(html).toContain("20 seconds");
    expect(html).toContain("Request id: r1");
  });

  it("says try again tomorrow, with no ticking counter, past an hour", () => {
    const html = renderToStaticMarkup(
      <RateLimitedNotice
        message="Daily limit."
        retryAfterS={61_200}
        requestId={null}
        onDone={noop}
      />,
    );
    expect(html).toContain("try again tomorrow");
    expect(html).not.toMatch(/\d+:\d\d/);
  });

  it("budget_exhausted is friendly; an error shows the request id or says there is none", () => {
    expect(renderToStaticMarkup(<BudgetNotice requestId={null} />)).toContain("daily budget");
    const withId = renderToStaticMarkup(
      <ErrorNotice code="validation_failed" message="m" requestId="r9" />,
    );
    expect(withId).toContain("request id: r9");
    const without = renderToStaticMarkup(
      <ErrorNotice code="upstream_timeout" message="m" requestId={null} />,
    );
    expect(without).toContain("no request id");
  });
});

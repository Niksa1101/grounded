import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AnswerMarkdown } from "./answer-markdown";

function render(markdown: string): string {
  return renderToStaticMarkup(<AnswerMarkdown markdown={markdown} />);
}

describe("AnswerMarkdown", () => {
  it("renders Markdown structure", () => {
    const html = render("Use **Depends** [1]\n\n- one\n- two\n\n```py\nx = 1\n```");
    expect(html).toContain("<strong>Depends</strong>");
    expect(html).toContain("<li>one</li>");
    expect(html).toContain("<pre>");
  });

  it("shows raw HTML from the model as inert text, never as elements", () => {
    const html = render(
      'before <script>alert(1)</script> <img src=x onerror="alert2"> <b onclick="x">bold</b> after',
    );
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toMatch(/<(script|img|b)[\s>]/);
  });

  it("drops links and images but keeps the link text", () => {
    const html = render(
      "[click here](https://evil.example/x) ![alt text](https://evil.example/p.png) https://auto.example/y",
    );
    expect(html).not.toContain("<a");
    expect(html).not.toContain("<img");
    expect(html).not.toContain("href=");
    expect(html).toContain("click here");
  });

  it("does not turn a javascript: link into an element", () => {
    const html = render("[x](javascript:alert(1))");
    expect(html).not.toContain("<a");
    expect(html).not.toContain("href");
  });
});

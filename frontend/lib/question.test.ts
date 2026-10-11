import { describe, expect, it } from "vitest";

import openapi from "./openapi.json";
import {
  MAX_QUESTION_CHARS,
  MIN_QUESTION_CHARS,
  normalizeQuestion,
  questionLength,
} from "./question";

describe("normalizeQuestion", () => {
  it("collapses newlines, tabs and runs of spaces into single spaces", () => {
    expect(normalizeQuestion("  how do I\n\tuse   Depends?\r\n")).toBe(
      "how do I use Depends?",
    );
  });

  it("replaces other control characters, which the backend refuses", () => {
    expect(normalizeQuestion("a\u0000b\u001bc\u007fd")).toBe("a b c d");
  });

  it("counts code points like the backend, not UTF-16 units", () => {
    expect(questionLength("😀😀")).toBe(2);
  });
});

describe("the input limits", () => {
  it("match the committed OpenAPI schema", () => {
    const question = openapi.components.schemas.AskRequest.properties.question;
    expect(question.minLength).toBe(MIN_QUESTION_CHARS);
    expect(question.maxLength).toBe(MAX_QUESTION_CHARS);
  });
});

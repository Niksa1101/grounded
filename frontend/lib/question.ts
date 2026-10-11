// The input rules of POST /v1/ask (Tech §12), mirrored for the question box. They are API contract,
// not tunables; lib/question.test.ts checks them against the committed OpenAPI schema.
export const MIN_QUESTION_CHARS = 3;
export const MAX_QUESTION_CHARS = 500;

/**
 * What the backend will see. It refuses a control character inside the question (a pasted newline
 * or tab included), so every control character and run of whitespace becomes one space first, and
 * the ends are trimmed. Spreading into code points counts characters like Python's `len`, not
 * UTF-16 units.
 */
export function normalizeQuestion(raw: string): string {
  return raw.replace(/[\p{Cc}\s]+/gu, " ").trim();
}

export function questionLength(normalized: string): number {
  return [...normalized].length;
}

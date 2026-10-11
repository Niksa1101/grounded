/**
 * The href of a source card, or null when it is not a plain web link. Citation URLs come from the
 * database, not from the model (AGENTS.md §6.3), so this is defense in depth: a `javascript:` URL
 * can never become a clickable link, whatever ends up in the data.
 */
export function safeHref(url: string): string | null {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:"
      ? parsed.href
      : null;
  } catch {
    return null;
  }
}

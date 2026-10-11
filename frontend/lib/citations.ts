// `[n]` markers in answer_markdown become buttons that move focus to the matching source card.
// The server already rewrote the model's `c1..cK` labels to display numbers and removed invalid
// ones (Tech §9.6), so a marker here is a number; one that matches no source card stays plain text.

const MARKER = /\[(\d{1,3})\]/g;

export type MarkerPart = { type: "text"; value: string } | { type: "cite"; n: number };

/** Splits plain text around the markers that name an existing source. */
export function splitCitationMarkers(text: string, valid: ReadonlySet<number>): MarkerPart[] {
  const parts: MarkerPart[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const n = Number(match[1]);
    if (!valid.has(n)) continue;
    if (match.index > last) parts.push({ type: "text", value: text.slice(last, match.index) });
    parts.push({ type: "cite", n });
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push({ type: "text", value: text.slice(last) });
  return parts;
}

// Minimal structural view of the Markdown tree: enough to walk it without depending on the
// transitive `mdast` types.
interface TreeNode {
  type: string;
  value?: string;
  children?: TreeNode[];
  data?: { hName: string; hProperties: Record<string, string | number> };
}

function transform(node: TreeNode, valid: ReadonlySet<number>): void {
  if (!node.children) return;
  const next: TreeNode[] = [];
  for (const child of node.children) {
    if (child.type !== "text" || child.value === undefined) {
      transform(child, valid);
      next.push(child);
      continue;
    }
    for (const part of splitCitationMarkers(child.value, valid)) {
      next.push(
        part.type === "text"
          ? { type: "text", value: part.value }
          : {
              // An unknown node type with `data.hName` becomes that HTML element (remark-rehype).
              type: "citationMarker",
              data: { hName: "button", hProperties: { dataCite: part.n } },
              children: [{ type: "text", value: `[${part.n}]` }],
            },
      );
    }
  }
  node.children = next;
}

/**
 * remark plugin: text nodes only, so a `[1]` inside inline code or a fenced block (Python's
 * `items[1]`) is never touched. Code is not a text node in mdast.
 */
export function remarkCitationButtons(valid: ReadonlySet<number>) {
  return () => (tree: TreeNode) => transform(tree, valid);
}

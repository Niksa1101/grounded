import { ExternalLink } from "lucide-react";
import type { Ref } from "react";

import type { Citation } from "@/lib/api";
import { safeHref } from "@/lib/safe-url";

// One source: breadcrumb, snippet and the link to the exact section. Everything on it comes from the
// database (Tech §12), never from the model. `tabIndex={-1}` lets a `[n]` button move focus here
// without adding the card to the Tab order; the visible ring says where focus landed.
export function SourceCard({ citation, ref }: { citation: Citation; ref?: Ref<HTMLLIElement> }) {
  const href = safeHref(citation.url);
  return (
    <li
      ref={ref}
      id={`source-${citation.n}`}
      tabIndex={-1}
      className="rounded-lg border border-border p-3 text-sm outline-none focus:border-ring focus:ring-3 focus:ring-ring/70"
    >
      <p className="font-medium">
        <span className="mr-1.5 rounded-md bg-secondary px-1.5 py-0.5 text-xs">[{citation.n}]</span>
        {citation.title}
      </p>
      <p className="mt-1 text-xs text-muted-foreground">{citation.breadcrumb}</p>
      <p className="mt-2 text-muted-foreground">{citation.snippet}</p>
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-2 inline-flex items-center gap-1 rounded-sm text-sm font-medium underline underline-offset-4 outline-none focus-visible:ring-3 focus-visible:ring-ring/70"
        >
          Open this section in the FastAPI docs
          <ExternalLink aria-hidden="true" className="size-3.5" />
          <span className="sr-only">(opens in a new tab)</span>
        </a>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">No link available for this source.</p>
      )}
    </li>
  );
}

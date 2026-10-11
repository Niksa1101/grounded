"use client";

import { ChevronDown } from "lucide-react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { AskResponse } from "@/lib/api";
import { debugRows } from "@/lib/debug-rows";

// Collapsed by default (Tech §19). The trigger is a real button (Enter or Space toggles it) and
// the content is only mounted while open.
export function DebugRow({ meta }: { meta: AskResponse["meta"] }) {
  return (
    <Collapsible className="rounded-lg border border-border">
      <CollapsibleTrigger className="group flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2 text-left text-sm font-medium outline-none focus-visible:ring-3 focus-visible:ring-ring/70">
        Debug details
        <ChevronDown
          aria-hidden="true"
          className="size-4 transition-transform group-data-[panel-open]:rotate-180"
        />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 border-t border-border px-3 py-3 text-xs">
          {debugRows(meta).map((row) => (
            <div key={row.label} className="contents">
              <dt className="text-muted-foreground">{row.label}</dt>
              <dd className="font-mono break-all">{row.value}</dd>
            </div>
          ))}
        </dl>
      </CollapsibleContent>
    </Collapsible>
  );
}

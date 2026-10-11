"use client";

import { CircleAlert, CircleCheck, CircleHelp, TriangleAlert } from "lucide-react";
import { useRef } from "react";

import { AnswerMarkdown } from "@/components/answer-markdown";
import { CitationButton } from "@/components/citation-button";
import { DebugRow } from "@/components/debug-row";
import { SourceCard } from "@/components/source-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { AskResponse } from "@/lib/api";

// The UI flags claims below this server-computed confidence (Tech §9.7). It is a heuristic score,
// not the model's self-assessment (AGENTS.md §6.6), and the page says so in visible text.
export const LOW_CONFIDENCE_BELOW = 0.5;

const STATUS = {
  answered: { label: "Answered", Icon: CircleCheck },
  partial: { label: "Partial answer", Icon: CircleAlert },
  insufficient_context: { label: "Not enough context", Icon: CircleHelp },
} as const;

export function AnswerView({
  response,
  onPickFollowUp,
}: {
  response: AskResponse;
  /** Puts a suggested question into the input; it does not send it. */
  onPickFollowUp: (question: string) => void;
}) {
  const { status, citations, claims, follow_up_questions: followUps } = response;
  const cards = useRef(new Map<number, HTMLLIElement>());
  const { label, Icon } = STATUS[status];
  const hasLowClaim = claims.some((claim) => claim.confidence < LOW_CONFIDENCE_BELOW);

  function goToSource(n: number) {
    cards.current.get(n)?.focus();
  }

  return (
    <article className="flex flex-col gap-5" aria-label="Answer">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={status === "answered" ? "secondary" : "outline"} className="h-6 px-2.5 text-sm">
          <Icon aria-hidden="true" />
          {label}
        </Badge>
        {status === "partial" && (
          <span className="text-sm text-muted-foreground">
            The docs cover only part of the question; the answer says what is missing.
          </span>
        )}
      </div>

      <AnswerMarkdown
        markdown={response.answer_markdown}
        sources={citations.map((citation) => citation.n)}
        onCite={goToSource}
      />

      {claims.length > 0 && (
        <section aria-labelledby="claims-heading" className="flex flex-col gap-2">
          <h2 id="claims-heading" className="text-sm font-semibold">
            Claims and confidence
          </h2>
          <ul className="flex flex-col gap-2">
            {claims.map((claim, index) => {
              const low = claim.confidence < LOW_CONFIDENCE_BELOW;
              return (
                <li key={index} className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
                  <span>{claim.text}</span>
                  {claim.citations.map((n) => (
                    <CitationButton key={n} n={n} onCite={goToSource} />
                  ))}
                  <span className="text-xs text-muted-foreground">
                    {Math.round(claim.confidence * 100)}%
                  </span>
                  {low && (
                    <Badge variant="outline" className="border-amber-700 text-amber-900">
                      <TriangleAlert aria-hidden="true" />
                      Low confidence
                    </Badge>
                  )}
                </li>
              );
            })}
          </ul>
          <p className="text-xs text-muted-foreground">
            {hasLowClaim && "Low confidence means the supporting evidence looked weak. "}
            Confidence is a heuristic computed by the server from retrieval strength and citation
            support, not the model&apos;s own certainty and not a guarantee that a claim is true.
          </p>
        </section>
      )}

      {citations.length > 0 && (
        <section aria-labelledby="sources-heading" className="flex flex-col gap-2">
          <h2 id="sources-heading" className="text-sm font-semibold">
            Sources
          </h2>
          <ol className="flex flex-col gap-2">
            {citations.map((citation) => (
              <SourceCard
                key={citation.n}
                citation={citation}
                ref={(el) => {
                  if (el) cards.current.set(citation.n, el);
                  else cards.current.delete(citation.n);
                }}
              />
            ))}
          </ol>
        </section>
      )}

      {followUps.length > 0 && (
        <section aria-labelledby="followups-heading" className="flex flex-col gap-2">
          <h2 id="followups-heading" className="text-sm font-semibold">
            {status === "insufficient_context" ? "What the docs do cover, try asking" : "Related questions"}
          </h2>
          <ul className="flex flex-col items-start gap-2">
            {followUps.map((question) => (
              <li key={question}>
                <Button variant="outline" onClick={() => onPickFollowUp(question)} className="h-auto whitespace-normal py-1.5 text-left">
                  {question}
                </Button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <DebugRow meta={response.meta} />
    </article>
  );
}

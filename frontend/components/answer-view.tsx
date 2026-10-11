"use client";

import { CircleAlert, CircleCheck, CircleHelp } from "lucide-react";

import { AnswerMarkdown } from "@/components/answer-markdown";
import { SourceCard } from "@/components/source-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { AskResponse } from "@/lib/api";

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
  const { status, citations, follow_up_questions: followUps } = response;
  const { label, Icon } = STATUS[status];

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

      <AnswerMarkdown markdown={response.answer_markdown} />

      {citations.length > 0 && (
        <section aria-labelledby="sources-heading" className="flex flex-col gap-2">
          <h2 id="sources-heading" className="text-sm font-semibold">
            Sources
          </h2>
          <ol className="flex flex-col gap-2">
            {citations.map((citation) => (
              <SourceCard key={citation.n} citation={citation} />
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
                <Button
                  variant="outline"
                  onClick={() => onPickFollowUp(question)}
                  className="h-auto whitespace-normal py-1.5 text-left"
                >
                  {question}
                </Button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}

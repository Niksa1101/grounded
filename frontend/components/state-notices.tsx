"use client";

import { Clock, Hourglass, OctagonAlert } from "lucide-react";
import { useEffect, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { describeWait, formatCountdown, remainingSeconds, retryDisplay } from "@/lib/countdown";

export function LoadingNotice({ waking }: { waking: boolean }) {
  return (
    <div className="flex items-start gap-2 text-sm text-muted-foreground">
      <Hourglass aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
      <div>
        <p>Looking through the docs…</p>
        {waking && (
          <p className="mt-1">
            Waking up the free-tier backend… the first question after a quiet period takes longer.
          </p>
        )}
      </div>
    </div>
  );
}

function RequestId({ id }: { id: string | null }) {
  return id ? <p className="mt-1 text-xs text-muted-foreground">Request id: {id}</p> : null;
}

/**
 * Ticks once a second while a wait of at most an hour is left; past that the page says "try again
 * tomorrow" (lib/countdown.ts). `onDone` (it must be stable) fires once when the wait is over so
 * the form can unlock.
 * Only the visible digits tick: they are aria-hidden, and a static sentence carries the wait, so a
 * screen reader hears it once instead of every second.
 */
export function RateLimitedNotice({
  message,
  retryAfterS,
  requestId,
  onDone,
}: {
  message: string;
  retryAfterS: number | null;
  requestId: string | null;
  onDone: () => void;
}) {
  const display = retryDisplay(retryAfterS);
  const seconds = display.mode === "countdown" ? display.seconds : 0;
  const [left, setLeft] = useState(seconds);

  useEffect(() => {
    if (seconds === 0) return;
    const end = Date.now() + seconds * 1000;
    const timer = setInterval(() => {
      const next = remainingSeconds(end, Date.now());
      setLeft(next);
      if (next === 0) {
        clearInterval(timer);
        onDone();
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [seconds, onDone]);

  return (
    <Alert role="status">
      <Clock aria-hidden="true" />
      <AlertTitle>Slow down a little</AlertTitle>
      <AlertDescription>
        <p>{message}</p>
        {display.mode === "countdown" && left > 0 && (
          <p className="mt-1">
            <span className="sr-only">You can ask again in about {describeWait(left)}.</span>
            <span aria-hidden="true">
              You can ask again in <strong className="font-mono">{formatCountdown(left)}</strong>
            </span>
          </p>
        )}
        {display.mode === "countdown" && left === 0 && <p className="mt-1">You can ask again now.</p>}
        {display.mode === "tomorrow" && (
          <p className="mt-1">Today&apos;s question limit is used up. Please try again tomorrow.</p>
        )}
        <RequestId id={requestId} />
      </AlertDescription>
    </Alert>
  );
}

// The text is ours, not the backend's `message`: it says the same thing in friendlier words.
export function BudgetNotice({ requestId }: { requestId: string | null }) {
  return (
    <Alert role="status">
      <Hourglass aria-hidden="true" />
      <AlertTitle>The demo&apos;s daily budget is used up</AlertTitle>
      <AlertDescription>
        <p>
          This demo runs on free AI quotas with a daily cap, and today&apos;s is spent. It resets
          at 00:00 UTC, so please come back later.
        </p>
        <RequestId id={requestId} />
      </AlertDescription>
    </Alert>
  );
}

export function ErrorNotice({
  code,
  message,
  requestId,
}: {
  code: string;
  message: string;
  requestId: string | null;
}) {
  return (
    <Alert variant="destructive">
      <OctagonAlert aria-hidden="true" />
      <AlertTitle>Something went wrong</AlertTitle>
      <AlertDescription>
        <p>{message}</p>
        <p className="mt-1 text-xs">
          {code}
          {requestId ? ` · request id: ${requestId}` : " · no request id (the backend never answered)"}
        </p>
      </AlertDescription>
    </Alert>
  );
}

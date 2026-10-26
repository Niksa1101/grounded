"use client";

import { Clock } from "lucide-react";
import { useEffect, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { dataAge, formatUtc } from "@/lib/metrics";

// The page's HTML is cached (ISR), so how old its numbers are can only be judged in the browser.
// The clock is read once per mount and per summary: no timer, no polling, no request. The server
// render and hydration show no age (state starts null), so there is no hydration mismatch, and a
// client-side navigation to another window mounts this again and reads the clock again.
export function DataAgeNote({ generatedAt }: { generatedAt: string }) {
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    // A one-off read of an external clock after mount (not derived state), hence the effect.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setNow(Date.now());
  }, [generatedAt]);
  const age = now === null ? null : dataAge(generatedAt, now);
  if (age === null) return null;
  if (!age.stale) {
    return <p className="text-sm text-muted-foreground">Updated {age.text}.</p>;
  }
  // Age alone says nothing about the backend: with ISR the first visit after a quiet period gets
  // the old HTML while the page regenerates in the background.
  return (
    <Alert role="status">
      <Clock aria-hidden="true" />
      <AlertTitle>These numbers are from {age.text}</AlertTitle>
      <AlertDescription>
        They are from {formatUtc(generatedAt)}. This page is cached and refreshes in the background
        when someone visits it, so reloading in a few seconds may show newer numbers.
      </AlertDescription>
    </Alert>
  );
}

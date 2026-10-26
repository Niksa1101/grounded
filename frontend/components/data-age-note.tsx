"use client";

import { Clock } from "lucide-react";
import { useSyncExternalStore } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { dataAge, formatUtc } from "@/lib/metrics";

// The page's HTML is cached (ISR), so how old its numbers are can only be judged in the browser.
// The clock is read once per page view: no timer, no polling, no request.
let viewedAt: number | null = null;
const subscribe = () => () => {};
const getSnapshot = () => (viewedAt ??= Date.now());
const getServerSnapshot = () => null; // the server render and hydration show no age

export function DataAgeNote({ generatedAt }: { generatedAt: string }) {
  const now = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const age = now === null ? null : dataAge(generatedAt, now);
  if (age === null) return null;
  if (!age.stale) {
    return <p className="text-sm text-muted-foreground">Updated {age.text}.</p>;
  }
  return (
    <Alert role="status">
      <Clock aria-hidden="true" />
      <AlertTitle>These numbers are not current</AlertTitle>
      <AlertDescription>
        They are from {formatUtc(generatedAt)}, {age.text}: the backend has not answered since. The
        page keeps the last numbers it received and tries again at most once a minute.
      </AlertDescription>
    </Alert>
  );
}

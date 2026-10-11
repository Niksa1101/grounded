// Display rules for a rate-limit wait. The backend's `retry_after_s` is the time to the end of the
// window that rejected the request: up to a minute for the per-minute window, up to a day for the
// per-day one (Tech §12). A counter that ticks for hours helps nobody, so past an hour the UI says
// "try again tomorrow" instead (5.01).
export const MAX_TICKING_SECONDS = 3_600;

export type RetryDisplay =
  | { mode: "countdown"; seconds: number }
  | { mode: "tomorrow" }
  | { mode: "unknown" };

export function retryDisplay(retryAfterS: number | null): RetryDisplay {
  if (retryAfterS === null || !Number.isFinite(retryAfterS) || retryAfterS <= 0) {
    return { mode: "unknown" };
  }
  if (retryAfterS > MAX_TICKING_SECONDS) return { mode: "tomorrow" };
  return { mode: "countdown", seconds: Math.ceil(retryAfterS) };
}

/** `m:ss`, e.g. 20 -> "0:20", 125 -> "2:05", 3600 -> "60:00". */
export function formatCountdown(seconds: number): string {
  const total = Math.max(0, Math.ceil(seconds));
  const minutes = Math.floor(total / 60);
  return `${minutes}:${String(total % 60).padStart(2, "0")}`;
}

/** The same wait in words, for screen readers: "20 seconds", "2 minutes". */
export function describeWait(seconds: number): string {
  const total = Math.max(1, Math.ceil(seconds));
  if (total < 60) return `${total} ${total === 1 ? "second" : "seconds"}`;
  const minutes = Math.ceil(total / 60);
  return `${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
}

/** Seconds left until `endMs`, never negative; the clock is passed in so it is testable. */
export function remainingSeconds(endMs: number, nowMs: number): number {
  return Math.max(0, Math.ceil((endMs - nowMs) / 1000));
}

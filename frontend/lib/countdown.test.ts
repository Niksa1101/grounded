import { describe, expect, it } from "vitest";

import { describeWait, formatCountdown, remainingSeconds, retryDisplay } from "./countdown";

describe("retryDisplay", () => {
  it("ticks for a wait of up to an hour", () => {
    expect(retryDisplay(20)).toEqual({ mode: "countdown", seconds: 20 });
    expect(retryDisplay(3600)).toEqual({ mode: "countdown", seconds: 3600 });
  });

  it("says tomorrow past an hour (the per-day window)", () => {
    expect(retryDisplay(3601)).toEqual({ mode: "tomorrow" });
    expect(retryDisplay(86_400)).toEqual({ mode: "tomorrow" });
  });

  it("has no countdown without a usable number", () => {
    for (const bad of [null, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(retryDisplay(bad)).toEqual({ mode: "unknown" });
    }
  });
});

describe("formatCountdown", () => {
  it("formats m:ss", () => {
    expect(formatCountdown(0)).toBe("0:00");
    expect(formatCountdown(9)).toBe("0:09");
    expect(formatCountdown(20)).toBe("0:20");
    expect(formatCountdown(125)).toBe("2:05");
    expect(formatCountdown(3600)).toBe("60:00");
    expect(formatCountdown(-5)).toBe("0:00");
  });
});

describe("describeWait", () => {
  it("speaks seconds under a minute and rounds minutes up", () => {
    expect(describeWait(1)).toBe("1 second");
    expect(describeWait(20)).toBe("20 seconds");
    expect(describeWait(60)).toBe("1 minute");
    expect(describeWait(61)).toBe("2 minutes");
  });
});

describe("remainingSeconds", () => {
  it("rounds up and never goes negative", () => {
    expect(remainingSeconds(10_000, 0)).toBe(10);
    expect(remainingSeconds(10_000, 9_001)).toBe(1);
    expect(remainingSeconds(10_000, 10_000)).toBe(0);
    expect(remainingSeconds(10_000, 12_000)).toBe(0);
  });
});

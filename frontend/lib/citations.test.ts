import { describe, expect, it } from "vitest";

import { splitCitationMarkers } from "./citations";

const valid = new Set([1, 2]);

describe("splitCitationMarkers", () => {
  it("splits around markers that name a source", () => {
    expect(splitCitationMarkers("Use it [1] and also [2].", valid)).toEqual([
      { type: "text", value: "Use it " },
      { type: "cite", n: 1 },
      { type: "text", value: " and also " },
      { type: "cite", n: 2 },
      { type: "text", value: "." },
    ]);
  });

  it("handles adjacent markers and a marker at either end", () => {
    expect(splitCitationMarkers("[1][2]", valid)).toEqual([
      { type: "cite", n: 1 },
      { type: "cite", n: 2 },
    ]);
  });

  it("leaves a marker without a source as text", () => {
    expect(splitCitationMarkers("see [7] and [0]", valid)).toEqual([
      { type: "text", value: "see [7] and [0]" },
    ]);
  });

  it("returns plain text unchanged and an empty string as nothing", () => {
    expect(splitCitationMarkers("no markers", valid)).toEqual([
      { type: "text", value: "no markers" },
    ]);
    expect(splitCitationMarkers("", valid)).toEqual([]);
  });
});

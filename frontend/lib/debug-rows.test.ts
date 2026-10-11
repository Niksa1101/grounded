import { describe, expect, it } from "vitest";

import type { AskResponse } from "./api";
import { debugRows } from "./debug-rows";

const meta: AskResponse["meta"] = {
  request_id: "r1",
  provider: null,
  model: null,
  fallback_used: false,
  cache_hit: true,
  rerank_used: false,
  prompt_version: "p@1",
  index_version: "i@2",
  retrieval_config_hash: "abc",
  latency_ms: { total: 1200, embed: 100, retrieval: 40, llm: 0 },
  tokens: { input: 0, output: 0 },
  shadow_cost_usd: 0,
};

describe("debugRows", () => {
  const rows = Object.fromEntries(debugRows(meta).map((r) => [r.label, r.value]));

  it("shows each reported stage and token count, and only those", () => {
    expect(rows["Latency, total"]).toBe("1200 ms");
    expect(rows["Latency, retrieval"]).toBe("40 ms");
    expect(rows["Latency, rerank"]).toBeUndefined();
    expect(rows["Tokens, input"]).toBe("0");
  });

  it("states flags, a missing provider and the shadow cost plainly", () => {
    expect(rows["Cache hit"]).toBe("yes");
    expect(rows["Fallback used"]).toBe("no");
    expect(rows["Rerank used"]).toBe("no");
    expect(rows["Provider"]).toMatch(/none/);
    expect(rows["Shadow cost"]).toBe("$0.000000");
    expect(rows["Prompt version"]).toBe("p@1");
    expect(rows["Index version"]).toBe("i@2");
  });
});

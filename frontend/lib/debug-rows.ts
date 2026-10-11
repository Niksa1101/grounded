import type { AskResponse } from "@/lib/api";

export type DebugRow = { label: string; value: string };

const yesNo = (flag: boolean) => (flag ? "yes" : "no");

/**
 * The debug row's content as label/value pairs (Tech §19): stage latencies, tokens, provider and
 * model, the fallback/cache/rerank flags, shadow cost and the versions. Built here, not in the
 * component, so what is shown is testable. A stage the backend did not report is not invented.
 */
export function debugRows(meta: AskResponse["meta"]): DebugRow[] {
  const rows: DebugRow[] = [];
  for (const [stage, ms] of Object.entries(meta.latency_ms)) {
    rows.push({ label: `Latency, ${stage}`, value: `${ms} ms` });
  }
  for (const [kind, count] of Object.entries(meta.tokens)) {
    rows.push({ label: `Tokens, ${kind}`, value: String(count) });
  }
  rows.push(
    { label: "Provider", value: meta.provider ?? "none (no model call)" },
    { label: "Model", value: meta.model ?? "none" },
    { label: "Fallback used", value: yesNo(meta.fallback_used) },
    { label: "Cache hit", value: yesNo(meta.cache_hit) },
    { label: "Rerank used", value: yesNo(meta.rerank_used) },
    // A shadow cost: what the call would cost at list price, not money spent (Tech §14).
    { label: "Shadow cost", value: `$${meta.shadow_cost_usd.toFixed(6)}` },
    { label: "Prompt version", value: meta.prompt_version },
    { label: "Index version", value: meta.index_version },
    { label: "Retrieval config", value: meta.retrieval_config_hash },
    { label: "Request id", value: meta.request_id },
  );
  return rows;
}

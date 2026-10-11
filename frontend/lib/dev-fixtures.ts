// DEV ONLY. Canned backend responses so every state of the ask page can be shown without a
// backend or an LLM call: open `/?fake=<name>` under `npm run dev` and press Ask.
//
// It cannot ship enabled: the only importer (components/ask-form.tsx) loads this module behind
// `process.env.NODE_ENV !== "production"`. Next replaces that expression with a constant at build
// time, so the branch and the dynamic import are removed from the production bundle. 5.07 checks
// that by grepping the build output for FIXTURE_MARKER, which appears nowhere else.
import type { AskResponse } from "@/lib/api";

export const FIXTURE_MARKER = "dev-fixtures-only-7f3a";

export interface FakeHttp {
  status: number;
  body: unknown;
}

const REQUEST_ID = "3f0c9d3e-6a1b-4c52-9d0e-5b7a2e8f1c44";

const META: AskResponse["meta"] = {
  request_id: REQUEST_ID,
  provider: "gemini",
  model: "gemini-fixture-model",
  fallback_used: false,
  cache_hit: false,
  rerank_used: false,
  prompt_version: "answer.v1@9c1d2e7a",
  index_version: "fastapi-docs@1a2b3c4d",
  retrieval_config_hash: "5e6f7a8b9c0d",
  latency_ms: { total: 3120, embed: 210, retrieval: 95, llm: 2740 },
  tokens: { input: 2310, output: 214 },
  shadow_cost_usd: 0.000357,
};

const CITATIONS: AskResponse["citations"] = [
  {
    n: 1,
    chunk_id: 101,
    url: "https://fastapi.tiangolo.com/tutorial/background-tasks/#using-backgroundtasks",
    title: "Background Tasks",
    breadcrumb: "Tutorial > Background Tasks > Using BackgroundTasks",
    snippet:
      "First, import BackgroundTasks and define a parameter in your path operation function with a type declaration of BackgroundTasks. FastAPI will create the object for you and pass it in as that parameter.",
  },
  {
    n: 2,
    chunk_id: 102,
    url: "https://fastapi.tiangolo.com/tutorial/background-tasks/#create-a-task-function",
    title: "Background Tasks",
    breadcrumb: "Tutorial > Background Tasks > Create a task function",
    snippet:
      "Create a function to be run as the background task. It is just a standard function that can receive parameters. It can be an async def or normal def function.",
  },
];

const ANSWER: AskResponse = {
  status: "answered",
  answer_markdown:
    "Declare a parameter of type `BackgroundTasks` in your path operation and call `add_task` on it [1]. The task function can be a normal `def` or an `async def` [2].\n\n```python\nfrom fastapi import BackgroundTasks, FastAPI\n\napp = FastAPI()\n\n\ndef write_log(message: str):\n    with open(\"log.txt\", mode=\"a\") as log:\n        log.write(message)  # runs after the response\n\n\n@app.post(\"/send/{email}\")\nasync def send(email: str, background_tasks: BackgroundTasks):\n    background_tasks.add_task(write_log, f\"sent to {email}\")\n    return {\"message\": \"queued\"}\n```",
  claims: [
    {
      text: "A path operation declares a BackgroundTasks parameter and FastAPI passes the object in.",
      citations: [1],
      confidence: 0.86,
      confidence_components: { retrieval: 0.9, agreement: 0.8, citations: 0.67, self_confidence: 0.9, rerank: 0 },
    },
    {
      text: "The task function may be either a normal function or an async function.",
      citations: [2],
      confidence: 0.41,
      confidence_components: { retrieval: 0.5, agreement: 0.3, citations: 0.5, self_confidence: 0.8, rerank: 0 },
    },
  ],
  citations: CITATIONS,
  follow_up_questions: ["How do I run a task after the response is sent in a dependency?"],
  min_confidence: 0.41,
  meta: META,
};

const PARTIAL: AskResponse = {
  ...ANSWER,
  status: "partial",
  answer_markdown:
    "The docs show how to declare `BackgroundTasks` and call `add_task` [1]. They do not say how to retry a failed task, so that part is missing.",
  claims: ANSWER.claims.slice(0, 1),
  citations: CITATIONS.slice(0, 1),
  min_confidence: 0.86,
};

const INSUFFICIENT: AskResponse = {
  status: "insufficient_context",
  answer_markdown:
    "The documentation snapshot does not cover deploying to a specific cloud provider's managed Kubernetes service.",
  claims: [],
  citations: [],
  follow_up_questions: [
    "How do I deploy FastAPI with Docker?",
    "How do I run FastAPI behind a proxy?",
  ],
  min_confidence: null,
  meta: { ...META, latency_ms: { total: 2210, embed: 190, retrieval: 80, llm: 1850 }, tokens: { input: 1900, output: 40 } },
};

function failure(status: number, code: string, message: string, extra: object = {}, requestId: string | null = REQUEST_ID): FakeHttp {
  return { status, body: { error: { code, message, ...extra }, request_id: requestId } };
}

const FIXTURES: Record<string, () => Promise<FakeHttp>> = {
  answered: async () => ({ status: 200, body: ANSWER }),
  partial: async () => ({ status: 200, body: PARTIAL }),
  insufficient: async () => ({ status: 200, body: INSUFFICIENT }),
  // Resolves after 12 s, well past the 5 s "waking up" threshold, to see the cold-start message.
  slow: async () => {
    await new Promise((resolve) => setTimeout(resolve, 12_000));
    return { status: 200, body: ANSWER };
  },
  rate_limited: async () => failure(429, "rate_limited", "Too many requests. Try again in 20s.", { retry_after_s: 20 }),
  rate_limited_day: async () =>
    failure(429, "rate_limited", "Daily limit reached. Try again in about 17 h.", { retry_after_s: 61_200 }),
  budget: async () =>
    failure(503, "budget_exhausted", "The demo's daily budget is used up. It resets at 00:00 UTC.", { retry_after_s: null }),
  error: async () => failure(502, "validation_failed", "The model's answer could not be validated. Try again."),
  proxy_timeout: async () =>
    failure(504, "upstream_timeout", "The backend took too long to answer. Try again.", {}, null),
  network: async () => {
    throw new TypeError("fixture: network down");
  },
};

/** The canned response for `?fake=<name>`, or null for an unknown name (the real call proceeds). */
export async function fakeFetch(name: string): Promise<FakeHttp | null> {
  const fixture = FIXTURES[name];
  return fixture ? fixture() : null;
}

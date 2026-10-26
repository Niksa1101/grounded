import { CircleCheck, CircleHelp, CircleX, CloudOff, Inbox } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { DataAgeNote } from "@/components/data-age-note";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { EvalRunSummary, MetricsSummary, MetricsWindow, Percentiles } from "@/lib/api";
import {
  formatCount,
  formatMetric,
  formatMs,
  formatRate,
  formatUsd,
  formatUtc,
  WINDOW_DAYS,
  WINDOWS,
  type MetricsView,
} from "@/lib/metrics";
import { safeHref } from "@/lib/safe-url";

// The /metrics page body (Tech §19, PRD FR-27): tables only, aggregates only. A Server Component;
// the only client code is the age note. Every table has a caption, column headers with a scope and
// a row header per row; every status is a word (an icon at most repeats it).

const RING = "outline-none focus-visible:ring-3 focus-visible:ring-ring/70";
const LINK = `rounded-sm underline underline-offset-4 ${RING}`;
const CELL = "px-3 py-2 text-left align-top";
const NUM = "px-3 py-2 text-right align-top tabular-nums whitespace-nowrap";

type Row = [label: string, cells: ReactNode[]];

interface TableProps {
  id: string;
  caption: string;
  head: string[];
  /** The column indexes (after the row header) that hold numbers, right-aligned. */
  numeric: number[];
  rows: Row[];
}

function Table({ id, caption, head, numeric, rows }: TableProps) {
  const align = (i: number) => (numeric.includes(i) ? NUM : CELL);
  // A table wider than a phone scrolls inside this region, not the page; it is focusable so a
  // keyboard can scroll it too. `relative` keeps absolutely positioned children (a link's sr-only
  // text) inside the scrolling box: without it they escape it and widen the page.
  return (
    <div
      role="region"
      aria-labelledby={id}
      tabIndex={0}
      className={`relative overflow-x-auto rounded-lg border ${RING}`}
    >
      <table className="w-full border-collapse text-sm">
        <caption id={id} className="px-3 pt-3 pb-2 text-left text-muted-foreground">
          {caption}
        </caption>
        <thead className="border-b bg-muted/50">
          <tr>
            {head.map((label, i) => (
              <th key={label} scope="col" className={`${i === 0 ? CELL : align(i - 1)} font-medium`}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, cells]) => (
            <tr key={label} className="border-b last:border-b-0">
              <th scope="row" className={`${CELL} font-medium`}>
                {label}
              </th>
              {cells.map((cell, i) => (
                <td key={i} className={align(i)}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

const ICON = "size-4 shrink-0";
const STATUS: Record<EvalRunSummary["status"], [string, ReactNode]> = {
  pass: ["Pass", <CircleCheck key="i" aria-hidden="true" className={ICON} />],
  fail: ["Fail", <CircleX key="i" aria-hidden="true" className={ICON} />],
  inconclusive: ["Inconclusive", <CircleHelp key="i" aria-hidden="true" className={ICON} />],
};

/** A report link only for an https URL (a CI run page); anything else is not linked. */
function httpsHref(url: string | null): string | null {
  const href = url === null ? null : safeHref(url);
  return href?.startsWith("https://") ? href : null;
}

function evalRow(run: EvalRunSummary): Row {
  const [status, icon] = STATUS[run.status];
  const report = httpsHref(run.report_url);
  return [
    `${run.suite} / ${run.config}`,
    [
      <span key="s" className="inline-flex items-center gap-1">
        {icon}
        {status}
      </span>,
      formatCount(run.case_count),
      <ul key="m">
        {Object.keys(run.metrics)
          .sort()
          .map((name) => (
            <li key={name}>
              <span className="font-mono text-xs">{name}</span> {formatMetric(run.metrics[name])}
            </li>
          ))}
      </ul>,
      <span key="r" className="flex flex-col">
        <span className="whitespace-nowrap">{formatUtc(run.created_at)}</span>
        <span className="font-mono text-xs">{run.git_sha.slice(0, 7)}</span>
        {report && (
          <a className={LINK} href={report} rel="noreferrer">
            Report<span className="sr-only"> of {run.suite} / {run.config}</span>
          </a>
        )}
      </span>,
    ],
  ];
}

const count = (value: number): ReactNode[] => [formatCount(value)];
const stage = (label: string, p: Percentiles): Row => [
  label,
  [formatMs(p.p50), formatMs(p.p95), formatCount(p.n)],
];

const RATES: [string, (r: MetricsSummary["rates"]) => number | null, string][] = [
  ["Cache hit", (r) => r.cache_hit, "served answers that came from the answer cache"],
  [
    "Fallback",
    (r) => r.fallback,
    "generated answers that needed a provider other than the primary",
  ],
  ["Validation retry", (r) => r.validation_retry, "generated answers that needed a second attempt"],
  [
    "Validation failure",
    (r) => r.validation_failure,
    "generated answers still invalid after the retry and every fallback",
  ],
];

const RATES_CAPTION =
  "Rates. Served = answered, partial or not in the docs, cache hits included; " +
  "generated = not from the cache and the model was called.";
const LATENCY_CAPTION =
  "Latency per stage over the requests that ran the pipeline (cache hits, throttled and " +
  "rejected requests excluded). n = requests in which the stage ran.";
const COST_CAPTION =
  "Shadow cost in USD: what these requests would cost at the providers' paid list prices " +
  "(the demo runs on free tiers).";

function Tables({ data }: { data: MetricsSummary }) {
  const { requests: r, rates, latency_ms: l, cost } = data;
  const days = WINDOW_DAYS[data.window];
  const dailyCaption =
    `Per UTC day, the last ${days === 1 ? "day" : `${days} days`} (today included). ` +
    "Latency over the requests that ran the pipeline.";
  return (
    <>
      <Section title="Requests">
        <Table
          id="requests-caption"
          caption="Requests in the window, by outcome"
          head={["Outcome", "Requests"]}
          numeric={[0]}
          rows={[
            ["Total", count(r.total)],
            ["Answered", count(r.answered)],
            ["Partial answers", count(r.partial)],
            ["Not in the docs (insufficient context)", count(r.insufficient_context)],
            ["Throttled (rate limit or daily budget)", count(r.throttled)],
            ["Errors", count(r.errors)],
            ["Cache hits (part of the served answers above)", count(r.cache_hits)],
          ]}
        />
      </Section>
      <Section title="Rates">
        <Table
          id="rates-caption"
          caption={RATES_CAPTION}
          head={["Rate", "Value", "Share of"]}
          numeric={[0]}
          rows={RATES.map(([label, value, of]): Row => [label, [formatRate(value(rates)), of]])}
        />
      </Section>
      <Section title="Latency">
        <Table
          id="latency-caption"
          caption={LATENCY_CAPTION}
          head={["Stage", "p50", "p95", "n"]}
          numeric={[0, 1, 2]}
          rows={[
            stage("Total (end to end)", l.total),
            stage("Embedding", l.embed),
            stage("Retrieval", l.retrieval),
            stage("Re-ranking", l.rerank),
            stage("LLM", l.llm),
          ]}
        />
      </Section>
      <Section title="Shadow cost">
        <Table
          id="cost-caption"
          caption={COST_CAPTION}
          head={["Measure", "USD", "How it is computed"]}
          numeric={[0]}
          rows={[
            ["Total", [formatUsd(cost.shadow_usd_total), "every request in the window, failed ones included"]],
            [
              "Per 1,000 questions",
              [
                formatUsd(cost.per_1k_questions_usd),
                "total ÷ served answers × 1,000, cache hits included: what the demo costs per thousand answers",
              ],
            ],
            [
              "Per 1,000 uncached questions",
              [
                formatUsd(cost.per_1k_uncached_usd),
                "total ÷ served answers not from the cache × 1,000: comparable with the eval's " +
                  "cost per 1k, which has no cache",
              ],
            ],
          ]}
        />
      </Section>
      <Section title="By day">
        <Table
          id="daily-caption"
          caption={dailyCaption}
          head={["Day (UTC)", "Requests", "p50", "p95", "Shadow cost"]}
          numeric={[0, 1, 2, 3]}
          rows={data.daily.map((d): Row => [
            d.day,
            [formatCount(d.requests), formatMs(d.p50_ms), formatMs(d.p95_ms), formatUsd(d.shadow_usd)],
          ])}
        />
        <p className="text-sm text-muted-foreground">
          The tables above cover the last {days * 24} hours up to the timestamp; this one covers
          whole UTC days, so its sums can differ from them.
        </p>
      </Section>
      <Section title="Latest evals">
        {data.evals.length === 0 ? (
          <p className="text-sm">No eval runs recorded yet.</p>
        ) : (
          <Table
            id="evals-caption"
            caption="The latest run of each eval suite and configuration (any date, not limited by the window)."
            head={["Suite / config", "Status", "Cases", "Metrics", "Run"]}
            numeric={[1]}
            rows={data.evals.map(evalRow)}
          />
        )}
      </Section>
    </>
  );
}

function Notice({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <Alert role="status">
      {icon}
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  );
}

function WindowNav({ current }: { current: MetricsWindow }) {
  return (
    <nav aria-label="Time window">
      <ul className="flex gap-2 text-sm">
        {WINDOWS.map((w) => {
          const shown = w === current;
          const style = shown ? "bg-primary font-medium text-primary-foreground" : "hover:bg-muted";
          return (
            <li key={w}>
              {/* No prefetch: showing one window must not regenerate the others (backend calls). */}
              <Link
                href={`/metrics/${w}`}
                prefetch={false}
                aria-current={shown ? "page" : undefined}
                className={`inline-block rounded-md border px-3 py-1 ${RING} ${style}`}
              >
                {WINDOW_DAYS[w] === 1 ? "Last 24 hours" : `Last ${WINDOW_DAYS[w]} days`}
                {shown && <span className="sr-only"> (shown)</span>}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

export function MetricsDashboard({ window, view }: { window: MetricsWindow; view: MetricsView }) {
  return (
    <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-8 px-4 py-12 sm:px-6">
      <header className="flex flex-col gap-3">
        <Link className={`${LINK} w-fit text-sm`} href="/">
          ← Ask a question
        </Link>
        <h1 className="text-3xl font-semibold tracking-tight">Metrics</h1>
        <p className="text-muted-foreground">
          Aggregates of the public demo&apos;s requests and the latest eval runs. No question text is
          ever shown. The page refreshes at most once a minute.
        </p>
        <WindowNav current={window} />
        {view.kind === "ready" && (
          <>
            <p className="text-sm">
              Numbers as of{" "}
              <time dateTime={view.data.generated_at}>{formatUtc(view.data.generated_at)}</time>.
            </p>
            <DataAgeNote generatedAt={view.data.generated_at} />
          </>
        )}
      </header>
      {view.kind === "unavailable" ? (
        <Notice icon={<CloudOff aria-hidden="true" />} title="The metrics are not available right now">
          {view.code === "proxy_misconfigured"
            ? "This deployment is not connected to the backend."
            : "The free-tier backend may be asleep or slow. Reload the page in a minute."}
        </Notice>
      ) : (
        <>
          {view.empty && (
            <Notice icon={<Inbox aria-hidden="true" />} title="No requests in this window yet">
              Nobody has asked a question in the selected window, so every rate and latency below
              says &ldquo;no data&rdquo;.
            </Notice>
          )}
          <Tables data={view.data} />
        </>
      )}
    </main>
  );
}

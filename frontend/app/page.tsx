import Link from "next/link";

import { AskForm } from "@/components/ask-form";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-6 px-6 py-16">
      <header className="flex flex-col gap-2">
        <div className="flex items-baseline justify-between gap-4">
          <h1 className="text-3xl font-semibold tracking-tight">Grounded</h1>
          <nav aria-label="Site">
            {/* No prefetch: a visit to the ask page should not touch the metrics page (and its
                regeneration, which calls the backend). */}
            <Link
              href="/metrics"
              prefetch={false}
              className="rounded-sm text-sm underline underline-offset-4 outline-none focus-visible:ring-3 focus-visible:ring-ring/70"
            >
              Metrics
            </Link>
          </nav>
        </div>
        <p className="text-lg text-muted-foreground">
          Answers about the FastAPI docs with per-claim citations and a
          server-computed confidence score.
        </p>
      </header>
      <AskForm />
    </main>
  );
}

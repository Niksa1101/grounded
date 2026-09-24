export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col justify-center gap-4 px-6 py-24">
      <h1 className="text-3xl font-semibold tracking-tight">Grounded</h1>
      <p className="text-lg text-muted-foreground">
        Answers about the FastAPI docs with per-claim citations and a
        server-computed confidence score.
      </p>
      <p className="text-sm text-muted-foreground">
        Under construction. The question box arrives in Phase 5.
      </p>
    </main>
  );
}

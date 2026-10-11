import { AskForm } from "@/components/ask-form";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-6 px-6 py-16">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">Grounded</h1>
        <p className="text-lg text-muted-foreground">
          Answers about the FastAPI docs with per-claim citations and a
          server-computed confidence score.
        </p>
      </header>
      <AskForm />
    </main>
  );
}

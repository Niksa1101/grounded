// A `[n]` marker: a real button, so it is reachable with Tab and works with Enter and Space. It
// moves focus to the source card (components/answer-view.tsx); nothing depends on hover.
export function CitationButton({
  n,
  onCite,
}: {
  n: number;
  onCite?: (n: number) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onCite?.(n)}
      aria-label={`Go to source ${n}`}
      className="mx-0.5 inline-flex min-w-6 items-center justify-center rounded-md border border-border bg-secondary px-1 align-baseline text-xs font-medium text-secondary-foreground outline-none hover:bg-muted focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/70"
    >
      [{n}]
    </button>
  );
}

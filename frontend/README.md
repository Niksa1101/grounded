# Grounded frontend

Thin Next.js (App Router, TypeScript strict, Tailwind v4, shadcn/ui) client for the Grounded API.
Project overview, architecture and roadmap live in the [root README](../README.md) and [docs/](../docs/).

```bash
npm ci
npm run dev        # http://localhost:3000
npm run lint
npm run typecheck  # next typegen && tsc --noEmit
npm run build
```

Development only: `/?fake=<name>` (`answered`, `partial`, `insufficient`, `slow`, `rate_limited`, `rate_limited_day`, `budget`, `error`, `proxy_timeout`, `network`) answers from canned fixtures (`lib/dev-fixtures.ts`), so every state of the ask page can be seen without a backend. It is compiled out of production builds (Tech §19).

The same for the metrics page: `/metrics?fake=<name>` (`full`, `empty`, `stale`, `error`, `misconfigured`) renders from `lib/metrics-dev-fixtures.ts` without a backend.

Anything that touches secrets runs server-side (Route Handlers). Never add `NEXT_PUBLIC_` secrets.

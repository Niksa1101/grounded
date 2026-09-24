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

Anything that touches secrets runs server-side (Route Handlers). Never add `NEXT_PUBLIC_` secrets.

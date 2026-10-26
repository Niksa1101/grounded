import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

// Unit tests of the proxy logic and the pure helpers run in Node; no browser or jsdom is needed
// (the markdown test renders to a string). `@/` mirrors the tsconfig path alias.
export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./", import.meta.url)),
      // Next resolves this marker itself; tests run on the server side, so it is a no-op here.
      "server-only": fileURLToPath(new URL("./test/server-only-stub.ts", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["**/*.test.{ts,tsx}"],
    exclude: ["node_modules/**", ".next/**"],
  },
});

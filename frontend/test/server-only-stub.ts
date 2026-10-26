// Vitest only. Next resolves `import "server-only"` itself (to an empty module on the server, to a
// throwing one in client bundles); vitest has no such alias, so vitest.config.mts points it here.
export {};

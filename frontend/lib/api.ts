// Names for the generated API types (lib/api-types.ts, produced by `npm run generate:api`).
// Never redeclare a backend shape by hand: import it from here so a contract change breaks
// `npm run typecheck` instead of failing at runtime (Tech §19).
import type { components } from "./api-types";

export type AskRequest = components["schemas"]["AskRequest"];
export type AskResponse = components["schemas"]["AskResponse"];
export type Citation = components["schemas"]["Citation"];
export type ErrorBody = components["schemas"]["ErrorBody"];
export type ErrorResponse = components["schemas"]["ErrorResponse"];

// Errors the Route Handler itself produces (lib/ask-proxy.ts). They are not part of the backend's
// contract (Tech §13): the request never got an answer from it, so there is no backend request id.
export type ProxyErrorCode =
  | "bad_request"
  | "upstream_timeout"
  | "upstream_unavailable"
  | "proxy_misconfigured";

export interface ProxyErrorResponse {
  error: {
    code: ProxyErrorCode;
    message: string;
    retry_after_s?: null;
  };
  request_id: null;
}

/** What the browser can receive from POST /api/ask instead of an AskResponse. */
export type AskFailure = ErrorResponse | ProxyErrorResponse;

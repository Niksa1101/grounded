import { forwardAsk } from "@/lib/ask-proxy";

// Longer than the proxy's fetch timeout (35 s), which is longer than the backend's deadline and
// platform limit (Tech §18, D53). A literal on purpose: Next reads it statically at build time.
export const maxDuration = 40;

// Server-only by construction: a Route Handler never ships to the browser, and this is the only
// file that reads the secret.
export async function POST(request: Request): Promise<Response> {
  return forwardAsk(request, {
    backendUrl: process.env.BACKEND_URL,
    sharedSecret: process.env.PROXY_SHARED_SECRET,
  });
}

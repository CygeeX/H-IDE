import { timingSafeEqual } from "node:crypto";
import type { MastraAuthConfig } from "@mastra/core/server";

function matchesToken(supplied: string, expected: string): boolean {
  const suppliedBytes = Buffer.from(supplied);
  const expectedBytes = Buffer.from(expected);
  return suppliedBytes.length === expectedBytes.length && timingSafeEqual(suppliedBytes, expectedBytes);
}

export function createInternalAuth(internalToken: string): MastraAuthConfig<{ id: string } | null> {
  return {
    // The generated Mastra server includes built-in Agent/Studio endpoints in
    // addition to our custom planning routes. Protect the whole surface, then
    // explicitly expose only liveness/readiness.
    protected: [/^\//],
    public: ["/healthz", "/readyz"],
    authenticateToken: async (token) =>
      matchesToken(token, internalToken) ? { id: "h-ide-control-api" } : null,
  };
}

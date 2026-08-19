import { describe, expect, it } from "vitest";
import { createInternalAuth } from "../src/http/auth.js";

describe("planning service authentication", () => {
  it("protects every non-health route with the internal token", async () => {
    const auth = createInternalAuth("internal-secret");

    expect(auth.protected).toEqual([/^\//]);
    expect(auth.public).toEqual(["/healthz", "/readyz"]);
    await expect(auth.authenticateToken?.("wrong", {} as never)).resolves.toBeNull();
    await expect(auth.authenticateToken?.("internal-secret", {} as never)).resolves.toEqual({
      id: "h-ide-control-api",
    });
  });
});

import { describe, expect, it, vi } from "vitest";
import { createPlanningRoutes } from "../src/http/routes.js";
import { visualMission } from "./fixtures.js";

function context(token?: string) {
  const body = { mission: visualMission(), stage: visualMission().stages[0], attempt: 1 };
  return {
    req: {
      header: (name: string) => name.toLowerCase() === "authorization" && token
        ? `Bearer ${token}` : undefined,
      json: async () => body,
      param: () => undefined,
      query: () => undefined,
    },
    json: (payload: unknown, status = 200) => new Response(JSON.stringify(payload), {
      status,
      headers: { "content-type": "application/json" },
    }),
  };
}

describe("visual stage route", () => {
  const service = { submit: vi.fn(), get: vi.fn(), cancel: vi.fn() };
  const handler = (route: ReturnType<typeof createPlanningRoutes>[number] | undefined) => {
    if (!route || !("handler" in route)) throw new Error("route handler is missing");
    return route.handler as never as (value: unknown) => Promise<Response>;
  };

  it("requires internal auth and returns no coordinates", async () => {
    const resolver = { resolve: vi.fn().mockResolvedValue({
      status: "succeeded",
      visualResolutionId: "vr-fixture-1",
      ambiguityCandidates: [],
      explanation: "canonical fixture resolution",
    }) };
    const routes = createPlanningRoutes(service as never, "internal-secret", resolver);
    const route = routes.find((item) => item.path === "/internal/v1/visual-stages/resolve");
    expect((await handler(route)(context())).status).toBe(401);
    const response = await handler(route)(context("internal-secret"));
    expect(response.status).toBe(200);
    expect(JSON.stringify(await response.json())).not.toContain("goalPose");
    expect(resolver.resolve).toHaveBeenCalledTimes(1);
  });
});

import { describe, expect, it, vi } from "vitest";
import { createPlanningRoutes } from "../src/http/routes.js";
import { envelope } from "./fixtures.js";

function context({
  token,
  body,
  taskId = "task-1",
  afterEvent = "0",
}: {
  token?: string;
  body?: unknown;
  taskId?: string;
  afterEvent?: string;
}) {
  return {
    req: {
      header: (name: string) =>
        name.toLowerCase() === "authorization" && token ? `Bearer ${token}` : undefined,
      json: async () => body,
      param: (name: string) => (name === "taskId" ? taskId : undefined),
      query: (name: string) => (name === "afterEvent" ? afterEvent : undefined),
    },
    json: (payload: unknown, status = 200) =>
      new Response(JSON.stringify(payload), {
        status,
        headers: { "content-type": "application/json" },
      }),
  };
}

describe("internal planning routes", () => {
  const handler = (route: ReturnType<typeof createPlanningRoutes>[number] | undefined) => {
    if (!route || !("handler" in route)) throw new Error("route handler is missing");
    return route.handler as never as (value: unknown) => Promise<Response>;
  };

  it("rejects missing internal bearer auth", async () => {
    const service = { submit: vi.fn(), get: vi.fn(), cancel: vi.fn() };
    const routes = createPlanningRoutes(service as never, "internal-secret");
    const submit = routes.find((route) => route.path === "/internal/v1/planning-turns");
    const response = await handler(submit)(context({ body: envelope() }));
    expect(response.status).toBe(401);
    expect(service.submit).not.toHaveBeenCalled();
  });

  it("accepts an async turn and forwards the event cursor", async () => {
    const record = {
      taskId: "task-1",
      conversationId: "conversation-1",
      status: "queued",
      result: null,
      error: null,
      cancellationRequested: false,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      events: [],
      lastEventId: 1,
    };
    const service = {
      submit: vi.fn().mockResolvedValue(record),
      get: vi.fn().mockResolvedValue(record),
      cancel: vi.fn().mockResolvedValue({ ...record, status: "cancelled" }),
    };
    const routes = createPlanningRoutes(service as never, "internal-secret");
    const submit = routes.find(
      (route) => route.path === "/internal/v1/planning-turns" && route.method === "POST",
    );
    const accepted = await handler(submit)(context({ token: "internal-secret", body: envelope() }));
    expect(accepted.status).toBe(202);
    expect(service.submit).toHaveBeenCalledTimes(1);

    const get = routes.find(
      (route) => route.path === "/internal/v1/planning-turns/:taskId" && route.method === "GET",
    );
    const fetched = await handler(get)(context({ token: "internal-secret", afterEvent: "7" }));
    expect(fetched.status).toBe(200);
    expect(service.get).toHaveBeenCalledWith("task-1", 7);
  });
});

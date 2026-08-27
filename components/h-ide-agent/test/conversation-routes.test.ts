import { describe, expect, it, vi } from "vitest";
import { createConversationRoutes } from "../src/conversation/routes.js";

function context({ token, body }: { token?: string; body?: unknown }) {
  const controller = new AbortController();
  return {
    req: {
      raw: { signal: controller.signal },
      header: (name: string) =>
        name.toLowerCase() === "authorization" && token ? `Bearer ${token}` : undefined,
      json: async () => body,
    },
    json: (payload: unknown, status = 200) =>
      new Response(JSON.stringify(payload), {
        status,
        headers: { "content-type": "application/json" },
      }),
  };
}

describe("Assistant V2 conversation route", () => {
  const handler = (routes: ReturnType<typeof createConversationRoutes>) => {
    const route = routes.find((item) => item.path === "/internal/v2/conversation-turns");
    if (!route || !("handler" in route)) throw new Error("conversation route is missing");
    return route.handler as never as (value: unknown) => Promise<Response>;
  };

  it("requires the internal bearer token", async () => {
    const service = { route: vi.fn() };
    const response = await handler(createConversationRoutes(service as never, "internal-secret"))(
      context({ body: { conversationId: "conversation-1", text: "你好" } }),
    );
    expect(response.status).toBe(401);
    expect(service.route).not.toHaveBeenCalled();
  });

  it("rejects an invalid body before invoking the model service", async () => {
    const service = { route: vi.fn() };
    const response = await handler(createConversationRoutes(service as never, "internal-secret"))(
      context({ token: "internal-secret", body: { conversationId: "conversation-1", text: "" } }),
    );
    expect(response.status).toBe(422);
    expect(service.route).not.toHaveBeenCalled();
  });

  it("returns the strict routed response synchronously", async () => {
    const service = {
      route: vi.fn().mockResolvedValue({
        schemaVersion: "h-ide.conversation-result/2",
        conversationId: "conversation-1",
        intent: "SOCIAL_CHAT",
        assistantMessage: "你好！",
        needsRobot: false,
        shouldCreateMission: false,
      }),
    };
    const response = await handler(createConversationRoutes(service as never, "internal-secret"))(
      context({
        token: "internal-secret",
        body: { conversationId: "conversation-1", locale: "zh-CN", text: "你好" },
      }),
    );
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      intent: "SOCIAL_CHAT",
      shouldCreateMission: false,
    });
    expect(service.route).toHaveBeenCalledTimes(1);
  });

  it("maps exhausted inference failures to a gateway error", async () => {
    const service = { route: vi.fn().mockRejectedValue(new Error("model unavailable")) };
    const response = await handler(createConversationRoutes(service as never, "internal-secret"))(
      context({
        token: "internal-secret",
        body: { conversationId: "conversation-1", text: "解释一下量子纠缠" },
      }),
    );
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ error: "model unavailable" });
  });
});

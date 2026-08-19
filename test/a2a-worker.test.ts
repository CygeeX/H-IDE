import { describe, expect, it, vi } from "vitest";
import { MastraA2AWorker, validateRemoteWorkerUrl } from "../src/a2a/worker.js";
import { envelope, motionFragment } from "./fixtures.js";

const agentCard = {
  name: "Remote Motion Planner",
  description: "Plans mobile-base actions",
  url: "https://agents.example.com/a2a/motion",
  version: "1.0",
  protocolVersion: "0.3.0",
  skills: [],
  defaultInputModes: ["text/plain"],
  defaultOutputModes: ["text/plain"],
  capabilities: {
    streaming: false,
    pushNotifications: false,
    stateTransitionHistory: false,
    extensions: [],
  },
  security: [],
  securitySchemes: {},
  additionalInterfaces: [],
  supportsAuthenticatedExtendedCard: false,
};

function messageResult(text: string) {
  return new Response(
    JSON.stringify({
      jsonrpc: "2.0",
      id: "1",
      result: {
        kind: "message",
        role: "agent",
        messageId: "message-1",
        parts: [{ kind: "text", text }],
      },
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

describe("Mastra A2A worker adapter", () => {
  it("discovers an A2A 0.3 agent, authenticates, and validates its fragment", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(new Headers(init?.headers).get("authorization")).toBe("Bearer remote-token");
      if (String(input).endsWith("/.well-known/agent-card.json")) {
        return new Response(JSON.stringify(agentCard), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return messageResult(JSON.stringify(motionFragment()));
    });
    const worker = new MastraA2AWorker(
      {
        id: "motion-worker",
        name: "Remote Motion Planner",
        description: "Plans motion",
        url: "https://agents.example.com",
        token: "remote-token",
        timeoutMs: 1_000,
      },
      fetchMock as typeof fetch,
    );

    const fragment = await worker.run(envelope(), new AbortController().signal);

    expect(fragment.agentId).toBe("motion-worker");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("fails closed on malformed remote output", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(agentCard), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(messageResult("not-json"));
    const worker = new MastraA2AWorker(
      {
        id: "motion-worker",
        name: "Remote Motion Planner",
        description: "Plans motion",
        url: "https://agents.example.com",
        timeoutMs: 1_000,
      },
      fetchMock as typeof fetch,
    );

    await expect(worker.run(envelope(), new AbortController().signal)).rejects.toThrow(/JSON/i);
  });

  it("enforces the configured origin allowlist and supports cancellation", async () => {
    expect(() =>
      validateRemoteWorkerUrl("https://evil.example.net", ["https://agents.example.com"]),
    ).toThrow(/allowlist/i);
    const controller = new AbortController();
    controller.abort();
    const worker = new MastraA2AWorker(
      {
        id: "motion-worker",
        name: "Remote Motion Planner",
        description: "Plans motion",
        url: "https://agents.example.com",
        timeoutMs: 1_000,
      },
      vi.fn() as typeof fetch,
    );
    await expect(worker.run(envelope(), controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  });

  it("does not follow a remote worker redirect outside the allowlisted origin", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(null, {
        status: 302,
        headers: { location: "http://127.0.0.1:8080/private" },
      }),
    );
    const worker = new MastraA2AWorker(
      {
        id: "motion-worker",
        name: "Remote Motion Planner",
        description: "Plans motion",
        url: "https://agents.example.com",
        timeoutMs: 1_000,
      },
      fetchMock as typeof fetch,
    );

    await expect(worker.run(envelope(), new AbortController().signal)).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalled();
    expect(fetchMock.mock.calls[0]![1]?.redirect).toBe("manual");
  });

  it("rejects an agent card that pivots subsequent requests to another origin", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ ...agentCard, url: "http://127.0.0.1:8080/private" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(messageResult(JSON.stringify(motionFragment())));
    const worker = new MastraA2AWorker(
      {
        id: "motion-worker",
        name: "Remote Motion Planner",
        description: "Plans motion",
        url: "https://agents.example.com",
        timeoutMs: 1_000,
      },
      fetchMock as typeof fetch,
    );

    await expect(worker.run(envelope(), new AbortController().signal)).rejects.toThrow(/origin/i);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

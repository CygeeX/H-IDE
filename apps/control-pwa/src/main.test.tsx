import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { App } from "./main";

const authHeaders = { "content-type": "application/json" };

function response(data: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: authHeaders,
    ...init,
  });
}

describe("control console", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/robots/robot-1/capabilities")) {
        return Promise.resolve(response({ robotId: "robot-1", agent: "fake-agent", protocol: "fake", capabilities: ["move.forward", "stop", "gripper.open"], safetyMode: "normal" }));
      }
      if (url.endsWith("/robots")) {
        return Promise.resolve(response([{ robotId: "robot-1", status: "online", connected: true, safetyMode: "normal", battery: 82 }]));
      }
      if (url.includes("/mappings")) {
        return Promise.resolve(response([]));
      }
      if (url.includes("/assistant/drafts")) {
        const plan = JSON.parse(String(init?.body ?? "{}"));
        return Promise.resolve(response({ draftId: plan.draftId, status: "draft", plan }));
      }
      if (url.includes("/commands")) {
        return Promise.resolve(response({ commandId: "cmd-test", robotId: "robot-1", status: "succeeded" }));
      }
      if (url.includes("/events")) {
        return Promise.resolve(response([]));
      }
      return Promise.resolve(response({}));
    }));
    class FakeWebSocket {
      static OPEN = 1;
      readyState = 1;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: (() => void) | null = null;
      onerror: (() => void) | null = null;
      constructor() { queueMicrotask(() => this.onopen?.()); }
      close() { this.onclose?.(); }
    }
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("keeps motion behind a high-level command button and exposes stop", async () => {
    render(<App />);
    expect((await screen.findAllByText("robot-1")).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /立即停止/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /向前移动/ })).toBeInTheDocument();
    expect(screen.queryByText("cmd_vel")).not.toBeInTheDocument();
  });

  it("requires explicit confirmation before an assistant plan is dispatched", async () => {
    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.click(screen.getByRole("tab", { name: /Agent 计划/ }));
    expect(screen.getByRole("button", { name: "确认并执行计划" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("计划 JSON"), {
      target: { value: JSON.stringify({ draftId: "ui-plan", robotId: "robot-1", steps: [{ stepId: "s1", commandKey: "move.forward", payload: {} }] }) },
    });
    fireEvent.click(screen.getByRole("button", { name: "生成待确认草案" }));
    expect(await screen.findByText("待确认：ui-plan")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "确认并执行计划" })).not.toBeDisabled();
  });

  it("shows interface center and protects activation behind validation/probe", async () => {
    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.click(screen.getByRole("tab", { name: /接口中心/ }));
    expect(await screen.findByText("接口版本中心")).toBeInTheDocument();
    expect(screen.getByText("先校验并探测，再启用版本")).toBeInTheDocument();
  });
});

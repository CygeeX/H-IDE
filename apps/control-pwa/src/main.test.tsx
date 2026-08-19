import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import {
  apiErrorDetail,
  App,
  isDraftExpired,
  isDraftReviewableForRobot,
  preferredRobotId,
  shouldReconcileMission,
} from "./main";
import type { MissionDraftRecordV2, MissionRecordV2, MissionStatus } from "./contracts";

vi.mock("./audio", () => ({
  BrowserWavRecorder: class {
    async start() { return undefined; }
    async stop() { return new Blob(["wav"], { type: "audio/wav" }); }
    async cancel() { return undefined; }
  },
}));

vi.mock("./streaming-speech", () => ({
  speechWebSocketUrl: () => "ws://localhost/api/v2/speech/stream",
  BrowserStreamingSpeechSession: class {
    private callbacks: {
      onPhase?: (phase: string) => void;
      onPartial?: (event: { text: string }) => void;
      onFinal?: (event: { text: string; utteranceId: string; transcriptionId: string }) => void;
      onMetrics?: (metrics: { framesSent: number; audioMsSent: number }) => void;
    };
    constructor(options: { callbacks?: {
      onPhase?: (phase: string) => void;
      onPartial?: (event: { text: string }) => void;
      onFinal?: (event: { text: string; utteranceId: string; transcriptionId: string }) => void;
      onMetrics?: (metrics: { framesSent: number; audioMsSent: number }) => void;
    } }) {
      this.callbacks = options.callbacks ?? {};
    }
    async start() {
      this.callbacks.onPhase?.("listening");
      this.callbacks.onMetrics?.({ framesSent: 5, audioMsSent: 200 });
      this.callbacks.onPartial?.({ text: "临时识别文本" });
    }
    async finish() {
      this.callbacks.onFinal?.({
        text: "语音要求机器人向前移动",
        utteranceId: "utterance-ui",
        transcriptionId: "transcription-stream-ui",
      });
      this.callbacks.onPhase?.("closed");
    }
    async cancel() { this.callbacks.onPhase?.("closed"); }
  },
  StreamingPcmPlayer: class {
    async unlock() { return undefined; }
    async speak() { return undefined; }
    stop() { return undefined; }
    async close() { return undefined; }
  },
}));

const authHeaders = { "content-type": "application/json" };
let draftExpiresAt = "2099-01-01T00:00:00Z";
let confirmConflict = false;

function response(data: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: authHeaders,
    ...init,
  });
}

function missionDraftFixture(objective = "向前移动后打开夹爪"): MissionDraftRecordV2 {
  return {
    draftId: "mission-ui",
    robotId: "robot-1",
    status: "draft",
    plan: {
      planType: "mission",
      schemaVersion: "h-ide.mission/2",
      planId: "mission-ui",
      robotId: "robot-1",
      objective,
      nodes: [
        {
          nodeId: "observe-target",
          skillId: "vision.observe",
          skillVersion: "2.0.0",
          purpose: "观察并定位目标",
          intent: { targetDescription: "目标" },
          dependsOn: [],
          preconditionIds: [],
          postconditionIds: ["target-pose"],
          riskLevel: "low",
        },
        {
          nodeId: "navigate-target",
          skillId: "navigation.navigate_to_pose",
          skillVersion: "2.0.0",
          purpose: "通过 Nav2 到达几何确认的目标点",
          intent: { targetDescription: "目标" },
          dependsOn: ["observe-target"],
          preconditionIds: ["target-pose"],
          postconditionIds: ["arrived"],
          riskLevel: "medium",
        },
      ],
      envelope: {
        schemaVersion: "h-ide.mission-envelope/2",
        robotId: "robot-1",
        robotBootId: "fixture-boot-1",
        localizationEpochId: "lab-map-epoch-1",
        capabilityManifestDigest: `sha256:${"a".repeat(64)}`,
        robotProfileDigest: `sha256:${"b".repeat(64)}`,
        workstationCapabilityDigest: `sha256:${"c".repeat(64)}`,
        allowedSkillIds: ["vision.observe", "navigation.navigate_to_pose"],
        maximumRiskLevel: "medium",
        maximumGoalDistanceM: 5,
        maximumCumulativeDistanceM: 20,
        maximumSpeedMps: 0.2,
        maximumDurationSeconds: 600,
        maximumAutoRevisions: 3,
        maximumStageAttempts: 3,
        maximumScanYawRad: 6.28,
        sceneRefs: [],
        expiresAt: "2099-01-01T00:00:00Z",
      },
      createdAt: "2026-08-17T00:00:00Z",
      expiresAt: "2099-01-01T00:00:00Z",
    },
  };
}

function missionRecordFixture(status: MissionStatus = "approved"): MissionRecordV2 {
  const draft = missionDraftFixture();
  return {
    missionId: "mission-ui",
    robotId: "robot-1",
    status,
    plan: draft.plan,
    authorization: { authorizationId: "auth-mission-ui-1" },
    activeRevision: {
      revisionId: "revision-mission-ui-1",
      sequence: 1,
      objective: draft.plan.objective,
      nodes: draft.plan.nodes,
      disposition: "human_approved",
      changeSummary: ["Initial Mission DAG approved"],
      createdAt: "2026-08-17T00:00:00Z",
    },
    revisions: [],
    snapshot: {
      missionId: "mission-ui",
      robotId: "robot-1",
      status,
      lastSequence: status === "completed" ? 8 : 2,
      activeRevisionId: "revision-mission-ui-1",
      activeNodeId: status === "completed" ? null : "observe-target",
      activeSkillRequestId: null,
      autoRevisionCount: 0,
      updatedAt: "2026-08-17T00:00:00Z",
      detail: null,
    },
    events: status === "completed" ? [{
      eventId: "event-mission-ui-8",
      missionId: "mission-ui",
      sequence: 8,
      eventType: "skill.succeeded",
      occurredAt: "2026-08-17T00:00:00Z",
      correlationId: "skill-nav-1",
      nodeId: "navigate-target",
      payload: {},
    }] : [],
    worldRecords: [],
    nodeStates: draft.plan.nodes.map((node) => ({
      missionId: "mission-ui",
      revisionId: "revision-mission-ui-1",
      nodeId: node.nodeId,
      status: status === "completed" ? "succeeded" : "pending",
      attempt: status === "completed" ? 1 : 0,
      state: {},
      updatedAt: "2026-08-17T00:00:00Z",
    })),
    skillExecutions: [],
  };
}

function missionVisualRecordFixture() {
  const mission = missionRecordFixture("completed");
  const observationNode = mission.activeRevision.nodes[0];
  mission.activeRevision.nodes = [observationNode];
  mission.plan.nodes = [observationNode];
  mission.nodeStates = [mission.nodeStates[0]];
  mission.skillExecutions = [{
    requestId: "skill-visual-ui-1",
    result: {
      status: "succeeded",
      result: {
        visualResult: {
          schemaVersion: "h-ide.visual-result/2",
          analysisId: "analysis-visual-ui-1",
          mode: "landmark_observation",
          status: "succeeded",
          source: {
            sourceMode: "synthetic",
            sourceId: "mujoco:unitree-g1:g1-sim",
            sensorId: "head_rgbd",
            capturedAt: "2026-08-18T14:48:34Z",
          },
          versions: {
            vlmProvider: "qwen",
            vlmModel: "qwen3-vl-plus",
            visualAlgorithmVersion: "rgbd-target-v2",
            segmentationProvider: "sam2",
            segmentationModel: "sam2.1_hiera_tiny",
          },
          hypotheses: [{
            hypothesisId: "qwen-display-1",
            source: "qwen",
            label: "显示器",
            pixelBox: [339, 221, 357, 270],
            score: 1,
          }],
          geometryCandidates: [{
            geometryCandidateId: "geometry-display-1",
            role: "landmark",
            verified: true,
            landmarkPose: { frameId: "map", x: 3.384, y: -0.616, z: 1.188, yaw: 0 },
            targetPose: null,
            safetyGates: [{ name: "stable_frames", passed: true, measured: 3, threshold: 3 }],
          }],
          evidence: [{ kind: "rgb_keyframe", sha256: "d".repeat(64), mediaType: "image/jpeg" }],
          warnings: [],
        },
      },
    },
  }];
  return mission;
}

describe("control console", () => {
  it("formats FastAPI validation details instead of showing object coercion", () => {
    expect(apiErrorDetail({
      detail: [{ loc: ["body", "groundPoints", 1, "z"], msg: "Input should be a valid number" }],
    }, "Unprocessable Entity")).toBe("groundPoints.1.z: Input should be a valid number");
  });

  it("never enables a stale draft after the operator switches robots", () => {
    expect(isDraftReviewableForRobot({
      draftId: "draft-robot-1",
      status: "draft",
      plan: { robotId: "robot-1", steps: [] },
    }, "robot-2", false)).toBe(false);
  });

  it("never enables a PlanDraft after its review window expires", () => {
    const expired = {
      draftId: "draft-expired",
      status: "draft",
      plan: { robotId: "robot-1", steps: [], expiresAt: "2000-01-01T00:00:00Z" },
    };
    expect(isDraftExpired(expired)).toBe(true);
    expect(isDraftReviewableForRobot(expired, "robot-1", false)).toBe(false);
    expect(isDraftExpired({ ...expired, status: "confirmed" })).toBe(false);
  });

  it("selects a connected online robot instead of the first offline record", () => {
    const robots = [
      { robotId: "g1-dev", status: "unknown", connected: false, safetyMode: "normal" },
      { robotId: "g1-sim", status: "online", connected: true, safetyMode: "normal" },
    ] as Parameters<typeof preferredRobotId>[0];

    expect(preferredRobotId(robots, "")).toBe("g1-sim");
    expect(preferredRobotId(robots, "g1-dev")).toBe("g1-dev");
  });

  it("keeps reconciling a failed skill while the Mission enters planning", () => {
    expect(shouldReconcileMission("executing")).toBe(true);
    expect(shouldReconcileMission("planning")).toBe(true);
    expect(shouldReconcileMission("needs_input")).toBe(false);
    expect(shouldReconcileMission("completed")).toBe(false);
  });

  beforeEach(() => {
    draftExpiresAt = "2099-01-01T00:00:00Z";
    confirmConflict = false;
    window.history.replaceState({}, "", "/developer");
    window.localStorage.clear();
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/auth/me")) {
        return Promise.resolve(response({
          csrfToken: "csrf-ui",
          principal: {
            userId: "developer-ui",
            username: "developer-ui",
            displayName: "Developer UI",
            role: "DEVELOPER",
            siteId: "default",
            authenticationMethod: "session",
            scopes: ["api:access", "robots:read", "assistant:use", "speech:use", "missions:use", "events:read", "commands:manage", "mappings:manage", "simulations:manage", "diagnostics:read"],
          },
        }));
      }
      if (url.includes("/robots/robot-1/capabilities")) {
        return Promise.resolve(response({ robotId: "robot-1", agent: "fake-agent", protocol: "fake", capabilities: ["move.forward", "stop", "gripper.open"], safetyMode: "normal" }));
      }
      if (url.endsWith("/robots")) {
        return Promise.resolve(response([{ robotId: "robot-1", status: "online", connected: true, safetyMode: "normal", battery: 82 }]));
      }
      if (url.includes("/mappings")) {
        return Promise.resolve(response([]));
      }
      if (url.endsWith("/scenes")) {
        return Promise.resolve(response([]));
      }
      if (url.endsWith("/api/v2/missions/drafts") && init?.method === "POST") {
        const body = typeof init.body === "string" ? JSON.parse(init.body) : {};
        return Promise.resolve(response(missionDraftFixture(body.objective), { status: 201 }));
      }
      if (url.endsWith("/api/v2/missions/drafts/mission-ui/approve")) {
        return Promise.resolve(response(missionRecordFixture("approved")));
      }
      if (url.endsWith("/api/v2/missions/drafts/mission-ui") && !init?.method) {
        return Promise.resolve(response(missionDraftFixture()));
      }
      if (url.endsWith("/api/v2/missions/drafts/mission-ui/reject")) {
        return Promise.resolve(response({
          ...missionDraftFixture(),
          status: "rejected",
          error: "人工审核拒绝",
        }));
      }
      if (url.endsWith("/api/v2/missions/mission-ui/advance")) {
        return Promise.resolve(response({
          mission: missionRecordFixture("completed"),
          actions: [
            { kind: "started" },
            { kind: "skill_issued", skillId: "vision.observe" },
            { kind: "skill_issued", skillId: "navigation.navigate_to_pose" },
          ],
          stoppedBecause: "completed",
        }));
      }
      if (url.endsWith("/assistant/conversations")) {
        if (init?.method === "POST") {
          return Promise.resolve(response({
            conversationId: "conversation-ui",
            robotId: "robot-1",
            requestedBy: "operator",
            locale: "zh-CN",
            status: "idle",
          }, { status: 201 }));
        }
        return Promise.resolve(response([]));
      }
      if (url.includes("/assistant/conversations/conversation-ui/messages")) {
        return Promise.resolve(response({ taskId: "task-ui", conversationId: "conversation-ui", status: "queued" }, { status: 202 }));
      }
      if (url.includes("/speech/transcriptions")) {
        return Promise.resolve(response({
          transcriptionId: "transcription-ui",
          text: "语音要求机器人向前移动",
          locale: "zh-CN",
          createdAt: new Date().toISOString(),
        }));
      }
      if (url.includes("/assistant/drafts/ui-draft/revise")) {
        return Promise.resolve(response({ taskId: "task-revision", conversationId: "conversation-ui", status: "queued" }, { status: 202 }));
      }
      if (url.includes("/assistant/turns/task-revision")) {
        return Promise.resolve(response({
          taskId: "task-revision",
          conversationId: "conversation-ui",
          status: "cannot_plan",
          result: { status: "cannot_plan", assistantMessage: "当前反馈无法形成计划" },
        }));
      }
      if (url.endsWith("/assistant/conversations/conversation-ui")) {
        return Promise.resolve(response({
          conversationId: "conversation-ui",
          robotId: "robot-1",
          requestedBy: "operator",
          locale: "zh-CN",
          status: "review",
          messages: [],
          turns: [],
          drafts: [{
            draftId: "ui-draft",
            status: "draft",
            plan: {
              planId: "ui-draft",
              robotId: "robot-1",
              expiresAt: draftExpiresAt,
              steps: [{ stepId: "move", commandKey: "move.forward", payload: {}, dependsOn: [] }],
            },
            review: {
              robotId: "robot-1",
              summary: "先移动，再打开夹爪",
              steps: [{ stepId: "move", commandKey: "move.forward", payload: {}, dependsOn: [], rationale: "接近目标", contributedBy: "motion-worker" }],
              assumptions: [],
              warnings: [],
              contributions: [],
              contextVersion: "ctx-ui",
            },
          }],
        }));
      }
      if (url.includes("/assistant/turns/task-ui")) {
        return Promise.resolve(response({
          taskId: "task-ui",
          conversationId: "conversation-ui",
          status: "completed",
          result: {
            schemaVersion: "h-ide.assistant-result/2",
            intent: "ROBOT_ACTION",
            route: "mission_review",
            routeStatus: "review",
            assistantMessage: "行动方案已生成，等待你确认。",
            missionDraftId: "mission-ui",
          },
        }));
      }
      if (url.includes("/assistant/drafts/ui-draft/confirm")) {
        if (confirmConflict) {
          return Promise.resolve(response({ detail: "assistant plan has expired" }, { status: 409 }));
        }
        return Promise.resolve(response({ draftId: "ui-draft", status: "confirmed", receipts: [] }));
      }
      if (url.endsWith("/assistant/drafts/ui-draft")) {
        return Promise.resolve(response({
          draftId: "ui-draft",
          status: "expired",
          error: "assistant plan has expired",
          plan: {
            planId: "ui-draft",
            robotId: "robot-1",
            expiresAt: "2000-01-01T00:00:00Z",
            steps: [{ stepId: "move", commandKey: "move.forward", payload: {}, dependsOn: [] }],
          },
        }));
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
    expect(screen.getByRole("button", { name: "停止机器人" })).toBeInTheDocument();
    expect(screen.getByText("工具")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "打开工具" }));
    expect(screen.getByRole("button", { name: /系统诊断/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /手动控制/ }));
    expect(screen.getByRole("button", { name: /立即停止/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /按住向前移动/ })).toBeInTheDocument();
    expect(screen.queryByText("cmd_vel")).not.toBeInTheDocument();
  });

  it("uses the durable conversation lifecycle API for history management", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    const conversation = {
      conversationId: "conversation-managed",
      robotId: "robot-1",
      requestedBy: "developer-ui",
      locale: "zh-CN",
      title: "巡检会话",
      status: "idle",
      archived: false,
      deleted: false,
      createdAt: "2026-08-19T00:00:00Z",
      updatedAt: "2026-08-19T00:00:00Z",
    };
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/assistant/conversations") && !init?.method) {
        return Promise.resolve(response([conversation]));
      }
      if (url.endsWith("/assistant/conversations/conversation-managed") && init?.method === "PATCH") {
        const body = typeof init.body === "string" ? JSON.parse(init.body) : {};
        return Promise.resolve(response({ ...conversation, ...body, updatedAt: "2026-08-19T00:01:00Z" }));
      }
      return fallback!(input, init) as Promise<Response>;
    });
    vi.spyOn(window, "prompt").mockReturnValue("会议区巡检");

    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.click(screen.getByRole("button", { name: "任务历史" }));
    expect((await screen.findAllByText("巡检会话")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "重命名" }));

    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([input, init]) => (
      String(input).endsWith("/assistant/conversations/conversation-managed")
      && init?.method === "PATCH"
      && init.body === JSON.stringify({ title: "会议区巡检" })
    ))).toBe(true));
    expect((await screen.findAllByText("会议区巡检")).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "归档" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "删除" })).not.toBeDisabled();
  });

  it("waits for robot discovery before requesting robot-scoped endpoints", async () => {
    const requestedUrls: string[] = [];
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.endsWith("/api/v1/auth/me")) {
        return Promise.resolve(response({ principal: {
          userId: "developer-ui", username: "developer-ui", displayName: "Developer UI",
          role: "DEVELOPER", siteId: "default", authenticationMethod: "session",
          scopes: ["api:access", "robots:read", "assistant:use", "speech:use", "missions:use", "events:read", "commands:manage", "mappings:manage", "simulations:manage", "diagnostics:read"],
        } }));
      }
      if (url.endsWith("/robots")) {
        return Promise.resolve(response([{
          robotId: "g1-dev",
          status: "online",
          connected: true,
          safetyMode: "normal",
        }]));
      }
      if (url.includes("/robots/g1-dev/capabilities")) {
        return Promise.resolve(response({
          robotId: "g1-dev",
          agent: "robot-agent",
          protocol: "robot-agent.v1",
          capabilities: ["navigation.navigate_to_pose"],
          safetyMode: "normal",
        }));
      }
      if (url.includes("/robots/g1-dev/scene-binding")) {
        return Promise.resolve(response({ detail: "scene binding not found" }, { status: 404 }));
      }
      if (url.endsWith("/mappings") || url.endsWith("/events") || url.endsWith("/scenes")
        || url.endsWith("/assistant/conversations")) {
        return Promise.resolve(response([]));
      }
      return Promise.resolve(response({}));
    });

    render(<App />);

    await screen.findAllByText("g1-dev");
    await waitFor(() => expect(requestedUrls.some((url) =>
      url.includes("/robots/g1-dev/capabilities"))).toBe(true));
    expect(requestedUrls.some((url) => url.includes("/robots/robot-1/"))).toBe(false);
    expect(requestedUrls.some((url) => url.includes("/robots//"))).toBe(false);
  });

  it("turns natural language into a reviewable continuous Mission before explicit confirmation", async () => {
    render(<App />);
    await screen.findAllByText("robot-1");
    expect(screen.queryByRole("button", { name: /导入 PCD|添加场景/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /开始语音输入/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "按住确认并执行" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("任务描述"), {
      target: { value: "向前移动后打开夹爪" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送给 H-IDE" }));
    expect((await screen.findAllByText("向前移动后打开夹爪")).length).toBeGreaterThan(0);
    expect(screen.getByText("vision.observe@2.0.0")).toBeInTheDocument();
    expect(screen.getByText("navigation.navigate_to_pose@2.0.0")).toBeInTheDocument();
    expect(screen.getByText(/请确认周围环境安全/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "按住确认并执行" })).not.toBeDisabled();
    const requestsBeforeConfirmation = vi.mocked(fetch).mock.calls.map(([input]) => String(input));
    expect(requestsBeforeConfirmation.some((url) => url.endsWith("/api/v1/commands"))).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "按住确认并执行" }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).endsWith("/api/v2/missions/drafts/mission-ui/approve"))).toBe(true));
    expect(await screen.findByText(/Mission 已完成/)).toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).endsWith("/api/v2/missions/mission-ui/advance"))).toBe(true);
  });

  it("rejects a Mission draft without dispatching any command", async () => {
    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.change(screen.getByLabelText("任务描述"), { target: { value: "向前移动" } });
    fireEvent.click(screen.getByRole("button", { name: "发送给 H-IDE" }));
    await screen.findByRole("region", { name: "Mission V2 计划审核" });

    fireEvent.change(screen.getByLabelText(/审核备注/), { target: { value: "目标描述不清楚" } });
    fireEvent.click(screen.getByRole("button", { name: "拒绝" }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).endsWith("/api/v2/missions/drafts/mission-ui/reject"))).toBe(true));
    expect(screen.queryByRole("button", { name: "按住确认并执行" })).not.toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).endsWith("/api/v1/commands"))).toBe(false);
  });

  it("routes through ConversationGateway before loading a Mission V2 draft", async () => {
    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.change(screen.getByLabelText("任务描述"), { target: { value: "去厨房" } });
    fireEvent.click(screen.getByRole("button", { name: "发送给 H-IDE" }));
    await screen.findByRole("region", { name: "Mission V2 计划审核" });
    fireEvent.click(screen.getByRole("button", { name: "按住确认并执行" }));
    await screen.findByRole("region", { name: "连续 Mission 执行状态" });
    const urls = vi.mocked(fetch).mock.calls.map(([input]) => String(input));
    expect(urls.some((url) => url.endsWith("/api/v2/missions/drafts/mission-ui"))).toBe(true);
    expect(urls.some((url) => url.includes("/assistant/conversations/conversation-ui/messages"))).toBe(true);
    expect(vi.mocked(fetch).mock.calls.some(([input, init]) =>
      String(input).endsWith("/api/v2/missions/drafts") && init?.method === "POST")).toBe(false);
    expect(urls.some((url) => url.endsWith("/api/v1/commands"))).toBe(false);
  });

  it("replaces an optimistic running node with the durable Robot Agent rejection", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    let advanceCalls = 0;
    const executing = missionRecordFixture("executing");
    executing.snapshot.activeNodeId = "navigate-target";
    executing.snapshot.activeSkillRequestId = "skill-nav-ui-1";
    executing.nodeStates = executing.nodeStates.map((item) => item.nodeId === "observe-target"
      ? { ...item, status: "succeeded", attempt: 1 }
      : { ...item, status: "running", attempt: 1 });
    const rejected = missionRecordFixture("planning");
    rejected.snapshot.activeNodeId = "navigate-target";
    rejected.snapshot.errorCode = "ROBOT_AGENT_REJECTED";
    rejected.snapshot.detail = "action goal REJECTED";
    rejected.nodeStates = rejected.nodeStates.map((item) => item.nodeId === "observe-target"
      ? { ...item, status: "succeeded", attempt: 1 }
      : { ...item, status: "rejected", attempt: 1 });

    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v2/missions/mission-ui/advance")) {
        advanceCalls += 1;
        if (advanceCalls === 1) {
          return Promise.resolve(response({ mission: executing, actions: [], stoppedBecause: "executing" }));
        }
        return Promise.resolve(response({ detail: "reconciliation already in progress" }, { status: 409 }));
      }
      if (url.endsWith("/api/v2/missions/mission-ui") && !init?.method) {
        return Promise.resolve(response(rejected));
      }
      return fallback!(input, init) as Promise<Response>;
    });

    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.change(screen.getByLabelText("任务描述"), { target: { value: "去会议桌前站着" } });
    fireEvent.click(screen.getByRole("button", { name: "发送给 H-IDE" }));
    await screen.findByRole("region", { name: "Mission V2 计划审核" });
    fireEvent.click(screen.getByRole("button", { name: "按住确认并执行" }));

    expect(await screen.findByText("action goal REJECTED")).toBeInTheDocument();
    expect(screen.getAllByText("已拒绝").length).toBeGreaterThan(0);
    expect(vi.mocked(fetch).mock.calls.some(([input, init]) =>
      String(input).endsWith("/api/v2/missions/mission-ui") && !init?.method)).toBe(true);
  });

  it("cancels an active Mission through the Mission V2 safety endpoint", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v2/missions/mission-ui/advance")) {
        return Promise.resolve(response({
          mission: {
            ...missionRecordFixture("executing"),
            snapshot: {
              ...missionRecordFixture("executing").snapshot,
              activeSkillRequestId: "skill-nav-ui-1",
            },
          },
          actions: [{ kind: "skill_issued", requestId: "skill-nav-ui-1" }],
          stoppedBecause: "executing",
        }));
      }
      if (url.endsWith("/api/v2/missions/mission-ui/cancel")) {
        return Promise.resolve(response({
          mission: missionRecordFixture("cancelled"),
          receipt: { commandId: "skill-nav-ui-1", robotId: "robot-1", status: "cancelled" },
        }));
      }
      return fallback!(input, init) as Promise<Response>;
    });

    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.change(screen.getByLabelText("任务描述"), { target: { value: "去厨房" } });
    fireEvent.click(screen.getByRole("button", { name: "发送给 H-IDE" }));
    await screen.findByRole("region", { name: "Mission V2 计划审核" });
    fireEvent.click(screen.getByRole("button", { name: "按住确认并执行" }));

    const cancelButton = await screen.findByRole("button", { name: "取消 Mission" });
    fireEvent.click(cancelButton);

    expect(await screen.findByText("活动的 Nav2 命令已取消，Mission 已停止。")).toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([input]) =>
      String(input).endsWith("/api/v2/missions/mission-ui/cancel"))).toBe(true);
  });

  it("sends a voice transcript directly to the supervisor while keeping PlanDraft review", async () => {
    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.click(screen.getByRole("button", { name: "开始语音输入" }));
    expect(await screen.findByText("临时识别文本")).toBeInTheDocument();
    expect(screen.getByLabelText("任务描述")).toHaveValue("");
    fireEvent.click(await screen.findByRole("button", { name: "停止录音并发送" }));

    expect((await screen.findAllByText("语音要求机器人向前移动")).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("语音转写")).toBeInTheDocument();
    expect(screen.getByLabelText("任务描述")).toHaveValue("");
    expect(await screen.findByRole("region", { name: "Mission V2 计划审核" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "按住确认并执行" })).not.toBeDisabled();
    expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).includes("/speech/transcriptions"))).toBe(false);
    const voiceMessage = vi.mocked(fetch).mock.calls.find(([input]) =>
      String(input).includes("/assistant/conversations/conversation-ui/messages"));
    expect(JSON.parse(String(voiceMessage?.[1]?.body))).toMatchObject({
      modality: "voice",
      transcriptionId: "transcription-stream-ui",
    });
    expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).endsWith("/api/v2/missions/drafts/mission-ui/approve"))).toBe(false);
  });

  it("uses complete-WAV Whisper only after the developer explicitly enables the fallback", async () => {
    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.click(screen.getByRole("button", { name: "打开工具" }));
    fireEvent.click(screen.getByRole("checkbox", { name: /离线 WAV 转写降级/ }));
    fireEvent.click(screen.getByRole("button", { name: "开始语音输入" }));
    fireEvent.click(await screen.findByRole("button", { name: "停止录音并发送" }));

    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([input]) =>
      String(input).includes("/speech/transcriptions"))).toBe(true));
  });

  it("restores a durable Mission V2 from history after page reload", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v2/missions") && !init?.method) {
        return Promise.resolve(response([missionRecordFixture("completed")]));
      }
      if (url.endsWith("/api/v2/missions/drafts") && !init?.method) {
        return Promise.resolve(response([]));
      }
      return fallback!(input, init) as Promise<Response>;
    });
    window.history.replaceState({}, "", "/tasks/mission-ui");

    render(<App />);

    expect(await screen.findByRole("region", { name: "连续 Mission 执行状态" })).toBeInTheDocument();
    expect((await screen.findAllByText("向前移动后打开夹爪")).length).toBeGreaterThan(0);
    expect(screen.getByText(/已恢复连续 Mission/)).toBeInTheDocument();
  });

  it("shows Mission V2 visual evidence, map geometry and the no-motion boundary", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v2/missions") && !init?.method) {
        return Promise.resolve(response([missionVisualRecordFixture()]));
      }
      if (url.endsWith("/api/v2/missions/drafts") && !init?.method) {
        return Promise.resolve(response([]));
      }
      if (url.includes("/api/v2/missions/mission-ui/visual-evidence/")) {
        return Promise.resolve(new Response(new Blob(["jpeg"], { type: "image/jpeg" }), {
          status: 200,
          headers: { "content-type": "image/jpeg" },
        }));
      }
      return fallback!(input, init) as Promise<Response>;
    });
    window.history.replaceState({}, "", "/tasks/mission-ui");

    render(<App />);

    expect(await screen.findByRole("region", { name: "Mission 视觉结果" })).toBeInTheDocument();
    expect(screen.getByText("显示器")).toBeInTheDocument();
    expect(screen.getByText("3.38, -0.62, 1.19")).toBeInTheDocument();
    expect(screen.getByText("仅观察，未向 Nav2 发动作")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看视觉证据" }));
    expect(await screen.findByRole("dialog", { name: "视觉理解与空间依据" })).toBeInTheDocument();
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([input]) =>
      String(input).includes("/api/v2/missions/mission-ui/visual-evidence/analysis-visual-ui-1/"))).toBe(true));
  });

  it("shows interface center and protects activation behind validation/probe", async () => {
    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.click(screen.getByRole("button", { name: "打开工具" }));
    fireEvent.click(screen.getByRole("checkbox", { name: /工程模式/ }));
    fireEvent.click(screen.getByRole("button", { name: /接口中心/ }));
    expect(await screen.findByText("接口版本中心")).toBeInTheDocument();
    expect(screen.getByText("先校验并探测，再启用版本")).toBeInTheDocument();
  });

  it("answers social chat through ConversationGateway without creating a plan", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/assistant/turns/task-ui")) {
        return Promise.resolve(response({
          taskId: "task-ui",
          conversationId: "conversation-ui",
          status: "completed",
          result: {
            schemaVersion: "h-ide.assistant-result/2",
            intent: "SOCIAL_CHAT",
            route: "reply",
            routeStatus: "completed",
            assistantMessage: "你好！今天想聊点什么？",
          },
        }));
      }
      if (url.endsWith("/assistant/conversations/conversation-ui")) {
        return Promise.resolve(response({
          conversationId: "conversation-ui",
          robotId: "robot-1",
          requestedBy: "developer-ui",
          locale: "zh-CN",
          status: "idle",
          messages: [{
            messageId: "assistant-social-ui",
            taskId: "task-ui",
            role: "assistant",
            content: "你好！今天想聊点什么？",
            modality: "text",
            createdAt: new Date().toISOString(),
          }],
          turns: [],
          drafts: [],
        }));
      }
      return fallback!(input, init) as Promise<Response>;
    });

    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.change(screen.getByLabelText("任务描述"), { target: { value: "你好" } });
    fireEvent.click(screen.getByRole("button", { name: "发送给 H-IDE" }));

    expect(await screen.findByText("你好！今天想聊点什么？")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Mission V2 计划审核" })).not.toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([input, init]) =>
      String(input).endsWith("/api/v2/missions/drafts") && init?.method === "POST")).toBe(false);
  });

  it("shows proxied visual evidence for a read-only observation without a plan", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    const visualResult = {
      analysisId: "analysis-observe-ui",
      hypotheses: [{ label: "会议桌", score: 0.94 }],
      evidence: [{ kind: "rgb_keyframe", mediaType: "image/jpeg", sha256: "e".repeat(64) }],
    };
    const result = {
      schemaVersion: "h-ide.assistant-result/2",
      intent: "ROBOT_OBSERVE",
      route: "visual_observation",
      routeStatus: "completed",
      assistantMessage: "我看到了会议桌。",
      visualResult,
      evidence: [{
        kind: "rgb_keyframe",
        mediaType: "image/jpeg",
        sha256: "e".repeat(64),
        proxyUrl: `/api/v1/assistant/turns/task-ui/visual-evidence/analysis-observe-ui/${"e".repeat(64)}`,
      }],
    };
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/assistant/turns/task-ui/visual-evidence/")) {
        return Promise.resolve(new Response(new Blob(["jpeg"], { type: "image/jpeg" }), {
          status: 200,
          headers: { "content-type": "image/jpeg" },
        }));
      }
      if (url.includes("/assistant/turns/task-ui")) {
        return Promise.resolve(response({ taskId: "task-ui", conversationId: "conversation-ui", status: "completed", result }));
      }
      if (url.endsWith("/assistant/conversations/conversation-ui")) {
        return Promise.resolve(response({
          conversationId: "conversation-ui", robotId: "robot-1", requestedBy: "developer-ui",
          locale: "zh-CN", status: "idle", messages: [],
          turns: [{ taskId: "task-ui", conversationId: "conversation-ui", status: "completed", result }], drafts: [],
        }));
      }
      return fallback!(input, init) as Promise<Response>;
    });

    render(<App />);
    await screen.findAllByText("robot-1");
    fireEvent.click(screen.getByRole("button", { name: "用户" }));
    fireEvent.change(screen.getByLabelText("任务描述"), { target: { value: "只观察当前画面" } });
    fireEvent.click(screen.getByRole("button", { name: "发送给 H-IDE" }));

    expect(await screen.findByRole("region", { name: "视觉证据" })).toBeInTheDocument();
    expect(screen.getByText("识别到：会议桌")).toBeInTheDocument();
    expect(screen.queryByText("视觉处理详情")).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Mission V2 计划审核" })).not.toBeInTheDocument();
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([input]) =>
      String(input).includes("/assistant/turns/task-ui/visual-evidence/"))).toBe(true));
  });

  it("keeps a normal user out of the developer shell and hides technical product entry points", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    window.history.replaceState({}, "", "/developer/diagnostics");
    const requestedUrls: string[] = [];
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.endsWith("/api/v1/auth/me")) {
        return Promise.resolve(response({ principal: {
          userId: "user-ui", username: "user-ui", displayName: "普通用户",
          role: "USER", siteId: "default", authenticationMethod: "session",
          scopes: ["api:access", "robots:read", "assistant:use", "speech:use", "missions:use", "events:read"],
        } }));
      }
      return fallback!(input, init) as Promise<Response>;
    });

    render(<App />);
    expect(await screen.findByText("今天想和机器人做什么？")).toBeInTheDocument();
    await waitFor(() => expect(window.location.pathname).toBe("/app"));
    expect(screen.queryByRole("button", { name: "打开工具" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("开发者系统诊断")).not.toBeInTheDocument();
    expect(requestedUrls.some((url) => url.includes("/api/v2/developer/diagnostics"))).toBe(false);
    expect(screen.queryByRole("button", { name: /导入 PCD|添加场景/ })).not.toBeInTheDocument();
    expect(screen.queryByText("robot-1")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "对话机器人" })).toHaveTextContent("机器人 1");
  });

  it("opens scene teaching from the scoped user tool drawer without exposing developer tools", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    window.history.replaceState({}, "", "/app");
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/auth/me")) {
        return Promise.resolve(response({ principal: {
          userId: "scene-user-ui", username: "scene-user-ui", displayName: "巡检用户",
          role: "USER", siteId: "default", authenticationMethod: "session",
          scopes: ["api:access", "robots:read", "assistant:use", "scene-memory:use", "speech:use", "missions:use", "events:read"],
        } }));
      }
      if (url.endsWith("/api/v2/scene-memory/capabilities")) {
        return Promise.resolve(response({
          mode: "SCENE_TEACH",
          stateFlow: ["stopped", "captured", "named", "confirmed"],
          semanticProvider: "qwen",
          allowedSources: ["qwen_vlm", "rgbd", "tf"],
          retiredSources: ["pcd_annotation", "oracle", "foundationpose", "fake"],
          authority: "observation_only",
          executionEligible: false,
          navigationDispatch: "forbidden",
          humanConfirmationRequired: true,
        }));
      }
      if (url.includes("/api/v2/scene-memory/inspections?") || url.includes("/api/v2/scene-memory/landmarks?")) {
        return Promise.resolve(response([]));
      }
      return fallback!(input, init) as Promise<Response>;
    });

    render(<App />);
    await screen.findByText("今天想和机器人做什么？");
    fireEvent.click(screen.getByRole("button", { name: "打开工具" }));
    expect(screen.getByRole("button", { name: /巡检与记忆/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /手动控制|系统诊断|仿真观测台/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /巡检与记忆/ }));

    expect(await screen.findByRole("region", { name: "巡检与记忆" })).toBeInTheDocument();
    expect(window.location.pathname).toBe("/app/scene-memory");
    expect(screen.getByText(/浏览器不会伪造停稳位姿/)).toBeInTheDocument();
  });

  it("logs in with Cookie credentials and applies the CSRF token to the first write", async () => {
    const fallback = vi.mocked(fetch).getMockImplementation();
    window.history.replaceState({}, "", "/app");
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/auth/me")) {
        return Promise.resolve(response({ detail: "not authenticated" }, { status: 401 }));
      }
      if (url.endsWith("/api/v1/auth/login")) {
        return Promise.resolve(response({
          principal: {
            userId: "user-login-ui", username: "login-user", displayName: "登录用户",
            role: "USER", siteId: "default", authenticationMethod: "session",
            scopes: ["api:access", "robots:read", "assistant:use", "speech:use", "missions:use", "events:read"],
          },
          csrfToken: "csrf-ui-token",
        }));
      }
      if (url.includes("/assistant/turns/task-ui")) {
        return Promise.resolve(response({
          taskId: "task-ui", conversationId: "conversation-ui", status: "completed",
          result: { intent: "SOCIAL_CHAT", route: "reply", routeStatus: "completed", assistantMessage: "你好！" },
        }));
      }
      return fallback!(input, init) as Promise<Response>;
    });

    render(<App />);
    expect(await screen.findByRole("region", { name: "登录 H-IDE" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "注册" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "用户名" }), { target: { value: "login-user" } });
    fireEvent.change(screen.getByLabelText("密码"), { target: { value: "Strong-password-1" } });
    fireEvent.click(screen.getAllByRole("button", { name: "登录" }).at(-1)!);
    await screen.findByText("今天想和机器人做什么？");
    fireEvent.change(screen.getByLabelText("任务描述"), { target: { value: "你好" } });
    fireEvent.click(screen.getByRole("button", { name: "发送给 H-IDE" }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([input]) =>
      String(input).includes("/assistant/conversations/conversation-ui/messages"))).toBe(true));

    const loginCall = vi.mocked(fetch).mock.calls.find(([input]) => String(input).endsWith("/api/v1/auth/login"));
    expect(loginCall?.[1]?.credentials).toBe("include");
    const messageCall = vi.mocked(fetch).mock.calls.find(([input]) =>
      String(input).includes("/assistant/conversations/conversation-ui/messages"));
    const headers = new Headers(messageCall?.[1]?.headers);
    expect(headers.get("x-csrf-token")).toBe("csrf-ui-token");
    expect(headers.has("authorization")).toBe(false);
  });
});

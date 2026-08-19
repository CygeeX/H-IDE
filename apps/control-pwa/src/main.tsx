import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type {
  ApiError,
  AssistantConversation,
  AssistantDraft,
  AssistantMessage,
  AssistantTurn,
  CapabilitySnapshot,
  CommandReceipt,
  ControlEvent,
  MappingSnapshot,
  MissionAdvanceResultV2,
  MissionDraftRecordV2,
  MissionRecordV2,
  RobotState,
  SceneSummary,
  SpeechTranscription,
  VisualMissionRecord,
} from "./contracts";
import { isAssistantTurnTerminal, isTerminal, isVisualMissionPlan, statusLabel } from "./contracts";
import { BrowserWavRecorder } from "./audio";
import { SceneWorkspace, type ActiveSceneContext } from "./scene-workspace";
import { VisualMissionCard } from "./visual-mission-card";
import { SimulationWorkspace } from "./simulation-workspace";
import {
  latestMissionVisualResult,
  MissionVisualEvidenceDrawer,
  MissionVisualSummary,
} from "./mission-visual-evidence";
import "./app.css";
import "./chat-shell.css";

type ConnectionState = "connecting" | "live" | "offline";
type AppRoute =
  | { kind: "chat"; conversationId?: string }
  | { kind: "scene" }
  | { kind: "simulation" }
  | { kind: "interfaces" };

const API_BASE = (import.meta.env.VITE_CONTROL_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";
const API_TOKEN = (import.meta.env.VITE_CONTROL_API_TOKEN as string | undefined) ?? "dev-token";

class ApiFailure extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiFailure";
    this.status = status;
  }
}

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export function apiErrorDetail(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object" || !("detail" in body)) return fallback;
  const detail = (body as ApiError).detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.flatMap((item) => {
      if (!item || typeof item !== "object") return [];
      const value = item as { loc?: unknown; msg?: unknown };
      if (typeof value.msg !== "string") return [];
      const location = Array.isArray(value.loc)
        ? value.loc.filter((part) => part !== "body").map(String).join(".")
        : "";
      return [`${location ? `${location}: ` : ""}${value.msg}`];
    });
    if (messages.length) return messages.join("；");
  }
  return fallback;
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("accept", "application/json");
  if (init.body && !(init.body instanceof FormData) && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  if (API_TOKEN) headers.set("authorization", `Bearer ${API_TOKEN}`);
  const response = await fetch(apiUrl(path), { ...init, headers });
  const text = await response.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!response.ok) {
    const detail = typeof body === "string"
      ? body
      : apiErrorDetail(body, response.statusText);
    throw new ApiFailure(detail || `请求失败 (${response.status})`, response.status);
  }
  return body as T;
}

async function apiFetchBlob(path: string): Promise<Blob> {
  const headers = new Headers({ accept: "image/jpeg,image/png,application/octet-stream" });
  if (API_TOKEN) headers.set("authorization", `Bearer ${API_TOKEN}`);
  const response = await fetch(apiUrl(path), { headers, cache: "no-store" });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = apiErrorDetail(await response.json(), detail);
    } catch {
      // A binary proxy error may not contain JSON.
    }
    throw new ApiFailure(detail || `证据请求失败 (${response.status})`, response.status);
  }
  const blob = await response.blob();
  if (blob.size > 8 * 1024 * 1024) throw new Error("视觉证据超过 8 MiB 安全上限");
  return blob;
}

function makeId(prefix: string): string {
  const random = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}-${random}`;
}

function wsUrl(after: number): string {
  if (API_BASE) {
    const parsed = new URL(API_BASE, window.location.href);
    parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
    parsed.pathname = `${parsed.pathname.replace(/\/$/, "")}/api/v1/events`;
    parsed.search = new URLSearchParams({ after: String(after), token: API_TOKEN }).toString();
    return parsed.toString();
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/v1/events?after=${after}&token=${encodeURIComponent(API_TOKEN)}`;
}

function friendlyError(error: unknown): string {
  if (error instanceof ApiFailure) {
    const normalized = error.message.toLowerCase();
    if (normalized.includes("assistant plan has expired") || normalized.includes("status=expired")) {
      return "计划已过期，未发送任何机器人命令。请根据原任务重新规划后再审核。";
    }
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return "请求失败，请检查工作站连接";
}

function isDraftExpired(draft: AssistantDraft | null, now = Date.now()): boolean {
  if (!draft) return false;
  if (draft.status === "expired") return true;
  if (draft.status !== "draft") return false;
  if (!draft.plan.expiresAt) return false;
  const expiresAt = new Date(draft.plan.expiresAt).valueOf();
  return Number.isFinite(expiresAt) && expiresAt <= now;
}

function isDraftReviewableForRobot(
  draft: AssistantDraft | null,
  selectedRobotId: string,
  busy: boolean,
): draft is AssistantDraft {
  return Boolean(
    draft
    && !busy
    && draft.status === "draft"
    && !isDraftExpired(draft)
    && draft.plan.robotId === selectedRobotId,
  );
}

function preferredRobotId(robots: RobotState[], current: string): string {
  if (current && robots.some((robot) => robot.robotId === current)) return current;
  const preferred = robots.find((robot) => robot.connected && robot.status === "online")
    ?? robots.find((robot) => robot.connected)
    ?? robots[0];
  return preferred?.robotId ?? "";
}

function navigationGoal(payload: Record<string, unknown>): { x: number; y: number; z: number; yaw: number } | null {
  const value = payload.goalPose;
  if (!value || typeof value !== "object") return null;
  const pose = value as Record<string, unknown>;
  const numbers = [pose.x, pose.y, pose.z, pose.yaw];
  if (!numbers.every((item) => typeof item === "number" && Number.isFinite(item))) return null;
  return { x: pose.x as number, y: pose.y as number, z: pose.z as number, yaw: pose.yaw as number };
}

export function shouldReconcileMission(status: string | undefined): boolean {
  return [
    "ready",
    "planning",
    "observing",
    "dispatching",
    "executing",
    "verifying",
  ].includes(status ?? "");
}

function StatusPill({ value, tone }: { value: string | undefined; tone?: string }) {
  const normalized = (value ?? "unknown").toLowerCase().replace(/[^a-z0-9-]/g, "-");
  return <span className={`status-pill ${tone ?? normalized}`}>{statusLabel(value)}</span>;
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="empty-state">{children}</div>;
}

function parseAppRoute(pathname: string): AppRoute {
  if (pathname === "/scenes" || pathname.startsWith("/scenes/")) return { kind: "scene" };
  if (pathname === "/simulation" || pathname.startsWith("/simulation/")) return { kind: "simulation" };
  if (pathname === "/interfaces" || pathname.startsWith("/interfaces/")) return { kind: "interfaces" };
  const task = pathname.match(/^\/tasks\/([^/]+)$/);
  if (task) return { kind: "chat", conversationId: decodeURIComponent(task[1]) };
  return { kind: "chat" };
}

function useAppRoute(): [AppRoute, (path: string, options?: { replace?: boolean }) => void] {
  const [route, setRoute] = useState<AppRoute>(() => parseAppRoute(window.location.pathname));
  useEffect(() => {
    const onPopState = () => setRoute(parseAppRoute(window.location.pathname));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);
  const navigate = useCallback((path: string, options?: { replace?: boolean }) => {
    if (options?.replace) window.history.replaceState({}, "", path);
    else window.history.pushState({}, "", path);
    setRoute(parseAppRoute(path));
  }, []);
  return [route, navigate];
}

function App() {
  const [route, navigate] = useAppRoute();
  const [robots, setRobots] = useState<RobotState[]>([]);
  const [selectedRobotId, setSelectedRobotId] = useState("");
  const [capabilities, setCapabilities] = useState<CapabilitySnapshot | null>(null);
  const [events, setEvents] = useState<ControlEvent[]>([]);
  const [mappings, setMappings] = useState<MappingSnapshot[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [lastReceipt, setLastReceipt] = useState<CommandReceipt | null>(null);
  const [lastCommandId, setLastCommandId] = useState<string | null>(null);
  const [assistantConversations, setAssistantConversations] = useState<AssistantConversation[]>([]);
  const [assistantConversation, setAssistantConversation] = useState<AssistantConversation | null>(null);
  const [assistantText, setAssistantText] = useState("");
  const [assistantTurn, setAssistantTurn] = useState<AssistantTurn | null>(null);
  const [assistantBusy, setAssistantBusy] = useState(false);
  const [transcriptionId, setTranscriptionId] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const [reviewFeedback, setReviewFeedback] = useState("");
  const [draft, setDraft] = useState<AssistantDraft | null>(null);
  const [missionDraft, setMissionDraft] = useState<MissionDraftRecordV2 | null>(null);
  const [missionRecord, setMissionRecord] = useState<MissionRecordV2 | null>(null);
  const [missionMessages, setMissionMessages] = useState<AssistantMessage[]>([]);
  const [missionHistory, setMissionHistory] = useState<MissionRecordV2[]>([]);
  const [missionDraftHistory, setMissionDraftHistory] = useState<MissionDraftRecordV2[]>([]);
  const [missionHistoryLoaded, setMissionHistoryLoaded] = useState(false);
  const [visualMission, setVisualMission] = useState<VisualMissionRecord | null>(null);
  const [activeSceneContext, setActiveSceneContext] = useState<ActiveSceneContext | null>(null);
  const [pendingSceneFile, setPendingSceneFile] = useState<File | null>(null);
  const [toolMenuOpen, setToolMenuOpen] = useState(false);
  const [controlPanelOpen, setControlPanelOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [advancedMode, setAdvancedMode] = useState(() => {
    try {
      return window.localStorage.getItem("h-ide-advanced-mode") === "true";
    } catch {
      return false;
    }
  });
  const [mappingJson, setMappingJson] = useState(
    JSON.stringify({
      mappingId: "ui-map-1",
      version: "v1",
      commandKey: "move.forward",
      transport: "fake",
      targetId: "robot-1",
      mapping: { targetId: "robot-1", topic: "/move.forward" },
    }, null, 2),
  );
  const [mappingMessage, setMappingMessage] = useState("");
  const [lastEventId, setLastEventId] = useState(0);
  const reconnectTimer = useRef<number | undefined>(undefined);
  const socketRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<BrowserWavRecorder | null>(null);
  const restoredConversationRef = useRef(false);
  const restoredMissionRef = useRef(false);
  const assistantConversationRef = useRef<AssistantConversation | null>(assistantConversation);
  assistantConversationRef.current = assistantConversation;
  const selectedRobotRef = useRef(selectedRobotId);
  selectedRobotRef.current = selectedRobotId;

  const selectedRobot = useMemo(
    () => robots.find((robot) => robot.robotId === selectedRobotId) ?? null,
    [robots, selectedRobotId],
  );

  const loadRobots = useCallback(async () => {
    const list = await apiFetch<RobotState[]>("/api/v1/robots");
    setRobots(list);
    setSelectedRobotId((current) => preferredRobotId(list, current));
  }, []);

  const loadMappings = useCallback(async () => {
    setMappings(await apiFetch<MappingSnapshot[]>("/api/v1/mappings"));
  }, []);

  const loadCapabilities = useCallback(async (robotId: string) => {
    try {
      setCapabilities(await apiFetch<CapabilitySnapshot>(`/api/v1/robots/${encodeURIComponent(robotId)}/capabilities`));
    } catch {
      setCapabilities(null);
    }
  }, []);

  const applyConversationDetail = useCallback((detail: AssistantConversation) => {
    setSelectedRobotId(detail.robotId);
    setAssistantConversation(detail);
    const latestTurn = detail.turns?.at(-1) ?? null;
    const drafts = [...(detail.drafts ?? [])].reverse();
    const latestDraft = drafts.find((item) => item.status === "draft") ?? drafts[0] ?? null;
    setAssistantTurn(latestTurn);
    setDraft(latestDraft);
    if (!latestDraft || !isVisualMissionPlan(latestDraft.plan) || latestDraft.status === "draft") {
      setVisualMission(null);
    }
  }, []);

  const loadVisualMission = useCallback(async (missionId: string) => {
    const mission = await apiFetch<VisualMissionRecord>(
      `/api/v1/visual-missions/${encodeURIComponent(missionId)}`,
    );
    setVisualMission(mission);
    return mission;
  }, []);

  const loadConversationSummaries = useCallback(async () => {
    const conversations = await apiFetch<AssistantConversation[]>("/api/v1/assistant/conversations");
    setAssistantConversations(conversations);
    return conversations;
  }, []);

  const loadMissionHistory = useCallback(async () => {
    try {
      const [missions, drafts] = await Promise.all([
        apiFetch<MissionRecordV2[]>("/api/v2/missions"),
        apiFetch<MissionDraftRecordV2[]>("/api/v2/missions/drafts"),
      ]);
      setMissionHistory(Array.isArray(missions) ? missions : []);
      setMissionDraftHistory(Array.isArray(drafts) ? drafts : []);
    } finally {
      setMissionHistoryLoaded(true);
    }
  }, []);

  const loadConversationDetail = useCallback(async (conversationId: string) => {
    const detail = await apiFetch<AssistantConversation>(
      `/api/v1/assistant/conversations/${encodeURIComponent(conversationId)}`,
    );
    applyConversationDetail(detail);
    const drafts = [...(detail.drafts ?? [])].reverse();
    const latestDraft = drafts.find((item) => item.status === "draft") ?? drafts[0] ?? null;
    if (latestDraft && isVisualMissionPlan(latestDraft.plan) && latestDraft.status !== "draft") {
      await loadVisualMission(latestDraft.plan.mission.missionId).catch(() => setVisualMission(null));
    }
    return detail;
  }, [applyConversationDetail, loadVisualMission]);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      loadRobots(),
      loadMappings(),
      apiFetch<ControlEvent[]>("/api/v1/events"),
      loadMissionHistory(),
    ])
      .then(([, , initialEvents]) => {
        if (cancelled) return;
        setEvents(initialEvents);
        const last = initialEvents.reduce((max, event) => Math.max(max, Number(event.eventId ?? 0)), 0);
        setLastEventId(last);
        setError("");
      })
      .catch((reason) => {
        if (!cancelled) setError(friendlyError(reason));
      });
    return () => { cancelled = true; };
  }, [loadMappings, loadMissionHistory, loadRobots]);

  useEffect(() => {
    if (!selectedRobot) {
      setCapabilities(null);
      return;
    }
    void loadCapabilities(selectedRobot.robotId);
  }, [loadCapabilities, selectedRobot?.robotId]);

  useEffect(() => {
    let cancelled = false;
    setActiveSceneContext(null);
    if (!selectedRobot) return undefined;
    void Promise.all([
      apiFetch<{ sceneId: string; revision: number }>(`/api/v1/robots/${encodeURIComponent(selectedRobot.robotId)}/scene-binding`),
      apiFetch<SceneSummary[]>("/api/v1/scenes"),
    ]).then(([binding, scenes]) => {
      if (cancelled) return;
      const scene = scenes.find((item) => item.sceneId === binding.sceneId);
      const revision = scene?.revisions.find((item) => item.revision === binding.revision);
      if (!revision?.boundsMap || revision.status !== "published") return;
      setActiveSceneContext({
        sceneId: binding.sceneId,
        revision: binding.revision,
        bounds: revision.boundsMap,
        annotations: revision.annotations,
        currentPose: selectedRobot.pose,
      });
    }).catch(() => {
      // Scene context is optional for non-spatial tasks. The scene workspace
      // exposes any actionable catalog or binding error when the user opens it.
    });
    return () => { cancelled = true; };
  }, [selectedRobot?.robotId, selectedRobot?.pose]);

  useEffect(() => {
    if (!missionHistoryLoaded) return undefined;
    let cancelled = false;
    void loadConversationSummaries()
      .then((conversations) => {
        if (cancelled || restoredConversationRef.current) return;
        restoredConversationRef.current = true;
        const initialRoute = parseAppRoute(window.location.pathname);
        if (initialRoute.kind !== "chat" || initialRoute.conversationId) return;
        if (
          missionHistory.some(
            (item) => !["completed", "failed", "cancelled"].includes(item.snapshot.status),
          )
          || missionDraftHistory.some((item) => item.status === "draft")
        ) return;
        const active = conversations.find((item) => ["planning", "review", "running"].includes(item.status));
        if (active) navigate(`/tasks/${encodeURIComponent(active.conversationId)}`, { replace: true });
      })
      .catch((reason) => {
        if (!cancelled) setError(friendlyError(reason));
      });
    return () => { cancelled = true; };
  }, [
    loadConversationSummaries,
    missionDraftHistory,
    missionHistory,
    missionHistoryLoaded,
    navigate,
  ]);

  useEffect(() => {
    if (!missionHistoryLoaded || restoredMissionRef.current) return;
    restoredMissionRef.current = true;
    const initialRoute = parseAppRoute(window.location.pathname);
    if (initialRoute.kind !== "chat" || initialRoute.conversationId) return;
    const activeMission = missionHistory.find(
      (item) => !["completed", "failed", "cancelled"].includes(item.snapshot.status),
    );
    const activeDraft = missionDraftHistory.find((item) => item.status === "draft");
    const identifier = activeMission?.missionId ?? activeDraft?.draftId;
    if (identifier) navigate(`/tasks/${encodeURIComponent(identifier)}`, { replace: true });
  }, [missionDraftHistory, missionHistory, missionHistoryLoaded, navigate]);

  useEffect(() => {
    if (route.kind !== "chat") return;
    if (!route.conversationId) {
      setAssistantConversation(null);
      setAssistantTurn(null);
      setDraft(null);
      setVisualMission(null);
      setMissionDraft(null);
      setMissionRecord(null);
      setMissionMessages([]);
      setReviewFeedback("");
      return;
    }
    if (!missionHistoryLoaded) return;
    if (
      missionRecord?.missionId === route.conversationId
      || missionDraft?.draftId === route.conversationId
    ) return;
    const historicalMission = missionHistory.find(
      (item) => item.missionId === route.conversationId,
    );
    const historicalDraft = missionDraftHistory.find(
      (item) => item.draftId === route.conversationId,
    );
    if (historicalMission || historicalDraft) {
      const plan = historicalMission?.plan ?? historicalDraft?.plan;
      if (!plan) return;
      setSelectedRobotId(plan.robotId);
      setAssistantConversation(null);
      setAssistantTurn(null);
      setDraft(null);
      setVisualMission(null);
      setMissionDraft(historicalDraft ?? null);
      setMissionRecord(historicalMission ?? null);
      setMissionMessages([
        {
          messageId: `mission-history-user-${plan.planId}`,
          role: "user",
          content: plan.objective,
          modality: "text",
          createdAt: plan.createdAt,
        },
        {
          messageId: `mission-history-assistant-${plan.planId}`,
          role: "assistant",
          content: historicalMission
            ? `已恢复连续 Mission，当前状态：${statusLabel(historicalMission.snapshot.status)}。`
            : `已恢复 Mission 草案，当前状态：${statusLabel(historicalDraft?.status)}。`,
          modality: "text",
          createdAt: historicalMission?.snapshot.updatedAt ?? historicalDraft?.updatedAt ?? plan.createdAt,
        },
      ]);
      return;
    }
    if (assistantConversationRef.current?.conversationId === route.conversationId) return;
    void loadConversationDetail(route.conversationId).catch((reason) => setError(friendlyError(reason)));
  }, [
    loadConversationDetail,
    missionDraft,
    missionDraftHistory,
    missionHistory,
    missionHistoryLoaded,
    missionRecord,
    route,
  ]);

  useEffect(() => () => {
    void recorderRef.current?.cancel();
    recorderRef.current = null;
  }, []);

  useEffect(() => {
    let disposed = false;
    let retry = 0;

    const connect = () => {
      if (disposed || typeof WebSocket === "undefined") {
        setConnection("offline");
        return;
      }
      setConnection(retry ? "offline" : "connecting");
      const socket = new WebSocket(wsUrl(lastEventId));
      socketRef.current = socket;
      socket.onopen = () => {
        retry = 0;
        setConnection("live");
      };
      socket.onmessage = (message) => {
        try {
          const event = JSON.parse(String(message.data)) as ControlEvent;
          setEvents((current) => {
            if (event.eventId && current.some((item) => item.eventId === event.eventId)) return current;
            return [...current, event].slice(-120);
          });
          if (event.eventId) setLastEventId((current) => Math.max(current, event.eventId ?? 0));
          if (event.commandId && event.status) {
            setLastReceipt((current) => current && current.commandId === event.commandId
              ? { ...current, status: event.status as CommandReceipt["status"], error: event.error as string | undefined }
              : current);
          }
        } catch {
          // Ignore malformed event frames; the next persisted event remains replayable.
        }
      };
      socket.onerror = () => setConnection("offline");
      socket.onclose = () => {
        if (disposed) return;
        setConnection("offline");
        retry = Math.min(retry + 1, 5);
        reconnectTimer.current = window.setTimeout(connect, Math.min(1000 * 2 ** retry, 8000));
      };
    };
    connect();
    return () => {
      disposed = true;
      if (reconnectTimer.current !== undefined) window.clearTimeout(reconnectTimer.current);
      socketRef.current?.close();
      socketRef.current = null;
    };
    // The socket itself tracks the cursor. Reconnecting on every event would
    // lose the benefit of a stable connection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runRequest = useCallback(async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (reason) {
      setError(friendlyError(reason));
    } finally {
      setBusy(false);
    }
  }, []);

  const submitCommand = useCallback((commandKey: string, payload: Record<string, unknown> = {}) => {
    void runRequest(async () => {
      const receipt = await apiFetch<CommandReceipt>("/api/v1/commands", {
        method: "POST",
        body: JSON.stringify({
          commandId: makeId("cmd"),
          idempotencyKey: makeId("idem"),
          robotId: selectedRobotId,
          commandKey,
          payload,
        }),
      });
      setLastReceipt(receipt);
      setLastCommandId(receipt.commandId);
      setNotice(`${commandKey}：${statusLabel(receipt.status)}`);
    });
  }, [runRequest, selectedRobotId]);

  const cancelLast = useCallback(() => {
    if (!lastCommandId) return;
    void runRequest(async () => {
      const receipt = await apiFetch<CommandReceipt>(`/api/v1/commands/${encodeURIComponent(lastCommandId)}/cancel`, { method: "POST" });
      setLastReceipt(receipt);
      setNotice(`命令 ${lastCommandId}：${statusLabel(receipt.status)}`);
    });
  }, [lastCommandId, runRequest]);

  const ensureAssistantConversation = useCallback(async (robotId: string) => {
    if (assistantConversation?.robotId === robotId) return assistantConversation;
    const created = await apiFetch<AssistantConversation>("/api/v1/assistant/conversations", {
      method: "POST",
      body: JSON.stringify({ robotId, requestedBy: "operator", locale: "zh-CN" }),
    });
    const conversation = { ...created, messages: [], turns: [], drafts: [] };
    if (selectedRobotRef.current === robotId) {
      assistantConversationRef.current = conversation;
      setAssistantConversation(conversation);
      setAssistantConversations((current) => [conversation, ...current.filter((item) => item.conversationId !== conversation.conversationId)]);
      navigate(`/tasks/${encodeURIComponent(conversation.conversationId)}`);
    }
    return conversation;
  }, [assistantConversation, navigate]);

  const refreshConversation = useCallback(async (conversationId: string, robotId: string) => {
    const detail = await apiFetch<AssistantConversation>(
      `/api/v1/assistant/conversations/${encodeURIComponent(conversationId)}`,
    );
    if (detail?.conversationId && detail.robotId === robotId && selectedRobotRef.current === robotId) {
      setAssistantConversation((current) => {
        if (current?.conversationId !== detail.conversationId) return detail;
        const messages = [...(detail.messages ?? [])];
        for (const local of current.messages ?? []) {
          const persisted = messages.some((message) =>
            message.role === local.role
            && message.content === local.content
            && (message.taskId === local.taskId || !message.taskId || !local.taskId));
          if (!persisted) messages.push(local);
        }
        messages.sort((left, right) => left.createdAt.localeCompare(right.createdAt));
        return { ...detail, messages };
      });
      setAssistantTurn(detail.turns?.at(-1) ?? null);
      const drafts = [...(detail.drafts ?? [])].reverse();
      const latestDraft = drafts.find((item) => item.status === "draft") ?? drafts[0] ?? null;
      setDraft((current) => current && latestDraft && current.draftId === latestDraft.draftId && current.status === latestDraft.status
        ? current
        : latestDraft);
      setAssistantConversations((current) => [detail, ...current.filter((item) => item.conversationId !== detail.conversationId)]);
    }
  }, []);

  const monitorAssistantTurn = useCallback(async (taskId: string, robotId: string) => {
    for (let attempt = 0; attempt < 250; attempt += 1) {
      const turn = await apiFetch<AssistantTurn>(`/api/v1/assistant/turns/${encodeURIComponent(taskId)}`);
      if (selectedRobotRef.current !== robotId) return turn;
      setAssistantTurn(turn);
      if (isAssistantTurnTerminal(turn.status)) {
        if (turn.draft) {
          setDraft(turn.draft);
          setNotice("多 Agent 计划已生成，等待人工审核");
        } else if (turn.status === "needs_input") {
          setNotice(turn.result?.assistantMessage ?? "调度 Agent 需要补充信息");
        } else if (turn.status === "cannot_plan") {
          setError(turn.result?.assistantMessage ?? "当前能力不足，无法规划任务");
        } else if (["failed", "unknown"].includes(turn.status)) {
          setError(turn.error ?? "Agent 规划失败");
        }
        try {
          await refreshConversation(turn.conversationId, robotId);
        } catch {
          // The terminal turn and immutable draft are already available even
          // if a follow-up conversation refresh loses connectivity.
        }
        return turn;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
    throw new Error("Agent 规划超时，可取消后重新提交");
  }, [refreshConversation]);

  const sendAssistantMessage = useCallback(async (
    rawText: string,
    modality: "text" | "voice" = "text",
    sourceTranscriptionId: string | null = null,
  ) => {
    const text = rawText.trim();
    if (!text || assistantBusy) return;
    if (connection !== "live") {
      setError("工作站当前离线。任务不会排队发送，请恢复连接后重新提交。");
      return;
    }
    setAssistantBusy(true);
    setError("");
    try {
      const robotId = selectedRobotId;
      const conversation = await ensureAssistantConversation(robotId);
      if (selectedRobotRef.current !== robotId) {
        throw new Error("机器人选择已变化，请重新提交任务");
      }
      const accepted = await apiFetch<{ taskId: string; conversationId: string; status: string }>(
        `/api/v1/assistant/conversations/${encodeURIComponent(conversation.conversationId)}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            text,
            modality,
            transcriptionId: sourceTranscriptionId,
            sceneSelection: activeSceneContext ? {
              sceneRef: {
                sceneId: activeSceneContext.sceneId,
                revision: activeSceneContext.revision,
              },
              selectedPoint: activeSceneContext.selectedPoint,
            } : undefined,
          }),
        },
      );
      setAssistantText("");
      setTranscriptionId(null);
      setAssistantTurn({
        taskId: accepted.taskId,
        conversationId: accepted.conversationId,
        status: accepted.status,
      });
      setDraft(null);
      setReviewFeedback("");
      setAssistantConversation((current) => current ? {
        ...current,
        status: "planning",
        messages: [
          ...(current.messages ?? []),
          {
            messageId: makeId("message-local"),
            taskId: accepted.taskId,
            role: "user",
            content: text,
            modality,
            transcriptionId: sourceTranscriptionId,
            createdAt: new Date().toISOString(),
          },
        ],
      } : current);
      setAssistantConversations((current) => current.map((item) => item.conversationId === conversation.conversationId
        ? { ...item, status: "planning", updatedAt: new Date().toISOString() }
        : item));
      await monitorAssistantTurn(accepted.taskId, conversation.robotId);
    } catch (reason) {
      setError(friendlyError(reason));
    } finally {
      setAssistantBusy(false);
    }
  }, [activeSceneContext, assistantBusy, connection, ensureAssistantConversation, monitorAssistantTurn, selectedRobotId]);

  const advanceMission = useCallback(async (missionId: string) => {
    const advanced = await apiFetch<MissionAdvanceResultV2>(
      `/api/v2/missions/${encodeURIComponent(missionId)}/advance`,
      { method: "POST", body: JSON.stringify({ maximumSteps: 20 }) },
    );
    setMissionRecord(advanced.mission);
    setMissionHistory((current) => [
      advanced.mission,
      ...current.filter((item) => item.missionId !== advanced.mission.missionId),
    ]);
    const status = advanced.mission.snapshot.status;
    const message = status === "completed"
      ? "任务已完成。视觉模型只提供语义假设，发送给 Robot Agent 的目标点来自 RGB-D / 场景几何事实。"
      : status === "needs_input"
        ? advanced.mission.snapshot.detail ?? "任务需要补充信息。"
        : status === "awaiting_review"
          ? "持续规划产生了超出自动修订策略的变化，需要再次审核。"
          : `任务已推进到：${statusLabel(status)}。`;
    setMissionMessages((current) => [
      ...current,
      {
        messageId: makeId("mission-message"),
        role: "assistant",
        content: message,
        modality: "text",
        createdAt: new Date().toISOString(),
      },
    ]);
    return advanced;
  }, []);

  const sendMissionMessage = useCallback(async (
    rawText: string,
    modality: "text" | "voice" = "text",
    sourceTranscriptionId: string | null = null,
  ) => {
    const text = rawText.trim();
    if (!text || assistantBusy) return;
    if (connection !== "live") {
      setError("工作站当前离线。任务不会排队发送，请恢复连接后重新提交。");
      return;
    }
    setAssistantBusy(true);
    setError("");
    setAssistantText("");
    setTranscriptionId(null);
    const userMessage: AssistantMessage = {
      messageId: makeId("mission-message"),
      role: "user",
      content: text,
      modality,
      transcriptionId: sourceTranscriptionId,
      createdAt: new Date().toISOString(),
    };
    setMissionMessages((current) => [...current, userMessage]);
    try {
      const activeMission = missionRecord
        && !["completed", "failed", "cancelled"].includes(missionRecord.snapshot.status)
        ? missionRecord
        : null;
      if (activeMission) {
        const response = await apiFetch<{
          decision: { kind?: string; reason?: string };
          mission: MissionRecordV2;
        }>(`/api/v2/missions/${encodeURIComponent(activeMission.missionId)}/messages`, {
          method: "POST",
          body: JSON.stringify({
            text,
            requestedBy: "operator",
            kind: activeMission.snapshot.status === "needs_input" ? "answer" : "modify_goal",
          }),
        });
        setMissionRecord(response.mission);
        setMissionHistory((current) => [
          response.mission,
          ...current.filter((item) => item.missionId !== response.mission.missionId),
        ]);
        setMissionMessages((current) => [
          ...current,
          {
            messageId: makeId("mission-message"),
            role: "assistant",
            content: response.decision?.kind === "replan"
              ? "已结合你的补充生成持续任务修订；请审核变化后继续。"
              : response.decision?.reason ?? "已记录补充信息，等待人工处理。",
            modality: "text",
            createdAt: new Date().toISOString(),
          },
        ]);
        return;
      }
      const created = await apiFetch<MissionDraftRecordV2>("/api/v2/missions/drafts", {
        method: "POST",
        body: JSON.stringify({
          requestId: makeId("mission-request"),
          robotId: selectedRobotId,
          objective: text,
          requestedBy: "operator",
          locale: "zh-CN",
          modality,
          sceneRefs: activeSceneContext ? [{
            sceneId: activeSceneContext.sceneId,
            revision: activeSceneContext.revision,
          }] : [],
        }),
      });
      if (
        !created
        || typeof created !== "object"
        || !created.plan
        || !Array.isArray(created.plan.nodes)
      ) {
        throw new Error("Mission Supervisor 返回了无效计划草案");
      }
      const nodeCount = created.plan.nodes.length;
      setMissionDraft(created);
      setMissionDraftHistory((current) => [
        created,
        ...current.filter((item) => item.draftId !== created.draftId),
      ]);
      setMissionRecord(null);
      setReviewFeedback("");
      setMissionMessages((current) => [
        ...current,
        {
          messageId: makeId("mission-message"),
          role: "assistant",
          content: `已生成连续 Mission 计划，共 ${nodeCount} 个语义节点。执行前请审核任务边界与每个技能。`,
          modality: "text",
          createdAt: new Date().toISOString(),
        },
      ]);
      setNotice("Mission 计划已生成，等待人工审核");
      navigate(`/tasks/${encodeURIComponent(created.draftId)}`);
    } catch (reason) {
      setError(friendlyError(reason));
    } finally {
      setAssistantBusy(false);
    }
  }, [activeSceneContext, assistantBusy, connection, missionRecord, navigate, selectedRobotId]);

  const submitAssistantMessage = useCallback((event: FormEvent) => {
    event.preventDefault();
    void sendMissionMessage(assistantText, "text", null);
  }, [assistantText, sendMissionMessage]);

  const confirmMissionDraft = useCallback(() => {
    if (!missionDraft || missionDraft.status !== "draft") return;
    if (missionDraft.robotId !== selectedRobotId) {
      setError("Mission 草案目标机器人与当前选择不一致");
      return;
    }
    void (async () => {
      setAssistantBusy(true);
      setError("");
      try {
        const approved = await apiFetch<MissionRecordV2>(
          `/api/v2/missions/drafts/${encodeURIComponent(missionDraft.draftId)}/approve`,
          {
            method: "POST",
            body: JSON.stringify({ approvedBy: "operator" }),
          },
        );
        setMissionDraft((current) => current ? { ...current, status: "approved" } : current);
        setMissionRecord(approved);
        setMissionHistory((current) => [
          approved,
          ...current.filter((item) => item.missionId !== approved.missionId),
        ]);
        setMissionDraftHistory((current) => current.map((item) => (
          item.draftId === missionDraft.draftId ? { ...item, status: "approved" } : item
        )));
        setNotice("Mission 已审核，开始连续执行");
        await advanceMission(approved.missionId);
        await loadRobots();
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [advanceMission, loadRobots, missionDraft, selectedRobotId]);

  const rejectMissionDraft = useCallback(() => {
    if (!missionDraft || missionDraft.status !== "draft") return;
    void (async () => {
      setAssistantBusy(true);
      setError("");
      try {
        const rejected = await apiFetch<MissionDraftRecordV2>(
          `/api/v2/missions/drafts/${encodeURIComponent(missionDraft.draftId)}/reject`,
          {
            method: "POST",
            body: JSON.stringify({ reason: reviewFeedback.trim() || "人工审核拒绝" }),
          },
        );
        setMissionDraft(rejected);
        setMissionDraftHistory((current) => [
          rejected,
          ...current.filter((item) => item.draftId !== rejected.draftId),
        ]);
        setNotice("Mission 草案已拒绝，未发送任何物理动作");
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [missionDraft, reviewFeedback]);

  const continueMission = useCallback(() => {
    if (!missionRecord) return;
    void (async () => {
      setAssistantBusy(true);
      setError("");
      try {
        await advanceMission(missionRecord.missionId);
        await loadRobots();
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [advanceMission, loadRobots, missionRecord]);

  const cancelMission = useCallback(() => {
    if (!missionRecord || ["completed", "failed", "cancelled"].includes(missionRecord.snapshot.status)) return;
    void (async () => {
      setAssistantBusy(true);
      setError("");
      try {
        const cancelled = await apiFetch<{
          mission: MissionRecordV2;
          receipt?: CommandReceipt | null;
        }>(`/api/v2/missions/${encodeURIComponent(missionRecord.missionId)}/cancel`, {
          method: "POST",
          body: JSON.stringify({
            requestedBy: "operator",
            reason: "操作员从 H-IDE 取消 Mission",
          }),
        });
        setMissionRecord(cancelled.mission);
        setMissionHistory((current) => [
          cancelled.mission,
          ...current.filter((item) => item.missionId !== cancelled.mission.missionId),
        ]);
        setMissionMessages((current) => [
          ...current,
          {
            messageId: makeId("mission-message"),
            role: "assistant",
            content: cancelled.receipt
              ? "活动的 Nav2 命令已取消，Mission 已停止。"
              : "Mission 已取消；没有活动的物理命令。",
            modality: "text",
            createdAt: new Date().toISOString(),
          },
        ]);
        setNotice("Mission 已安全取消");
        await loadRobots();
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [loadRobots, missionRecord]);

  const approveMissionRevision = useCallback(() => {
    if (!missionRecord || missionRecord.snapshot.status !== "awaiting_review") return;
    const pending = [...missionRecord.revisions].reverse().find(
      (item) => item.disposition === "human_required" && typeof item.revisionId === "string",
    );
    if (!pending || typeof pending.revisionId !== "string") {
      setError("没有找到可审核的 Mission 修订");
      return;
    }
    void (async () => {
      setAssistantBusy(true);
      setError("");
      try {
        const approved = await apiFetch<MissionRecordV2>(
          `/api/v2/missions/${encodeURIComponent(missionRecord.missionId)}/revisions/${encodeURIComponent(pending.revisionId as string)}/approve`,
          { method: "POST", body: JSON.stringify({ approvedBy: "operator" }) },
        );
        setMissionRecord(approved);
        setMissionHistory((current) => [
          approved,
          ...current.filter((item) => item.missionId !== approved.missionId),
        ]);
        await advanceMission(approved.missionId);
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [advanceMission, missionRecord]);

  useEffect(() => {
    const missionId = missionRecord?.missionId;
    const status = missionRecord?.snapshot.status;
    if (!missionId || !shouldReconcileMission(status)) {
      return undefined;
    }
    let disposed = false;
    let refreshInFlight = false;
    const applyLatestMission = (latest: MissionRecordV2) => {
      if (disposed) return;
      setMissionRecord(latest);
      setMissionHistory((current) => [
        latest,
        ...current.filter((item) => item.missionId !== latest.missionId),
      ]);
    };
    const refreshMission = async () => {
      if (refreshInFlight) return;
      refreshInFlight = true;
      try {
        const latest = await apiFetch<MissionAdvanceResultV2>(
          `/api/v2/missions/${encodeURIComponent(missionId)}/advance`,
          { method: "POST", body: JSON.stringify({ maximumSteps: 4 }) },
        );
        applyLatestMission(latest.mission);
      } catch {
        // Advancing can briefly conflict with an in-flight reconciliation.
        // Always read the durable record as a fallback so a Robot Agent/Nav2
        // rejection replaces the optimistic "running" node in the browser.
        try {
          const latest = await apiFetch<MissionRecordV2>(
            `/api/v2/missions/${encodeURIComponent(missionId)}`,
            { cache: "no-store" },
          );
          applyLatestMission(latest);
        } catch {
          // Keep the last durable snapshot and let the next interval retry.
        }
      } finally {
        refreshInFlight = false;
      }
    };
    void refreshMission();
    const timer = window.setInterval(() => void refreshMission(), 1200);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [missionRecord?.missionId, missionRecord?.snapshot.status]);

  const cancelAssistantTurn = useCallback(() => {
    if (!assistantTurn || !["queued", "running"].includes(assistantTurn.status)) return;
    void (async () => {
      try {
        const turn = await apiFetch<AssistantTurn>(
          `/api/v1/assistant/turns/${encodeURIComponent(assistantTurn.taskId)}/cancel`,
          { method: "POST" },
        );
        setAssistantTurn(turn);
        setNotice("规划已取消；人工控制仍可继续使用");
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [assistantTurn]);

  const confirmDraft = useCallback(() => {
    if (!isDraftReviewableForRobot(draft, selectedRobotId, false)) {
      setError("草案目标机器人与当前选择不一致，请切回对应机器人后再审核");
      return;
    }
    void (async () => {
      setAssistantBusy(true);
      setError("");
      try {
        const result = await apiFetch<{ draftId: string; status: string; receipts?: CommandReceipt[]; mission?: VisualMissionRecord }>(`/api/v1/assistant/drafts/${encodeURIComponent(draft.draftId)}/confirm`, {
          method: "POST",
          body: JSON.stringify({ confirmed: true }),
        });
        setDraft((current) => current ? { ...current, status: result.status } : current);
        if (result.mission) setVisualMission(result.mission);
        setAssistantConversation((current) => current ? {
          ...current,
          status: result.mission ? "executing" : "completed",
        } : current);
        if (result.receipts?.length) {
          const receipt = result.receipts[result.receipts.length - 1];
          setLastReceipt(receipt);
          setLastCommandId(receipt.commandId);
        }
        await loadRobots();
        setNotice(`计划 ${statusLabel(result.status)}`);
      } catch (reason) {
        if (reason instanceof ApiFailure && reason.status === 409) {
          try {
            const refreshed = await apiFetch<AssistantDraft>(
              `/api/v1/assistant/drafts/${encodeURIComponent(draft.draftId)}`,
            );
            setDraft(refreshed);
            if (["expired", "failed", "partial", "unknown"].includes(refreshed.status)) {
              setAssistantConversation((current) => current ? { ...current, status: "failed" } : current);
            }
          } catch {
            // Keep the original conflict: it is more useful than a follow-up
            // refresh failure and no action was dispatched by this client.
          }
        }
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [draft, loadRobots, selectedRobotId]);

  const runVisualMission = useCallback(() => {
    if (!draft || !isVisualMissionPlan(draft.plan)) return;
    const missionId = draft.plan.mission.missionId;
    void (async () => {
      setAssistantBusy(true);
      setError("");
      try {
        const mission = await apiFetch<VisualMissionRecord>(
          `/api/v1/visual-missions/${encodeURIComponent(missionId)}/run`,
          { method: "POST" },
        );
        setVisualMission(mission);
        const receipt = mission.state.receipts?.at(-1) ?? mission.stages.at(-1)?.state?.receipt;
        if (receipt) {
          setLastReceipt(receipt);
          setLastCommandId(receipt.commandId);
        }
        setNotice(mission.status === "completed" ? "视觉目标已通过模拟 Gateway 完成发送" : `视觉任务：${statusLabel(mission.status)}`);
        await loadRobots();
      } catch (reason) {
        setError(friendlyError(reason));
        await loadVisualMission(missionId).catch(() => undefined);
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [draft, loadRobots, loadVisualMission]);

  const selectVisualMissionCandidate = useCallback((candidateId: string) => {
    if (!draft || !isVisualMissionPlan(draft.plan)) return;
    const missionId = draft.plan.mission.missionId;
    void (async () => {
      setAssistantBusy(true);
      setError("");
      try {
        const mission = await apiFetch<VisualMissionRecord>(
          `/api/v1/visual-missions/${encodeURIComponent(missionId)}/candidates/${encodeURIComponent(candidateId)}`,
          { method: "POST" },
        );
        setVisualMission(mission);
        setNotice("候选目标已冻结；请再次开始解析与发送");
      } catch (reason) {
        setError(friendlyError(reason));
        await loadVisualMission(missionId).catch(() => undefined);
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [draft, loadVisualMission]);

  const cancelVisualMission = useCallback(() => {
    if (!draft || !isVisualMissionPlan(draft.plan)) return;
    const missionId = draft.plan.mission.missionId;
    void (async () => {
      setAssistantBusy(true);
      try {
        const mission = await apiFetch<VisualMissionRecord>(
          `/api/v1/visual-missions/${encodeURIComponent(missionId)}/cancel`,
          { method: "POST" },
        );
        setVisualMission(mission);
        setNotice("视觉任务已取消");
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [draft]);

  const loadVisualEvidence = useCallback(async (missionId: string, sha256: string) => {
    const blob = await apiFetchBlob(
      `/api/v1/visual-missions/${encodeURIComponent(missionId)}/evidence/${encodeURIComponent(sha256)}`,
    );
    return URL.createObjectURL(blob);
  }, []);

  const replanExpiredDraft = useCallback(() => {
    const source = [...(assistantConversation?.messages ?? [])]
      .reverse()
      .find((message) => message.role === "user" && message.content.trim());
    if (!source) {
      setError("找不到原始任务内容，请在输入框中重新描述任务。");
      return;
    }
    setError("");
    void sendAssistantMessage(source.content, "text", null);
  }, [assistantConversation, sendAssistantMessage]);

  const rejectDraft = useCallback(() => {
    if (!isDraftReviewableForRobot(draft, selectedRobotId, false)) {
      setError("草案目标机器人与当前选择不一致，请切回对应机器人后再审核");
      return;
    }
    void (async () => {
      setAssistantBusy(true);
      try {
        const rejected = await apiFetch<AssistantDraft>(
          `/api/v1/assistant/drafts/${encodeURIComponent(draft.draftId)}/reject`,
          { method: "POST", body: JSON.stringify({ reason: reviewFeedback.trim() || "人工审核拒绝" }) },
        );
        setDraft((current) => current ? { ...current, status: rejected.status } : current);
        setAssistantConversation((current) => current ? { ...current, status: "rejected" } : current);
        setNotice("计划已拒绝，未发送任何机器人命令");
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [draft, reviewFeedback, selectedRobotId]);

  const reviseDraft = useCallback(() => {
    if (!isDraftReviewableForRobot(draft, selectedRobotId, false)) {
      setError("草案目标机器人与当前选择不一致，请切回对应机器人后再审核");
      return;
    }
    if (!reviewFeedback.trim()) return;
    void (async () => {
      setAssistantBusy(true);
      setError("");
      try {
        const accepted = await apiFetch<{ taskId: string; conversationId: string; status: string }>(
          `/api/v1/assistant/drafts/${encodeURIComponent(draft.draftId)}/revise`,
          { method: "POST", body: JSON.stringify({ feedback: reviewFeedback.trim() }) },
        );
        setDraft((current) => current ? { ...current, status: "revision_requested" } : current);
        setReviewFeedback("");
        setAssistantTurn({ taskId: accepted.taskId, conversationId: accepted.conversationId, status: accepted.status });
        await monitorAssistantTurn(accepted.taskId, draft.plan.robotId);
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantBusy(false);
      }
    })();
  }, [draft, monitorAssistantTurn, reviewFeedback, selectedRobotId]);

  const toggleVoiceRecording = useCallback(() => {
    void (async () => {
      setError("");
      if (!recording) {
        if (connection !== "live") {
          setError("工作站当前离线，语音任务不会录制后排队发送。");
          return;
        }
        try {
          const recorder = new BrowserWavRecorder();
          await recorder.start();
          recorderRef.current = recorder;
          setRecording(true);
          setNotice("正在录音；再次点击后将转写内容直接发送给调度 Agent");
        } catch (reason) {
          setError(friendlyError(reason));
        }
        return;
      }
      setVoiceBusy(true);
      setRecording(false);
      try {
        const wav = await recorderRef.current?.stop();
        recorderRef.current = null;
        if (!wav) throw new Error("没有可转写的录音");
        const form = new FormData();
        form.append("file", wav, `voice-${Date.now()}.wav`);
        form.append("locale", "zh-CN");
        const transcript = await apiFetch<SpeechTranscription>("/api/v1/speech/transcriptions", {
          method: "POST",
          body: form,
        });
        setNotice("语音已转写并发送；任何真机动作仍需审核 PlanDraft");
        await sendMissionMessage(transcript.text, "voice", transcript.transcriptionId);
      } catch (reason) {
        await recorderRef.current?.cancel();
        recorderRef.current = null;
        setError(friendlyError(reason));
      } finally {
        setVoiceBusy(false);
      }
    })();
  }, [connection, recording, sendMissionMessage]);

  const createMapping = useCallback((event: FormEvent) => {
    event.preventDefault();
    void runRequest(async () => {
      let mapping: Record<string, unknown>;
      try {
        mapping = JSON.parse(mappingJson) as Record<string, unknown>;
      } catch {
        throw new Error("映射 JSON 格式不正确");
      }
      await apiFetch<MappingSnapshot>("/api/v1/mappings/drafts", { method: "POST", body: JSON.stringify(mapping) });
      setMappingMessage("草稿已保存");
      await loadMappings();
    });
  }, [loadMappings, mappingJson, runRequest]);

  const lifecycle = useCallback((record: MappingSnapshot, action: "validate" | "probe" | "activate") => {
    void runRequest(async () => {
      await apiFetch<MappingSnapshot>(`/api/v1/mappings/${encodeURIComponent(record.mappingId)}/${encodeURIComponent(record.version)}/${action}`, { method: "POST" });
      setMappingMessage(`${record.mappingId}@${record.version}：${statusLabel(action === "validate" ? "validated" : action === "probe" ? "probed" : "active")}`);
      await loadMappings();
    });
  }, [loadMappings, runRequest]);

  const rollback = useCallback((record: MappingSnapshot) => {
    void runRequest(async () => {
      await apiFetch<MappingSnapshot>(`/api/v1/mappings/${encodeURIComponent(record.commandKey)}/rollback`, {
        method: "POST",
        body: JSON.stringify({ version: record.version }),
      });
      setMappingMessage(`${record.commandKey} 已回滚到 ${record.version}`);
      await loadMappings();
    });
  }, [loadMappings, runRequest]);

  const refresh = useCallback(() => {
    void runRequest(async () => {
      await Promise.all([loadRobots(), loadMappings(), loadCapabilities(selectedRobotId)]);
      setNotice("状态已刷新");
    });
  }, [loadCapabilities, loadMappings, loadRobots, runRequest, selectedRobotId]);

  useEffect(() => {
    try {
      window.localStorage.setItem("h-ide-advanced-mode", String(advancedMode));
    } catch {
      // Local preferences are optional; private browsing may reject writes.
    }
  }, [advancedMode]);

  const startNewTask = useCallback((robotId = selectedRobotId) => {
    if (robotId !== selectedRobotId) setSelectedRobotId(robotId);
    setAssistantConversation(null);
    setAssistantTurn(null);
    setDraft(null);
    setMissionDraft(null);
    setMissionRecord(null);
    setMissionMessages([]);
    setAssistantText("");
    setTranscriptionId(null);
    setReviewFeedback("");
    setError("");
    navigate("/");
  }, [navigate, selectedRobotId]);

  const selectRobotForNewTask = useCallback((robotId: string) => {
    if (robotId === selectedRobotId) return;
    startNewTask(robotId);
    setNotice(`已为 ${robotId} 打开新的任务会话；原会话上下文未被修改`);
  }, [selectedRobotId, startNewTask]);

  const openConversation = useCallback((conversationId: string) => {
    setHistoryOpen(false);
    navigate(`/tasks/${encodeURIComponent(conversationId)}`);
  }, [navigate]);

  const missionConversationEntries = useMemo<AssistantConversation[]>(() => {
    const records = missionHistory.map((item) => ({
      conversationId: item.missionId,
      robotId: item.robotId,
      requestedBy: "operator",
      locale: "zh-CN",
      status: item.snapshot.status,
      createdAt: item.plan.createdAt,
      updatedAt: item.snapshot.updatedAt,
      messages: [{
        messageId: `mission-tab-${item.missionId}`,
        role: "user" as const,
        content: item.plan.objective,
        modality: "text" as const,
        createdAt: item.plan.createdAt,
      }],
    }));
    const missionIds = new Set(records.map((item) => item.conversationId));
    const drafts = missionDraftHistory
      .filter((item) => !missionIds.has(item.draftId))
      .map((item) => ({
        conversationId: item.draftId,
        robotId: item.robotId,
        requestedBy: "operator",
        locale: "zh-CN",
        status: item.status,
        createdAt: item.plan.createdAt,
        updatedAt: item.updatedAt ?? item.plan.createdAt,
        messages: [{
          messageId: `mission-draft-tab-${item.draftId}`,
          role: "user" as const,
          content: item.plan.objective,
          modality: "text" as const,
          createdAt: item.plan.createdAt,
        }],
      }));
    return [...records, ...drafts];
  }, [missionDraftHistory, missionHistory]);
  const allConversationEntries = useMemo(() => [
    ...missionConversationEntries,
    ...assistantConversations.filter((item) => !missionConversationEntries.some(
      (mission) => mission.conversationId === item.conversationId,
    )),
  ], [assistantConversations, missionConversationEntries]);
  const currentConversationEntry = missionConversationEntries.find(
    (item) => item.conversationId === (missionRecord?.missionId ?? missionDraft?.draftId),
  ) ?? assistantConversation;
  const conversationPath = missionRecord
    ? `/tasks/${encodeURIComponent(missionRecord.missionId)}`
    : missionDraft
      ? `/tasks/${encodeURIComponent(missionDraft.draftId)}`
      : assistantConversation
        ? `/tasks/${encodeURIComponent(assistantConversation.conversationId)}`
        : "/";

  const latestEvents = [...events].reverse().slice(0, 12);
  const visualMissionInFlight = Boolean(
    visualMission && ["resolving", "dispatching", "navigating", "verifying"].includes(visualMission.status),
  );
  const missionInFlight = Boolean(
    missionRecord
    && ["ready", "observing", "dispatching", "executing", "verifying"].includes(
      missionRecord.snapshot.status,
    ),
  );
  const planExecutionInFlight = visualMissionInFlight || missionInFlight || (
    draft?.status === "confirmed" && Boolean(lastReceipt && !isTerminal(lastReceipt.status))
  );
  const motionDisabled = busy || planExecutionInFlight || !selectedRobot || selectedRobot.safetyMode === "estop" || selectedRobot.status === "offline";
  const activeTaskStatus = visualMission
    ? visualMission.status
    : missionRecord?.snapshot.status
      ?? missionDraft?.status
      ?? (draft?.status === "confirmed" && lastReceipt
        ? lastReceipt.status
        : draft?.status ?? assistantTurn?.status ?? assistantConversation?.status ?? "idle");

  return (
    <div className="hide-app-shell">
      <AppTitlebar
        conversations={allConversationEntries}
        current={currentConversationEntry}
        route={route}
        onNew={() => startNewTask()}
        onOpenConversation={openConversation}
        onOpenHistory={() => setHistoryOpen(true)}
        onOpenTools={() => setToolMenuOpen((value) => !value)}
      />
      <StatusBar
        robots={robots}
        robot={selectedRobot}
        selectedRobotId={selectedRobotId}
        connection={connection}
        activeSceneContext={activeSceneContext}
        taskStatus={activeTaskStatus}
        onRobotChange={selectRobotForNewTask}
        onOpenScene={() => navigate("/scenes")}
        onStop={() => submitCommand("stop", {})}
        stopDisabled={!selectedRobot || connection !== "live" || busy || !capabilities?.capabilities.includes("stop")}
      />

      {connection === "offline" ? <div className="global-connection-banner" role="alert">工作站连接已中断。离线期间不会排队发送规划、审核或控制命令。</div> : null}
      {error ? <div className="global-error-banner" role="alert"><Icon name="warning" /><span><strong>操作未完成</strong>{error}</span><button type="button" onClick={() => setError("")} aria-label="关闭错误"><Icon name="close" /></button></div> : null}

      {route.kind === "chat" ? <ChatWorkspace
        conversation={assistantConversation}
        missionMessages={missionMessages}
        robot={selectedRobot}
        turn={assistantTurn}
        draft={draft}
        text={assistantText}
        setText={setAssistantText}
        feedback={reviewFeedback}
        setFeedback={setReviewFeedback}
        activeSceneContext={activeSceneContext}
        lastReceipt={lastReceipt}
        visualMission={visualMission}
        missionDraft={missionDraft}
        mission={missionRecord}
        busy={assistantBusy}
        voiceBusy={voiceBusy}
        recording={recording}
        connection={connection}
        onSubmit={submitAssistantMessage}
        onVoice={toggleVoiceRecording}
        onCancelPlanning={cancelAssistantTurn}
        onConfirm={confirmDraft}
        onConfirmMission={confirmMissionDraft}
        onRejectMission={rejectMissionDraft}
        onContinueMission={continueMission}
        onCancelMission={cancelMission}
        onApproveMissionRevision={approveMissionRevision}
        onReject={rejectDraft}
        onRevise={reviseDraft}
        onReplan={replanExpiredDraft}
        onRunVisualMission={runVisualMission}
        onSelectVisualCandidate={selectVisualMissionCandidate}
        onCancelVisualMission={cancelVisualMission}
        onLoadVisualEvidence={loadVisualEvidence}
        onOpenScene={() => navigate("/scenes")}
        onPcdFile={(file) => { setPendingSceneFile(file); navigate("/scenes"); }}
        onPrompt={setAssistantText}
      /> : null}

      {route.kind === "scene" ? <WorkspacePage
        title="场景地图"
        subtitle="PCD 可视化、坐标标定与语义位置"
        onBack={() => navigate(conversationPath)}
      >
        <SceneWorkspace
          apiFetch={apiFetch}
          selectedRobotId={selectedRobotId}
          currentPose={selectedRobot?.pose}
          onContextChange={setActiveSceneContext}
          onClose={() => navigate(conversationPath)}
          initialFile={pendingSceneFile}
          onInitialFileConsumed={() => setPendingSceneFile(null)}
        />
      </WorkspacePage> : null}

      {route.kind === "simulation" ? <WorkspacePage
        title="仿真观测台"
        subtitle="MuJoCo + SONIC、Nav2 联调与连续任务回放"
        onBack={() => navigate(conversationPath)}
      >
        <SimulationWorkspace
          apiFetch={apiFetch}
          apiFetchBlob={apiFetchBlob}
          onBack={() => navigate(conversationPath)}
          robotId={selectedRobotId || "robot-1"}
          robotBootId={typeof selectedRobot?.details?.robotBootId === "string" ? selectedRobot.details.robotBootId : undefined}
          localizationEpochId={typeof selectedRobot?.details?.localizationEpochId === "string" ? selectedRobot.details.localizationEpochId : undefined}
          missionId={missionRecord && !["completed", "failed", "cancelled"].includes(missionRecord.snapshot.status)
            ? missionRecord.missionId
            : undefined}
        />
      </WorkspacePage> : null}

      {route.kind === "interfaces" ? <WorkspacePage
        title="接口中心"
        subtitle="逻辑动作与代理接口的版本化映射"
        onBack={() => navigate(conversationPath)}
      >
        <InterfaceView
          mappings={mappings}
          mappingJson={mappingJson}
          setMappingJson={setMappingJson}
          message={mappingMessage}
          busy={busy}
          onCreate={createMapping}
          onLifecycle={lifecycle}
          onRollback={rollback}
        />
      </WorkspacePage> : null}

      {toolMenuOpen ? <ToolMenu
        advancedMode={advancedMode}
        onAdvancedModeChange={setAdvancedMode}
        onClose={() => setToolMenuOpen(false)}
        onControl={() => { setToolMenuOpen(false); setControlPanelOpen(true); }}
        onScene={() => { setToolMenuOpen(false); navigate("/scenes"); }}
        onSimulation={() => { setToolMenuOpen(false); navigate("/simulation"); }}
        onHistory={() => { setToolMenuOpen(false); setHistoryOpen(true); }}
        onInterfaces={() => { setToolMenuOpen(false); navigate("/interfaces"); }}
        onRefresh={() => { setToolMenuOpen(false); refresh(); }}
      /> : null}

      {controlPanelOpen ? <SidePanel title="手动控制" subtitle="所有动作记录到控制审计" onClose={() => setControlPanelOpen(false)}>
        <ManualControlPanel
          robot={selectedRobot}
          capabilities={capabilities}
          lastReceipt={lastReceipt}
          lastCommandId={lastCommandId}
          latestEvents={latestEvents}
          disabled={motionDisabled || connection !== "live"}
          busy={busy}
          onCommand={submitCommand}
          onCancel={cancelLast}
        />
      </SidePanel> : null}

      {historyOpen ? <SidePanel title="任务历史" subtitle={`${allConversationEntries.length} 个会话`} onClose={() => setHistoryOpen(false)} side="left">
        <HistoryPanel
          conversations={allConversationEntries}
          currentId={currentConversationEntry?.conversationId}
          onOpen={openConversation}
          onNew={() => { setHistoryOpen(false); startNewTask(); }}
        />
      </SidePanel> : null}

      {notice ? <div className="app-toast" role="status"><Icon name="check" /><span>{notice}</span><button type="button" onClick={() => setNotice("")} aria-label="关闭提示"><Icon name="close" /></button></div> : null}
    </div>
  );
}

type IconName =
  | "logo" | "plus" | "history" | "tools" | "robot" | "map" | "stop"
  | "close" | "warning" | "check" | "mic" | "send" | "attach" | "control"
  | "interface" | "refresh" | "back" | "spark" | "chevron" | "activity";

function Icon({ name }: { name: IconName }) {
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  const paths: Record<IconName, React.ReactNode> = {
    logo: <><path d="M6 7.5 12 4l6 3.5v7L12 18l-6-3.5z" /><path d="m9 9 3-1.7L15 9v3.5L12 14l-3-1.5z" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    history: <><path d="M4.8 8A8 8 0 1 1 4 12" /><path d="M4 4v4h4M12 8v4l2.8 1.8" /></>,
    tools: <><circle cx="12" cy="12" r="3" /><path d="M19 13.5v-3l-2-.7-.8-1.8.9-1.9L15 4l-1.9.9-1.9-.8-.7-2h-3l-.7 2-1.9.8L3 4 1 6.1 1.9 8l-.8 1.8-2 .7v3l2 .7.8 1.8L1 18l2 2 1.9-.9 1.9.8.7 2h3l.7-2 1.9-.8L15 20l2.1-2-.9-2 .8-1.8z" transform="translate(2.5 0) scale(.8)" /></>,
    robot: <><rect x="5" y="7" width="14" height="11" rx="3" /><path d="M12 4v3M8.5 12h.01M15.5 12h.01M9 15h6" /></>,
    map: <><path d="m4 6 5-2 6 2 5-2v14l-5 2-6-2-5 2zM9 4v14M15 6v14" /></>,
    stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
    close: <path d="m7 7 10 10M17 7 7 17" />,
    warning: <><path d="M12 3 2.8 20h18.4z" /><path d="M12 9v4M12 17h.01" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    mic: <><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M9 21h6" /></>,
    send: <><path d="m3 11 18-8-8 18-2-8zM11 13l5-5" /></>,
    attach: <path d="m9 12 6.2-6.2a3 3 0 1 1 4.2 4.2l-8.6 8.6a5 5 0 0 1-7.1-7.1L12 3.2" />,
    control: <><path d="M4 7h10M18 7h2M4 17h2M10 17h10" /><circle cx="16" cy="7" r="2" /><circle cx="8" cy="17" r="2" /></>,
    interface: <><rect x="4" y="4" width="16" height="16" rx="3" /><path d="M8 9h8M8 13h5M8 17h8" /></>,
    refresh: <><path d="M20 7v5h-5" /><path d="M18.5 16a8 8 0 1 1 1.2-7.7L20 12" /></>,
    back: <path d="m15 18-6-6 6-6" />,
    spark: <><path d="m12 3 1.2 4.3L17 9l-3.8 1.7L12 15l-1.2-4.3L7 9l3.8-1.7z" /><path d="m18.5 14 .6 2.1L21 17l-1.9.9-.6 2.1-.6-2.1L16 17l1.9-.9z" /></>,
    chevron: <path d="m9 6 6 6-6 6" />,
    activity: <><path d="M3 12h4l2-6 4 12 2-6h6" /></>,
  };
  return <svg className="app-icon" viewBox="0 0 24 24" aria-hidden="true" {...common}>{paths[name]}</svg>;
}

function conversationLabel(conversation: AssistantConversation, current: AssistantConversation | null): string {
  const detail = current?.conversationId === conversation.conversationId ? current : conversation;
  const firstUserMessage = detail.messages?.find((message) => message.role === "user")?.content.trim();
  if (firstUserMessage) return firstUserMessage.length > 22 ? `${firstUserMessage.slice(0, 22)}…` : firstUserMessage;
  return `${conversation.robotId} 的新任务`;
}

function AppTitlebar({
  conversations,
  current,
  route,
  onNew,
  onOpenConversation,
  onOpenHistory,
  onOpenTools,
}: {
  conversations: AssistantConversation[];
  current: AssistantConversation | null;
  route: AppRoute;
  onNew: () => void;
  onOpenConversation: (conversationId: string) => void;
  onOpenHistory: () => void;
  onOpenTools: () => void;
}) {
  const tabs = [current, ...conversations]
    .filter((item): item is AssistantConversation => Boolean(item))
    .filter((item, index, list) => list.findIndex((candidate) => candidate.conversationId === item.conversationId) === index)
    .slice(0, 5);
  return <header className="app-titlebar">
    <button className="brand-button" type="button" onClick={onNew} aria-label="H-IDE 新任务"><Icon name="logo" /><span>H-IDE</span></button>
    <nav className="mission-tabs" aria-label="活动任务">
      {tabs.map((conversation) => <button
        key={conversation.conversationId}
        className={route.kind === "chat" && current?.conversationId === conversation.conversationId ? "active" : ""}
        type="button"
        onClick={() => onOpenConversation(conversation.conversationId)}
      ><span className={`tab-state tab-${conversation.status}`} />{conversationLabel(conversation, current)}</button>)}
      <button className={`new-mission-tab ${route.kind === "chat" && !current ? "active" : ""}`} type="button" onClick={onNew}><Icon name="plus" />新任务</button>
    </nav>
    <div className="titlebar-actions">
      <button className="icon-button" type="button" onClick={onOpenHistory} aria-label="任务历史"><Icon name="history" /></button>
      <button className="icon-button titlebar-tools-button" type="button" onClick={onOpenTools} aria-label="打开工具" title="工具">
        <Icon name="tools" /><span>工具</span>
      </button>
    </div>
  </header>;
}

function poseFreshness(robot: RobotState | null): { label: string; stale: boolean } {
  const value = robot?.pose?.observedAt;
  if (!value) return { label: "位姿时间未知", stale: true };
  const milliseconds = Date.parse(value);
  if (!Number.isFinite(milliseconds)) return { label: "位姿时间未知", stale: true };
  const seconds = Math.max(0, Math.floor((Date.now() - milliseconds) / 1000));
  if (seconds <= 5) return { label: "位姿刚刚更新", stale: false };
  if (seconds < 60) return { label: `位姿 ${seconds} 秒前`, stale: seconds > 30 };
  return { label: `位姿 ${Math.floor(seconds / 60)} 分钟前`, stale: true };
}

function StatusBar({
  robots,
  robot,
  selectedRobotId,
  connection,
  activeSceneContext,
  taskStatus,
  onRobotChange,
  onOpenScene,
  onStop,
  stopDisabled,
}: {
  robots: RobotState[];
  robot: RobotState | null;
  selectedRobotId: string;
  connection: ConnectionState;
  activeSceneContext: ActiveSceneContext | null;
  taskStatus: string;
  onRobotChange: (robotId: string) => void;
  onOpenScene: () => void;
  onStop: () => void;
  stopDisabled: boolean;
}) {
  const pose = poseFreshness(robot);
  return <div className="global-statusbar" aria-label="当前控制上下文">
    <label className="status-selector"><Icon name="robot" /><select value={selectedRobotId} onChange={(event) => onRobotChange(event.target.value)} aria-label="当前机器人">
      {robots.length ? robots.map((item) => <option key={item.robotId} value={item.robotId}>{item.name ?? item.robotId}</option>) : <option value={selectedRobotId}>{selectedRobotId}</option>}
    </select><span className={`context-dot ${robot?.connected ? "live" : "offline"}`} /></label>
    <span className={`status-context connection-${connection}`}><span className="context-dot" />{connection === "live" ? "工作站在线" : connection === "connecting" ? "正在连接" : "工作站离线"}</span>
    <button className={`status-context context-button ${activeSceneContext ? "ready" : ""}`} type="button" onClick={onOpenScene}><Icon name="map" />{activeSceneContext ? `场景 r${activeSceneContext.revision}` : "未绑定场景"}</button>
    <span className={`status-context ${pose.stale ? "stale" : "ready"}`}><Icon name="activity" />{pose.label}</span>
    <span className="status-context task-context">任务 <StatusPill value={taskStatus} /></span>
    <button className="global-stop-button" type="button" onClick={onStop} disabled={stopDisabled}><Icon name="stop" />停止机器人</button>
  </div>;
}

function WorkspacePage({ title, subtitle, onBack, children }: { title: string; subtitle: string; onBack: () => void; children: React.ReactNode }) {
  return <main className="full-workspace">
    <header className="workspace-header"><button className="icon-button" type="button" onClick={onBack} aria-label="返回会话"><Icon name="back" /></button><div><h1>{title}</h1><p>{subtitle}</p></div></header>
    <div className="workspace-content">{children}</div>
  </main>;
}

function ToolMenu({
  advancedMode,
  onAdvancedModeChange,
  onClose,
  onControl,
  onScene,
  onSimulation,
  onHistory,
  onInterfaces,
  onRefresh,
}: {
  advancedMode: boolean;
  onAdvancedModeChange: (value: boolean) => void;
  onClose: () => void;
  onControl: () => void;
  onScene: () => void;
  onSimulation: () => void;
  onHistory: () => void;
  onInterfaces: () => void;
  onRefresh: () => void;
}) {
  return <><button className="popover-scrim" type="button" onClick={onClose} aria-label="关闭工具" /><aside className="tool-menu" aria-label="H-IDE 工具">
    <div className="tool-menu-header"><span>工具</span><button className="icon-button" type="button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button></div>
    <button type="button" onClick={onControl}><span className="tool-icon"><Icon name="control" /></span><span><strong>手动控制</strong><small>测试高层动作与停止</small></span><Icon name="chevron" /></button>
    <button type="button" onClick={onScene}><span className="tool-icon"><Icon name="map" /></span><span><strong>场景地图</strong><small>PCD、标定与语义位置</small></span><Icon name="chevron" /></button>
    <button type="button" onClick={onSimulation}><span className="tool-icon"><Icon name="activity" /></span><span><strong>仿真观测台</strong><small>连续任务、障碍与回放</small></span><Icon name="chevron" /></button>
    <button type="button" onClick={onHistory}><span className="tool-icon"><Icon name="history" /></span><span><strong>任务历史</strong><small>恢复规划、审核和执行记录</small></span><Icon name="chevron" /></button>
    <button type="button" onClick={onRefresh}><span className="tool-icon"><Icon name="refresh" /></span><span><strong>刷新状态</strong><small>重新读取机器人与接口快照</small></span><Icon name="chevron" /></button>
    <div className="tool-separator" />
    <label className="advanced-toggle"><span><strong>工程模式</strong><small>显示接口与调试工具</small></span><input type="checkbox" checked={advancedMode} onChange={(event) => onAdvancedModeChange(event.target.checked)} /><i /></label>
    {advancedMode ? <button type="button" onClick={onInterfaces}><span className="tool-icon"><Icon name="interface" /></span><span><strong>接口中心</strong><small>映射版本、探测与回滚</small></span><Icon name="chevron" /></button> : null}
  </aside></>;
}

function SidePanel({ title, subtitle, onClose, side = "right", children }: { title: string; subtitle: string; onClose: () => void; side?: "left" | "right"; children: React.ReactNode }) {
  return <div className="side-panel-layer"><button className="side-panel-scrim" type="button" onClick={onClose} aria-label={`关闭${title}`} /><aside className={`side-panel side-${side}`} aria-label={title}>
    <header><div><h2>{title}</h2><p>{subtitle}</p></div><button className="icon-button" type="button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button></header>
    <div className="side-panel-content">{children}</div>
  </aside></div>;
}

function HistoryPanel({ conversations, currentId, onOpen, onNew }: { conversations: AssistantConversation[]; currentId?: string; onOpen: (conversationId: string) => void; onNew: () => void }) {
  return <div className="history-panel"><button className="new-history-task" type="button" onClick={onNew}><Icon name="plus" />开始新任务</button>
    <div className="history-list">
      {conversations.length ? conversations.map((conversation) => <button className={conversation.conversationId === currentId ? "active" : ""} key={conversation.conversationId} type="button" onClick={() => onOpen(conversation.conversationId)}>
        <span className={`history-state tab-${conversation.status}`}><Icon name="spark" /></span><span><strong>{conversationLabel(conversation, conversation.conversationId === currentId ? conversation : null)}</strong><small>{conversation.robotId} · {formatDateTime(conversation.updatedAt ?? conversation.createdAt)}</small></span><StatusPill value={conversation.status} />
      </button>) : <EmptyState>还没有任务会话。发送第一条消息后，会话会保存在这里。</EmptyState>}
    </div>
  </div>;
}

function DeadmanButton({ disabled, onStart, onStop }: { disabled: boolean; onStart: () => void; onStop: () => void }) {
  const pressed = useRef(false);
  const start = () => {
    if (disabled || pressed.current) return;
    pressed.current = true;
    onStart();
  };
  const stop = () => {
    if (!pressed.current) return;
    pressed.current = false;
    onStop();
  };
  return <button
    className="deadman-button"
    type="button"
    disabled={disabled}
    onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); start(); }}
    onPointerUp={stop}
    onPointerCancel={stop}
    onKeyDown={(event) => { if (["Enter", " "].includes(event.key) && !event.repeat) start(); }}
    onKeyUp={(event) => { if (["Enter", " "].includes(event.key)) stop(); }}
    aria-label="按住向前移动"
  ><span className="deadman-arrow">↑</span><strong>按住向前移动</strong><small>松开立即发送停止</small></button>;
}

function ManualControlPanel({ robot, capabilities, lastReceipt, lastCommandId, latestEvents, disabled, busy, onCommand, onCancel }: {
  robot: RobotState | null;
  capabilities: CapabilitySnapshot | null;
  lastReceipt: CommandReceipt | null;
  lastCommandId: string | null;
  latestEvents: ControlEvent[];
  disabled: boolean;
  busy: boolean;
  onCommand: (commandKey: string, payload?: Record<string, unknown>) => void;
  onCancel: () => void;
}) {
  const can = (key: string) => Boolean(capabilities?.capabilities.includes(key));
  return <div className="manual-control-panel">
    <section className="manual-robot-card"><span className={`context-dot ${robot?.connected ? "live" : "offline"}`} /><div><strong>{robot?.name ?? robot?.robotId ?? "未选择机器人"}</strong><small>{robot?.connected ? "在线" : "离线"} · 电量 {robot?.battery == null ? "—" : `${robot.battery}%`} · {statusLabel(robot?.safetyMode)}</small></div></section>
    <p className="manual-safety-note"><Icon name="warning" />手动动作只发送给当前机器人。物理急停和本地安全控制始终优先。</p>
    <DeadmanButton disabled={disabled || !can("move.forward") || !can("stop")} onStart={() => onCommand("move.forward", {})} onStop={() => onCommand("stop", {})} />
    <div className="manual-actions">
      <button type="button" disabled={disabled || !can("gripper.open")} onClick={() => onCommand("gripper.open", {})}><span><Icon name="control" /></span><strong>打开夹爪</strong><small>离散高层命令</small></button>
      <button className="manual-stop" type="button" disabled={busy || !robot || !can("stop")} onClick={() => onCommand("stop", {})}><span><Icon name="stop" /></span><strong>立即停止</strong><small>{can("stop") ? "无二次确认" : "当前代理未提供"}</small></button>
      <button type="button" disabled={!lastCommandId || busy} onClick={onCancel}><span><Icon name="close" /></span><strong>取消最近命令</strong><small>{lastCommandId ?? "暂无命令"}</small></button>
    </div>
    {lastReceipt ? <ReceiptCard receipt={lastReceipt} /> : null}
    <section className="manual-events"><div className="panel-section-title"><span>最近回执</span><small>{latestEvents.length} 条</small></div>{latestEvents.length ? latestEvents.slice(0, 6).map((event, index) => <TimelineItem event={event} key={`${event.eventId ?? "event"}-${index}`} />) : <EmptyState>还没有手动或 Agent 执行记录。</EmptyState>}</section>
  </div>;
}

function ChatWorkspace({
  conversation,
  missionMessages,
  robot,
  turn,
  draft,
  text,
  setText,
  feedback,
  setFeedback,
  activeSceneContext,
  lastReceipt,
  visualMission,
  missionDraft,
  mission,
  busy,
  voiceBusy,
  recording,
  connection,
  onSubmit,
  onVoice,
  onCancelPlanning,
  onConfirm,
  onConfirmMission,
  onRejectMission,
  onContinueMission,
  onCancelMission,
  onApproveMissionRevision,
  onReject,
  onRevise,
  onReplan,
  onRunVisualMission,
  onSelectVisualCandidate,
  onCancelVisualMission,
  onLoadVisualEvidence,
  onOpenScene,
  onPcdFile,
  onPrompt,
}: {
  conversation: AssistantConversation | null;
  missionMessages: AssistantMessage[];
  robot: RobotState | null;
  turn: AssistantTurn | null;
  draft: AssistantDraft | null;
  text: string;
  setText: (value: string) => void;
  feedback: string;
  setFeedback: (value: string) => void;
  activeSceneContext: ActiveSceneContext | null;
  lastReceipt: CommandReceipt | null;
  visualMission: VisualMissionRecord | null;
  missionDraft: MissionDraftRecordV2 | null;
  mission: MissionRecordV2 | null;
  busy: boolean;
  voiceBusy: boolean;
  recording: boolean;
  connection: ConnectionState;
  onSubmit: (event: FormEvent) => void;
  onVoice: () => void;
  onCancelPlanning: () => void;
  onConfirm: () => void;
  onConfirmMission: () => void;
  onRejectMission: () => void;
  onContinueMission: () => void;
  onCancelMission: () => void;
  onApproveMissionRevision: () => void;
  onReject: () => void;
  onRevise: () => void;
  onReplan: () => void;
  onRunVisualMission: () => void;
  onSelectVisualCandidate: (candidateId: string) => void;
  onCancelVisualMission: () => void;
  onLoadVisualEvidence: (missionId: string, sha256: string) => Promise<string>;
  onOpenScene: () => void;
  onPcdFile: (file: File) => void;
  onPrompt: (prompt: string) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const messages = missionMessages.length ? missionMessages : conversation?.messages ?? [];
  const activeTurn = Boolean(turn && ["queued", "running"].includes(turn.status));
  const empty = !messages.length && !turn && !draft && !missionDraft && !mission;
  useEffect(() => {
    const element = timelineRef.current;
    if (!element) return;
    if (typeof element.scrollTo === "function") element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
    else element.scrollTop = element.scrollHeight;
  }, [busy, draft?.draftId, messages.length, mission?.snapshot.lastSequence, missionDraft?.draftId, turn?.status]);
  const choosePcd = (file: File | undefined) => {
    if (!file) return;
    setDragging(false);
    onPcdFile(file);
  };
  const suggestions = activeSceneContext
    ? ["前进 5 米", "去厨房", "查看当前场景中的目标位置", "测试夹爪"]
    : ["向前移动 5 米", "测试夹爪", "查看机器人当前状态", "导入场景地图"];
  return <main className={`chat-workspace ${dragging ? "dragging" : ""}`} onDragEnter={(event) => { if (event.dataTransfer.types.includes("Files")) { event.preventDefault(); setDragging(true); } }} onDragOver={(event) => { if (dragging) event.preventDefault(); }} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false); }} onDrop={(event) => { event.preventDefault(); choosePcd(event.dataTransfer.files?.[0]); }}>
    {dragging ? <div className="chat-drop-overlay"><Icon name="map" /><strong>导入 PCD 场景</strong><span>释放后进入场景工作区并开始转换</span></div> : null}
    <div className="chat-timeline" ref={timelineRef} aria-live="polite">
      <div className="chat-thread">
        {empty ? <section className="chat-empty-state"><span className="empty-logo"><Icon name="logo" /></span><h1>现在，要让机器人做什么？</h1><p>描述目标即可。Supervisor 会选择专属 Agent 形成计划，任何真机动作都要由你审核后执行。</p><div className="suggestion-grid">{suggestions.map((prompt) => <button key={prompt} type="button" onClick={() => prompt === "导入场景地图" ? onOpenScene() : onPrompt(prompt)}>{prompt}<Icon name="chevron" /></button>)}</div></section> : null}
        {messages.map((message) => <article className={`chat-message message-${message.role}`} key={message.messageId}>
          <div className="message-author">{message.role === "user" ? "你" : message.role === "assistant" ? <><span className="agent-avatar"><Icon name="spark" /></span>H-IDE</> : "系统"}<time>{formatTime(message.createdAt)}</time></div>
          <div className="message-body">{message.content}</div>
          {message.modality === "voice" ? <span className="message-modality"><Icon name="mic" />语音转写</span> : null}
        </article>)}
        {turn && !missionDraft && !mission ? <PlanningActivityCard turn={turn} draft={draft} /> : null}
        {turn?.result?.assistantMessage && !messages.some((message) => message.taskId === turn.taskId && message.role === "assistant") ? <article className="chat-message message-assistant"><div className="message-author"><span className="agent-avatar"><Icon name="spark" /></span>H-IDE</div><div className="message-body">{turn.result.assistantMessage}</div></article> : null}
        {missionDraft && !mission ? <MissionPlanCard
          draft={missionDraft}
          robot={robot}
          busy={busy}
          feedback={feedback}
          setFeedback={setFeedback}
          onConfirm={onConfirmMission}
          onReject={onRejectMission}
        /> : null}
        {mission ? <MissionExecutionCard
          mission={mission}
          busy={busy}
          onContinue={onContinueMission}
          onCancel={onCancelMission}
          onApproveRevision={onApproveMissionRevision}
        /> : null}
        {!missionDraft && !mission && draft && isVisualMissionPlan(draft.plan) ? <VisualMissionCard
          draft={draft}
          robot={robot}
          mission={visualMission}
          busy={busy}
          expired={isDraftExpired(draft)}
          reviewable={isDraftReviewableForRobot(draft, robot?.robotId ?? "", busy)}
          expiry={formatDraftExpiry(draft.plan.expiresAt, Date.now())}
          onConfirm={onConfirm}
          onRun={onRunVisualMission}
          onSelectCandidate={onSelectVisualCandidate}
          onCancel={onCancelVisualMission}
          feedback={feedback}
          setFeedback={setFeedback}
          onReject={onReject}
          onRevise={onRevise}
          onLoadEvidence={onLoadVisualEvidence}
          onReplan={onReplan}
        /> : !missionDraft && !mission && draft ? <PlanDraftCard
          draft={draft}
          robot={robot}
          activeSceneContext={activeSceneContext}
          busy={busy}
          feedback={feedback}
          setFeedback={setFeedback}
          onConfirm={onConfirm}
          onReject={onReject}
          onRevise={onRevise}
          onReplan={onReplan}
        /> : null}
        {!mission && draft?.status === "confirmed" && lastReceipt && !isVisualMissionPlan(draft.plan) ? <ExecutionCard receipt={lastReceipt} /> : null}
      </div>
    </div>
    <div className="composer-region">
      {activeTurn ? <div className="active-task-dock"><span className="planning-pulse" /><span>Supervisor 正在编排任务</span><button type="button" onClick={onCancelPlanning}>取消规划</button></div> : busy ? <div className="active-task-dock"><span className="planning-pulse" /><span>Mission Supervisor 正在规划或推进任务</span></div> : null}
      <form className="chat-composer" onSubmit={onSubmit}>
        <div className="composer-contexts"><span><Icon name="robot" />{robot?.name ?? robot?.robotId ?? "未选择机器人"}</span><button type="button" className={activeSceneContext ? "attached" : ""} onClick={onOpenScene}><Icon name="map" />{activeSceneContext ? `场景 r${activeSceneContext.revision}${activeSceneContext.selectedPoint ? " · 已选目标" : ""}` : "添加场景"}</button></div>
        <textarea id="assistant-task" aria-label="任务描述" value={text} onChange={(event) => setText(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} rows={1} maxLength={4000} placeholder={connection === "live" ? "描述任务，或按住麦克风说话…" : "工作站离线，任务草稿不会发送"} disabled={connection !== "live"} />
        <div className="composer-actions"><div><input ref={fileInputRef} type="file" hidden accept=".pcd,application/octet-stream" onChange={(event) => { choosePcd(event.target.files?.[0]); event.target.value = ""; }} /><button className="composer-icon-button" type="button" onClick={() => fileInputRef.current?.click()} aria-label="导入 PCD"><Icon name="attach" /></button><button className={`composer-icon-button voice-button ${recording ? "recording" : ""}`} type="button" onClick={onVoice} disabled={voiceBusy || busy || connection !== "live"} aria-label={recording ? "停止录音并发送" : "开始语音输入"}><Icon name="mic" /></button><span className="composer-hint">{recording ? "正在录音，再次点击直接发送" : voiceBusy ? "正在转写并发送…" : "Enter 发送 · Shift+Enter 换行"}</span></div><button className="send-button" type="submit" disabled={busy || !text.trim() || connection !== "live"} aria-label="发送给调度 Agent"><Icon name="send" /></button></div>
      </form>
      <p className="composer-disclaimer">H-IDE 可能生成错误计划。执行前请核对机器人、场景、目标坐标和安全条件。</p>
    </div>
  </main>;
}

function MissionPlanCard({ draft, robot, busy, feedback, setFeedback, onConfirm, onReject }: {
  draft: MissionDraftRecordV2;
  robot: RobotState | null;
  busy: boolean;
  feedback: string;
  setFeedback: (value: string) => void;
  onConfirm: () => void;
  onReject: () => void;
}) {
  const plan = draft.plan;
  const reviewable = draft.status === "draft" && !busy && plan.robotId === robot?.robotId;
  return <section className="plan-draft-card mission-v2-card" aria-label="Mission V2 计划审核">
    <header>
      <div><span className="card-kicker"><Icon name="spark" />MISSION V2 · SUPERVISOR–WORKER</span><h2>{plan.objective}</h2></div>
      <StatusPill value={draft.status} />
    </header>
    <div className="plan-context-grid">
      <span><small>目标机器人</small><strong>{plan.robotId}</strong></span>
      <span><small>风险上限</small><strong>{plan.envelope.maximumRiskLevel}</strong></span>
      <span><small>单段 / 累计距离</small><strong>{plan.envelope.maximumGoalDistanceM}m / {plan.envelope.maximumCumulativeDistanceM}m</strong></span>
      <span><small>审核有效期</small><strong>{formatDraftExpiry(plan.expiresAt, Date.now())}</strong></span>
    </div>
    <p className="card-warning"><Icon name="warning" />视觉模型只提出语义假设；目标坐标必须由 RGB-D、场景几何或人工事实确认。所有底盘位移和转向统一交给 Nav2。</p>
    <ol className="plan-step-list">
      {plan.nodes.map((node, index) => <li key={node.nodeId}>
        <span>{index + 1}</span>
        <div><strong>{node.skillId}@{node.skillVersion}</strong><p>{node.purpose}</p><small>{node.dependsOn.length ? `依赖：${node.dependsOn.join("、")}` : "起始节点"}</small></div>
        <StatusPill value={node.riskLevel} />
      </li>)}
    </ol>
    <details className="plan-technical-details">
      <summary><span>本次授权边界</span><Icon name="chevron" /></summary>
      <div>
        <section><strong>能力快照</strong><p>{plan.envelope.allowedSkillIds.join("、")}</p></section>
        <section><strong>固定身份</strong><p>boot: {plan.envelope.robotBootId}<br />epoch: {plan.envelope.localizationEpochId}</p></section>
        <section><strong>自动修订</strong><p>最多 {plan.envelope.maximumAutoRevisions} 次；超出授权边界必须重新审核。</p></section>
        <section><strong>场景先验</strong><p>{plan.envelope.sceneRefs.length ? plan.envelope.sceneRefs.map((item) => `${item.sceneId}@${item.revision}`).join("、") : "未绑定 PCD 场景"}</p></section>
      </div>
    </details>
    {draft.status === "draft" ? <div className="plan-review-controls">
      <label htmlFor="mission-review-feedback">审核备注（拒绝时会保存原因）</label>
      <textarea id="mission-review-feedback" value={feedback} onChange={(event) => setFeedback(event.target.value)} placeholder="例如：目标描述不清楚，先补充视觉参照物" />
      <div className="plan-review-actions"><button className="reject-plan" type="button" onClick={onReject} disabled={!reviewable}>拒绝</button><HoldConfirmButton disabled={!reviewable} onConfirm={onConfirm} /></div>
    </div> : <p className="frozen-plan-note"><Icon name="check" />该草案已{statusLabel(draft.status)}，没有发送未审核动作。</p>}
  </section>;
}

function MissionExecutionCard({ mission, busy, onContinue, onCancel, onApproveRevision }: {
  mission: MissionRecordV2;
  busy: boolean;
  onContinue: () => void;
  onCancel: () => void;
  onApproveRevision: () => void;
}) {
  const [visualEvidenceOpen, setVisualEvidenceOpen] = useState(false);
  const status = mission.snapshot.status;
  const states = new Map(
    mission.nodeStates
      .filter((item) => item.revisionId === mission.snapshot.activeRevisionId)
      .map((item) => [item.nodeId, item]),
  );
  const hypotheses = mission.worldRecords.filter((item) => item.kind === "hypothesis").length;
  const facts = mission.worldRecords.filter((item) => item.kind === "fact" && item.predicate !== "mission_origin").length;
  const latestEvents = [...mission.events].reverse().slice(0, 8);
  const pendingRevision = [...mission.revisions].reverse().find(
    (item) => item.disposition === "human_required",
  );
  const visualResult = latestMissionVisualResult(mission);
  const observationOnly = !mission.activeRevision.nodes.some((node) =>
    ["navigation.navigate_to_pose", "navigation.rotate"].includes(node.skillId));
  const loadMissionVisualEvidence = useCallback(async (
    missionId: string,
    analysisId: string,
    sha256: string,
  ) => {
    const blob = await apiFetchBlob(
      `/api/v2/missions/${encodeURIComponent(missionId)}/visual-evidence/${encodeURIComponent(analysisId)}/${encodeURIComponent(sha256)}`,
    );
    return URL.createObjectURL(blob);
  }, []);
  return <>
  <section className="plan-draft-card mission-v2-card mission-runtime-card" aria-label="连续 Mission 执行状态">
    <header>
      <div><span className="card-kicker"><Icon name="activity" />CONTINUOUS MISSION</span><h2>{mission.activeRevision.objective}</h2></div>
      <StatusPill value={status} />
    </header>
    <div className="mission-runtime-summary">
      <span><small>当前节点</small><strong>{mission.snapshot.activeNodeId ?? "—"}</strong></span>
      <span><small>活动修订</small><strong>r{mission.activeRevision.sequence}</strong></span>
      <span><small>视觉假设</small><strong>{hypotheses}</strong></span>
      <span><small>可信空间事实</small><strong>{facts}</strong></span>
    </div>
    {visualResult ? <MissionVisualSummary
      result={visualResult}
      observationOnly={observationOnly}
      onOpen={() => setVisualEvidenceOpen(true)}
    /> : null}
    {mission.snapshot.detail ? <p className={`mission-runtime-notice notice-${status}`}><Icon name={status === "needs_input" ? "warning" : "activity"} />{mission.snapshot.detail}</p> : null}
    <ol className="plan-step-list mission-node-list">
      {mission.activeRevision.nodes.map((node, index) => {
        const state = states.get(node.nodeId);
        const nodeStatus = state?.status ?? "pending";
        return <li key={node.nodeId} className={`node-${nodeStatus}`}>
          <span>{nodeStatus === "succeeded" ? "✓" : nodeStatus === "skipped" ? "–" : index + 1}</span>
          <div><strong>{node.skillId}</strong><p>{node.purpose}</p><small>尝试 {state?.attempt ?? 0} 次 · {statusLabel(nodeStatus)}</small></div>
          <StatusPill value={nodeStatus} />
        </li>;
      })}
    </ol>
    {latestEvents.length ? <details className="plan-technical-details mission-event-details">
      <summary><span>事件时间线（{mission.events.length}）</span><Icon name="chevron" /></summary>
      <div>{latestEvents.map((event) => <section key={event.eventId}><strong>#{event.sequence} · {event.eventType}</strong><p>{event.nodeId ?? "Mission"} · {formatTime(event.occurredAt)}</p></section>)}</div>
    </details> : null}
    {status === "needs_input" ? <p className="mission-input-hint">请在下方继续用文字或语音补充。补充内容会进入 PlanRevision，不会绕过审核直接执行。</p> : null}
    {status === "awaiting_review" ? <div className="mission-revision-review"><p><strong>修订需要审核</strong><span>{Array.isArray(pendingRevision?.changeSummary) ? pendingRevision.changeSummary.join("；") : "持续规划改变了任务步骤或授权范围。"}</span></p><button type="button" onClick={onApproveRevision} disabled={busy}>审核修订并继续</button></div> : null}
    {!['completed', 'failed', 'cancelled'].includes(status) ? <div className="mission-runtime-actions">
      <button className="mission-cancel-button" type="button" onClick={onCancel} disabled={busy}>取消 Mission</button>
      {status === "ready" ? <button type="button" onClick={onContinue} disabled={busy}>继续执行就绪节点</button> : null}
    </div> : null}
    {status === "completed" ? <p className="frozen-plan-note"><Icon name="check" />Mission 已完成；每个目标与执行回执均保留在事件记录中。</p> : null}
  </section>
  {visualEvidenceOpen && visualResult ? <MissionVisualEvidenceDrawer
    missionId={mission.missionId}
    result={visualResult}
    observationOnly={observationOnly}
    onClose={() => setVisualEvidenceOpen(false)}
    onLoadEvidence={loadMissionVisualEvidence}
  /> : null}
  </>;
}

function PlanningActivityCard({ turn, draft }: { turn: AssistantTurn; draft: AssistantDraft | null }) {
  const active = ["queued", "running"].includes(turn.status);
  const contributions = draft?.review?.contributions ?? [];
  return <details className={`planning-activity planning-${turn.status}`} open={active || undefined}>
    <summary><span className="agent-avatar"><Icon name="spark" /></span><span><strong>{active ? "正在规划任务" : turn.status === "draft_ready" ? "规划已完成" : "规划轮次"}</strong><small>{active ? "Supervisor 正在选择并协调专属 Agent" : statusLabel(turn.status)}</small></span><StatusPill value={turn.status} /><Icon name="chevron" /></summary>
    <div className="planning-details"><div className="planning-stage-list"><span className="done"><i />理解任务</span><span className={turn.status === "queued" ? "active" : "done"}><i />选择 Agent</span><span className={active ? "active" : draft ? "done" : ""}><i />合并计划</span></div>
      {contributions.length ? <div className="worker-list">{contributions.map((worker) => <div key={worker.agentId}><span><Icon name="spark" /></span><p><strong>{worker.agentId}</strong><small>{worker.summary}</small></p></div>)}</div> : <p className="planning-placeholder">专属 Agent 的贡献会在规划完成后显示。原始请求与响应仅在工程模式中提供。</p>}
      {turn.error ? <p className="planning-error">{turn.error}</p> : null}
    </div>
  </details>;
}

function HoldConfirmButton({ disabled, onConfirm }: { disabled: boolean; onConfirm: () => void }) {
  const [holding, setHolding] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const fired = useRef(false);
  const cancel = () => {
    if (timer.current !== undefined) window.clearTimeout(timer.current);
    timer.current = undefined;
    if (!fired.current) setHolding(false);
  };
  const start = () => {
    if (disabled || timer.current !== undefined) return;
    fired.current = false;
    setHolding(true);
    timer.current = window.setTimeout(() => {
      timer.current = undefined;
      fired.current = true;
      setHolding(false);
      onConfirm();
    }, 900);
  };
  useEffect(() => () => { if (timer.current !== undefined) window.clearTimeout(timer.current); }, []);
  return <button
    className={`hold-confirm ${holding ? "holding" : ""}`}
    type="button"
    disabled={disabled}
    onPointerDown={start}
    onPointerUp={cancel}
    onPointerCancel={cancel}
    onPointerLeave={cancel}
    onKeyDown={(event) => { if (["Enter", " "].includes(event.key) && !event.repeat) start(); }}
    onKeyUp={(event) => { if (["Enter", " "].includes(event.key)) cancel(); }}
    onClick={(event) => { if (event.detail === 0 && !holding && !disabled) onConfirm(); }}
  ><span className="hold-progress" /><Icon name="check" /><span>{holding ? "继续按住…" : "按住确认并执行"}</span></button>;
}

function PlanDraftCard({ draft, robot, activeSceneContext, busy, feedback, setFeedback, onConfirm, onReject, onRevise, onReplan }: {
  draft: AssistantDraft;
  robot: RobotState | null;
  activeSceneContext: ActiveSceneContext | null;
  busy: boolean;
  feedback: string;
  setFeedback: (value: string) => void;
  onConfirm: () => void;
  onReject: () => void;
  onRevise: () => void;
  onReplan: () => void;
}) {
  if (isVisualMissionPlan(draft.plan)) return null;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    setNow(Date.now());
    if (draft.status !== "draft" || !draft.plan.expiresAt) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [draft.draftId, draft.plan.expiresAt, draft.status]);
  const review = draft.review;
  const expired = isDraftExpired(draft, now);
  const reviewable = !expired && isDraftReviewableForRobot(draft, robot?.robotId ?? "", busy);
  const expiry = formatDraftExpiry(draft.plan.expiresAt, now);
  return <article className={`plan-draft-card draft-${expired ? "expired" : draft.status}`}>
    <header><div><span className={`card-kicker ${expired ? "expired" : ""}`}><Icon name={expired ? "warning" : "check"} />{expired ? "计划已过期" : "等待人工审核"}</span><h2>{review?.summary ?? `${draft.plan.steps.length} 步机器人计划`}</h2></div><StatusPill value={expired ? "expired" : draft.status} /></header>
    <div className="plan-context-grid"><span><small>目标机器人</small><strong>{draft.plan.robotId}</strong></span><span><small>场景版本</small><strong>{activeSceneContext ? `r${activeSceneContext.revision}` : "未绑定"}</strong></span><span><small>起点位姿</small><strong>{robot?.pose ? `${robot.pose.x.toFixed(2)}, ${robot.pose.y.toFixed(2)}` : "未知"}</strong></span><span><small>审核有效期</small><strong className={expired ? "expired" : ""}>{expiry}</strong></span></div>
    {draft.plan.robotId !== robot?.robotId ? <p className="card-warning"><Icon name="warning" />当前机器人与草案不一致，不能执行。</p> : null}
    {review?.warnings?.length ? <div className="plan-warnings"><strong><Icon name="warning" />执行前确认</strong>{review.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div> : null}
    <ol className="plan-step-list">{draft.plan.steps.map((step, index) => {
      const detail = review?.steps.find((item) => item.stepId === step.stepId);
      const goal = navigationGoal(step.payload);
      return <li key={step.stepId}><span>{index + 1}</span><div><strong>{step.commandKey}</strong>{detail ? <p>{detail.rationale}</p> : null}{goal ? <small>目标坐标 · {goal.x.toFixed(2)}, {goal.y.toFixed(2)}, {goal.z.toFixed(2)} · yaw {goal.yaw.toFixed(2)}</small> : null}{step.dependsOn?.length ? <small>等待步骤：{step.dependsOn.join("、")}</small> : null}</div><StatusPill value="pending" /></li>;
    })}</ol>
    {(review?.contributions?.length || review?.spatialChecks?.length) ? <details className="plan-technical-details"><summary>查看 Agent 贡献与空间依据 <Icon name="chevron" /></summary><div>{review?.contributions?.map((worker) => <section key={worker.agentId}><strong>{worker.agentId}</strong><p>{worker.summary}</p></section>)}{review?.spatialChecks?.map((check) => <section key={check.stepId}><strong>{check.stepId} · {({ observed_free: "未发现障碍", occupied: "障碍警告", unknown: "观测不足", outside: "地图外" } as const)[check.check.classification]}</strong><p>map {check.goalPose.x.toFixed(2)}, {check.goalPose.y.toFixed(2)} · 净空 {check.check.clearanceMeters.toFixed(2)} m</p></section>)}</div></details> : null}
    {expired ? <div className="expired-plan-actions"><p><Icon name="warning" />有效期结束后，机器人状态和场景可能已经变化，因此不能执行这份旧计划。</p><button type="button" onClick={onReplan} disabled={busy}>根据原任务重新规划</button></div> : draft.status === "draft" ? <div className="plan-review-controls"><label htmlFor="review-feedback">修改意见</label><textarea id="review-feedback" aria-label="审核意见" value={feedback} onChange={(event) => setFeedback(event.target.value)} rows={2} maxLength={2000} placeholder="需要修改时，说明新的约束、目标或顺序" /><div className="plan-review-actions"><HoldConfirmButton disabled={!reviewable} onConfirm={onConfirm} /><button type="button" onClick={onRevise} disabled={!reviewable || !feedback.trim()}>要求修改</button><button className="reject-plan" type="button" onClick={onReject} disabled={!reviewable}>拒绝</button></div></div> : <p className="frozen-plan-note"><Icon name="check" />该草案已冻结为不可变审核记录：{statusLabel(draft.status)}</p>}
  </article>;
}

function ExecutionCard({ receipt }: { receipt: CommandReceipt }) {
  const active = ["pending", "accepted", "running"].includes(receipt.status);
  return <article className={`execution-card execution-${receipt.status}`}><span className={`execution-icon ${active ? "active" : ""}`}><Icon name="activity" /></span><div><small>ActionRuntime 执行</small><strong>{statusLabel(receipt.status)}</strong><code>{receipt.commandId}</code>{receipt.error ? <p>{receipt.error}</p> : null}</div><StatusPill value={receipt.status} /></article>;
}

function formatDateTime(value?: string): string {
  if (!value) return "刚刚";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function WorkspaceView({
  robot,
  capabilities,
  lastReceipt,
  lastCommandId,
  disabled,
  busy,
  onCommand,
  onCancel,
}: {
  robot: RobotState | null;
  capabilities: CapabilitySnapshot | null;
  lastReceipt: CommandReceipt | null;
  lastCommandId: string | null;
  disabled: boolean;
  busy: boolean;
  onCommand: (commandKey: string, payload?: Record<string, unknown>) => void;
  onCancel: () => void;
}) {
  const can = (key: string) => Boolean(capabilities?.capabilities.includes(key));
  return <>
    <section className="hero-panel panel">
      <div className="hero-copy"><p className="eyebrow">SELECTED ROBOT</p><h2>{robot?.name ?? robot?.robotId ?? "未选择机器人"}</h2><p className="hero-description">通过版本化高层动作控制机器人；每条命令都带有有效期、幂等键和可追踪回执。</p></div>
      <div className="robot-state-summary"><div><span>连接</span><strong>{robot?.connected ? "在线" : "离线"}</strong></div><div><span>电量</span><strong>{robot?.battery == null ? "—" : `${robot.battery}%`}</strong></div><div><span>安全模式</span><strong>{statusLabel(robot?.safetyMode)}</strong></div></div>
    </section>
    <section className="panel command-panel">
      <div className="section-heading"><div><p className="eyebrow">HIGH-LEVEL ACTIONS</p><h2>动作控制</h2></div><span className="muted">逻辑命令 · adapter 负责接口映射</span></div>
      <div className="action-grid">
        <ActionButton label="向前移动" hint="move.forward" icon="↑" disabled={disabled || !can("move.forward")} onClick={() => onCommand("move.forward", {})} />
        <ActionButton label="打开夹爪" hint="gripper.open" icon="◌" disabled={disabled || !can("gripper.open")} onClick={() => onCommand("gripper.open", {})} />
        <ActionButton label="立即停止" hint="stop · safety" icon="■" danger disabled={busy || !robot || !can("stop")} onClick={() => onCommand("stop", {})} />
        <button className="action-button secondary-action" type="button" disabled={!lastCommandId || busy} onClick={onCancel}><span className="action-icon">×</span><span><strong>取消最近命令</strong><small>{lastCommandId ?? "暂无可取消命令"}</small></span></button>
      </div>
      {(!robot || robot.status === "offline" || robot.status === "unknown") && <p className="inline-warning">机器人状态未确认，远程动作已锁定；仍可取消正在跟踪的命令，现场安全控制始终优先。</p>}
      {robot?.safetyMode === "estop" && <p className="inline-warning">机器人处于急停状态，远程动作已锁定。请由现场人员确认后恢复。</p>}
      {lastReceipt && <ReceiptCard receipt={lastReceipt} />}
    </section>
    <section className="panel capability-panel"><div className="section-heading compact"><div><p className="eyebrow">CAPABILITY SNAPSHOT</p><h2>能力快照</h2></div><span className="muted">{capabilities?.agent ?? "未读取"}</span></div><div className="chip-row">{capabilities?.capabilities.map((item) => <span className="capability-chip" key={item}>{item}</span>) ?? <span className="muted">选择在线机器人后读取能力</span>}</div></section>
  </>;
}

function ActionButton({ label, hint, icon, disabled, danger, onClick }: { label: string; hint: string; icon: string; disabled: boolean; danger?: boolean; onClick: () => void }) {
  return <button className={`action-button ${danger ? "danger-action" : ""}`} type="button" disabled={disabled} onClick={onClick}><span className="action-icon">{icon}</span><span><strong>{label}</strong><small>{hint}</small></span><span className="arrow">→</span></button>;
}

function ReceiptCard({ receipt }: { receipt: CommandReceipt }) {
  return <div className={`receipt-card receipt-${receipt.status}`}><div><span className="eyebrow">LATEST RECEIPT</span><strong>{statusLabel(receipt.status)}</strong></div><code>{receipt.commandId}</code>{receipt.error && <p>{receipt.error}</p>}{receipt.safetyWarning && <p className="warning-text">安全提醒：{receipt.safetyWarning}</p>}</div>;
}

interface AssistantViewProps {
  conversation: AssistantConversation | null;
  selectedRobotId: string;
  turn: AssistantTurn | null;
  text: string;
  setText: (value: string) => void;
  transcriptionId: string | null;
  draft: AssistantDraft | null;
  busy: boolean;
  voiceBusy: boolean;
  recording: boolean;
  feedback: string;
  setFeedback: (value: string) => void;
  activeSceneContext: ActiveSceneContext | null;
  scenePanelOpen: boolean;
  onScene: () => void;
  onSubmit: (event: FormEvent) => void;
  onVoice: () => void;
  onCancel: () => void;
  onConfirm: () => void;
  onReject: () => void;
  onRevise: () => void;
}

function AssistantView({
  conversation,
  selectedRobotId,
  turn,
  text,
  setText,
  transcriptionId,
  draft,
  busy,
  voiceBusy,
  recording,
  feedback,
  setFeedback,
  activeSceneContext,
  scenePanelOpen,
  onScene,
  onSubmit,
  onVoice,
  onCancel,
  onConfirm,
  onReject,
  onRevise,
}: AssistantViewProps) {
  const review = draft?.review;
  const actionPlan = draft && !isVisualMissionPlan(draft.plan) ? draft.plan : null;
  const activeTurn = turn && ["queued", "running"].includes(turn.status);
  const reviewable = isDraftReviewableForRobot(draft, selectedRobotId, busy);
  return (
    <section className="panel assistant-panel">
      <div className="section-heading">
        <div><p className="eyebrow">SUPERVISOR · WORKERS</p><h2>Agent 任务编排</h2></div>
        <div className="assistant-heading-status">
          <span className="muted">规划与执行分离</span>
          <StatusPill value={conversation?.status ?? turn?.status ?? "idle"} />
        </div>
      </div>
      <div className="assistant-layout">
        <div className="assistant-input-column">
          <div className="input-mode-grid" aria-label="任务输入方式">
            <button
              className={`input-mode-card ${scenePanelOpen ? "active" : ""}`}
              type="button"
              onClick={onScene}
              aria-label={scenePanelOpen ? "收起 PCD 场景" : "打开 PCD 场景"}
            >
              <span className="input-mode-icon">◫</span><strong>导入 PCD</strong><small>可视化 · 标定 · 标注</small>
            </button>
            <button
              className="input-mode-card"
              type="button"
              onClick={() => document.getElementById("assistant-task")?.focus()}
              aria-label="输入文本任务"
            >
              <span className="input-mode-icon">Aa</span><strong>文本任务</strong><small>自然语言描述目标</small>
            </button>
            <button
              className={`input-mode-card ${recording ? "recording" : ""}`}
              type="button"
              onClick={onVoice}
              disabled={voiceBusy || busy}
              aria-label={recording ? "停止录音并转写" : "开始语音输入"}
            >
              <span className="input-mode-icon">◉</span><strong>{recording ? "停止并转写" : "语音助手"}</strong><small>{voiceBusy ? "正在转写…" : "本地 Whisper · WAV"}</small>
            </button>
          </div>
          <div className={`scene-context-chip ${activeSceneContext ? "attached" : ""}`}>
            <span className="scene-context-dot" />
            {activeSceneContext
              ? <span>已附加场景 <b>{activeSceneContext.sceneId.slice(0, 8)} · r{activeSceneContext.revision}</b>{activeSceneContext.selectedPoint ? " · 已选目标点" : ""}</span>
              : <span>Agent 尚未附加场景；请发布版本并绑定当前机器人</span>}
          </div>
          <form className="stack-form assistant-composer" onSubmit={onSubmit}>
            <label htmlFor="assistant-task">任务描述</label>
            <textarea
              id="assistant-task"
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={6}
              maxLength={4000}
              placeholder="例如：向前移动到工作区，然后打开夹爪"
            />
            <div className="composer-meta">
              <span>{text.length}/4000</span>
              {transcriptionId && <span className="transcript-badge">语音转写 · 可编辑</span>}
            </div>
            <div className="assistant-submit-row">
              <button className="button button-primary" type="submit" disabled={busy || !text.trim()}>
                {busy ? "调度 Agent 规划中…" : "发送给调度 Agent"}
              </button>
              <button className="button button-ghost" type="button" onClick={onCancel} disabled={!activeTurn}>取消规划</button>
            </div>
            <div className="form-help">文本或语音只创建规划轮次；所有专属 Agent 输出合并成草案后，仍需人工审核。</div>
          </form>
          <div className="conversation-log" aria-live="polite">
            <div className="list-caption"><span>会话记录</span><span>{conversation?.messages?.length ?? 0} 条</span></div>
            {conversation?.messages?.length ? conversation.messages.slice(-6).map((message) => (
              <div className={`conversation-message ${message.role}`} key={message.messageId}>
                <span>{message.role === "user" ? "你" : message.role === "assistant" ? "调度 Agent" : "系统"}</span>
                <p>{message.content}</p>
              </div>
            )) : <EmptyState>输入任务后，这里会保留多轮澄清与规划记录。</EmptyState>}
            {turn?.result?.assistantMessage && !conversation?.messages?.some((message) => message.taskId === turn.taskId && message.role === "assistant")
              ? <div className="conversation-message assistant"><span>调度 Agent</span><p>{turn.result.assistantMessage}</p></div>
              : null}
          </div>
        </div>

        <div className="draft-review">
          <p className="eyebrow">HUMAN REVIEW GATE</p>
          {!draft ? <>
            <EmptyState>{activeTurn ? "Supervisor 正在调度专属 Agent…" : "尚无待审核草案。规划完成后，在这里核对贡献、步骤、依赖和风险。"}</EmptyState>
            <button className="button button-confirm" type="button" disabled>确认并执行计划</button>
          </> : <>
            <div className="draft-header">
              <div><span className="muted">草案：{draft.draftId}</span><h3>{review?.summary ?? (actionPlan ? `${actionPlan.steps.length} 个高层步骤` : "视觉任务")}</h3></div>
              <StatusPill value={draft.status} />
            </div>
            <div className="plan-target">目标机器人 <strong>{draft.plan.robotId}</strong></div>
            {draft.plan.robotId !== selectedRobotId ? <p className="inline-warning">当前已切换到其他机器人，请切回 {draft.plan.robotId} 后再审核。</p> : null}
            {review?.contributions?.length ? <div className="worker-contributions">
              {review.contributions.map((item) => <span key={item.agentId}><b>{item.agentId}</b><small>{item.summary}</small></span>)}
            </div> : null}
            {review?.warnings?.length ? <div className="review-warnings"><strong>执行提醒</strong>{review.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div> : null}
            {review?.spatialChecks?.length ? <div className="spatial-review-checks">
              <strong>地图目标证据</strong>
              {review.spatialChecks.map((item) => <div className={`spatial-check spatial-${item.check.classification}`} key={item.stepId}>
                <span>{item.stepId}</span>
                <b>{({ observed_free: "未发现障碍", occupied: "障碍警告", unknown: "观测不足", outside: "地图外" } as const)[item.check.classification]}</b>
                <small>净空 {item.check.clearanceMeters.toFixed(2)} m · map {item.goalPose.x.toFixed(2)}, {item.goalPose.y.toFixed(2)}</small>
              </div>)}
              <p>这是工作站由点云派生的审核证据，不替代真机定位与避障。</p>
            </div> : null}
            <ol className="plan-steps">
              {actionPlan?.steps.map((step) => {
                const detail = review?.steps.find((item) => item.stepId === step.stepId);
                const goal = navigationGoal(step.payload);
                return <li key={step.stepId}>
                  <span>{step.stepId}</span><strong>{step.commandKey}</strong>
                  {detail ? <small>{detail.contributedBy} · {detail.rationale}</small> : null}
                  {goal ? <small className="plan-map-goal">map 目标：{goal.x.toFixed(2)}, {goal.y.toFixed(2)}, {goal.z.toFixed(2)} · yaw {goal.yaw.toFixed(2)}</small> : null}
                  {step.dependsOn?.length ? <small>依赖：{step.dependsOn.join(", ")}</small> : null}
                </li>;
              })}
            </ol>
            <label className="review-label" htmlFor="review-feedback">审核意见</label>
            <textarea
              id="review-feedback"
              className="review-feedback"
              value={feedback}
              onChange={(event) => setFeedback(event.target.value)}
              rows={3}
              maxLength={2000}
              placeholder="如需修改，请说明约束或顺序"
            />
            <div className="review-actions">
              <button className="button button-confirm" type="button" onClick={onConfirm} disabled={!reviewable}>确认并执行计划</button>
              <button className="button" type="button" onClick={onRevise} disabled={!reviewable || !feedback.trim()}>要求修改</button>
              <button className="button button-danger" type="button" onClick={onReject} disabled={!reviewable}>拒绝计划</button>
            </div>
            <p className="form-help">只有“确认并执行计划”会进入 ActionRuntime；修改会生成新草案，原草案保持不可变。</p>
          </>}
        </div>
      </div>
    </section>
  );
}

function InterfaceView({ mappings, mappingJson, setMappingJson, message, busy, onCreate, onLifecycle, onRollback }: { mappings: MappingSnapshot[]; mappingJson: string; setMappingJson: (value: string) => void; message: string; busy: boolean; onCreate: (event: FormEvent) => void; onLifecycle: (record: MappingSnapshot, action: "validate" | "probe" | "activate") => void; onRollback: (record: MappingSnapshot) => void }) {
  return <section className="panel interface-panel"><div className="section-heading"><div><p className="eyebrow">INTERFACE CENTER</p><h2>接口版本中心</h2></div><span className="muted">先校验并探测，再启用版本</span></div><div className="interface-layout"><form className="stack-form" onSubmit={onCreate}><label htmlFor="mapping-json">映射 JSON</label><textarea id="mapping-json" value={mappingJson} onChange={(event) => setMappingJson(event.target.value)} rows={14} spellCheck={false} /><div className="form-help">映射只描述逻辑命令到 adapter 的声明式关系。targetId、transport 和 commandKey 会在 BFF 再次校验。</div><button className="button button-primary" type="submit" disabled={busy}>保存新草稿</button>{message && <p className="inline-success">{message}</p>}</form><div className="mapping-list"><div className="list-caption"><span>版本历史</span><span>{mappings.length} 条记录</span></div>{mappings.length === 0 ? <EmptyState>暂无自定义版本，右侧创建的草稿会出现在这里。</EmptyState> : mappings.map((record) => <MappingCard key={`${record.mappingId}:${record.version}`} record={record} busy={busy} onLifecycle={onLifecycle} onRollback={onRollback} />)}</div></div></section>;
}

function MappingCard({ record, busy, onLifecycle, onRollback }: { record: MappingSnapshot; busy: boolean; onLifecycle: (record: MappingSnapshot, action: "validate" | "probe" | "activate") => void; onRollback: (record: MappingSnapshot) => void }) {
  const canValidate = record.status === "draft" || record.status === "validated";
  const canProbe = record.status === "validated" || record.status === "probed";
  const canActivate = record.status === "probed";
  return <article className={`mapping-card mapping-${record.status}`}><div className="mapping-card-heading"><div><strong>{record.mappingId}</strong><span>{record.commandKey} · {record.version}</span></div><StatusPill value={record.status} /></div><div className="mapping-details"><span>transport <b>{String(record.mapping.transport ?? "—")}</b></span><span>target <b>{String(record.targetId ?? record.mapping.targetId ?? "未绑定")}</b></span>{record.probe?.error && <span className="warning-text">探测失败：{record.probe.error}</span>}</div><div className="mapping-actions"><button type="button" className="mini-button" disabled={busy || !canValidate} onClick={() => onLifecycle(record, "validate")}>校验</button><button type="button" className="mini-button" disabled={busy || !canProbe} onClick={() => onLifecycle(record, "probe")}>探测</button><button type="button" className="mini-button mini-primary" disabled={busy || !canActivate} onClick={() => onLifecycle(record, "activate")}>启用</button><button type="button" className="mini-button" disabled={busy || record.status !== "active"} onClick={() => onRollback(record)}>回滚</button></div></article>;
}

function TimelineItem({ event }: { event: ControlEvent }) {
  return <div className="timeline-item"><span className={`timeline-dot ${event.status ?? "unknown"}`} /><div><div className="timeline-title"><strong>{event.commandId ?? "系统事件"}</strong><StatusPill value={event.status} /></div><p>{event.robotId ? `${event.robotId} · ` : ""}{event.error ?? "状态已更新"}</p><time>{formatTime(event.at ?? event.createdAt)}</time></div></div>;
}

function formatDraftExpiry(value: string | undefined, now = Date.now()): string {
  if (!value) return "服务端校验";
  const expiresAt = new Date(value).valueOf();
  if (!Number.isFinite(expiresAt)) return "时间未知";
  const remainingSeconds = Math.ceil((expiresAt - now) / 1000);
  if (remainingSeconds <= 0) return "已过期";
  const minutes = Math.floor(remainingSeconds / 60);
  const seconds = remainingSeconds % 60;
  return minutes > 0 ? `剩余 ${minutes} 分 ${seconds} 秒` : `剩余 ${seconds} 秒`;
}

function formatTime(value?: string): string {
  if (!value) return "刚刚";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

if (typeof document !== "undefined") {
  const root = document.getElementById("root");
  if (root && !root.hasAttribute("data-test-root")) createRoot(root).render(<App />);
}

if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    void navigator.serviceWorker.register("/sw.js");
  });
}

export { App, apiFetch, ApiFailure, isDraftExpired, isDraftReviewableForRobot, preferredRobotId };

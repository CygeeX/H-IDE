import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type {
  ApiError,
  AssistantConversation,
  AssistantConversationBulkDeleteResult,
  AssistantConversationPage,
  AssistantDraft,
  AssistantMessage,
  AuthPrincipal,
  AuthSessionResponse,
  AssistantTurn,
  CapabilitySnapshot,
  CommandReceipt,
  ControlEvent,
  MappingSnapshot,
  MissionAdvanceResultV2,
  MissionDraftRecordV2,
  MissionRecordV2,
  RobotState,
  SpeechTranscription,
  VisualMissionRecord,
} from "./contracts";
import { isAssistantTurnTerminal, isTerminal, isVisualMissionPlan, statusLabel } from "./contracts";
import { BrowserWavRecorder } from "./audio";
import {
  BrowserStreamingSpeechSession,
  speechWebSocketUrl,
  StreamingPcmPlayer,
  type StreamingSpeechMetrics,
  type StreamingSpeechPhase,
} from "./streaming-speech";
import type { ActiveSceneContext } from "./scene-workspace";
import { VisualMissionCard } from "./visual-mission-card";
import { SimulationWorkspace } from "./simulation-workspace";
import { DeveloperDiagnostics } from "./developer-diagnostics";
import { RobotMonitor } from "./robot-monitor";
import { SceneMemoryWorkspace } from "./scene-memory-workspace";
import {
  currentMissionVisualAmbiguity,
  latestMissionVisualResult,
  MissionVisualCandidateSelection,
  MissionVisualEvidenceDrawer,
  MissionVisualSummary,
} from "./mission-visual-evidence";
import {
  currentMissionOfficeReport,
  MissionOfficeReportCard,
} from "./mission-office-report";
import {
  browserSpeakChinese,
  LEADER_SKILLS,
  RobotAvatar,
  RobotPicker,
  SkillModeBanner,
  SkillSelection,
  type LeaderSkillDefinition,
  type LeaderSkillId,
  type RobotDisplayOption,
} from "./leader-demo";
import {
  buildConversationTimeline,
  mergeConversationMessages,
} from "./conversation-timeline";
import { missionUserProgress } from "./mission-progress";
import "./app.css";
import "./chat-shell.css";

type ConnectionState = "connecting" | "live" | "offline";
type VoiceCaptureMode = "streaming" | "offline_wav";
type ConversationLifecycleView = "active" | "archived" | "deleted";
interface VoiceUiState {
  phase: StreamingSpeechPhase;
  partial: string;
  metrics: StreamingSpeechMetrics;
}

interface AssistantRetryRequest {
  text: string;
  modality: "text" | "voice";
  transcriptionId: string | null;
  route: "conversation" | "mission";
  candidateSelection?: {
    candidateId: string;
    selectionSourceAnalysisId: string;
  } | null;
}

interface AssistantClientIssue {
  occurredAt: string;
  operation: string;
  detail: string;
  taskId?: string;
}
type AppRoute =
  | { kind: "chat"; shell: "user" | "developer"; conversationId?: string }
  | { kind: "scene-memory"; shell: "user" | "developer" }
  | { kind: "simulation"; shell: "developer" }
  | { kind: "diagnostics"; shell: "developer" }
  | { kind: "robot-monitor"; shell: "developer" }
  | { kind: "interfaces"; shell: "developer" };

const API_BASE = (import.meta.env.VITE_CONTROL_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";
// Browser sessions are the product path. A bearer token remains an explicit,
// opt-in compatibility path for developer diagnostics and automated fixtures.
const API_TOKEN = (import.meta.env.VITE_CONTROL_API_TOKEN as string | undefined)?.trim() ?? "";
let inMemoryCsrfToken = "";

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

function cookieValue(name: string): string {
  if (typeof document === "undefined") return "";
  const prefix = `${encodeURIComponent(name)}=`;
  const entry = document.cookie.split(";").map((item) => item.trim()).find((item) => item.startsWith(prefix));
  if (!entry) return "";
  try {
    return decodeURIComponent(entry.slice(prefix.length));
  } catch {
    return "";
  }
}

function setCsrfToken(value: string | undefined): void {
  inMemoryCsrfToken = value?.trim() ?? "";
}

function csrfToken(): string {
  return inMemoryCsrfToken || cookieValue("hide_csrf");
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
  const method = (init.method ?? "GET").toUpperCase();
  if (["POST", "PUT", "PATCH", "DELETE"].includes(method) && !API_TOKEN) {
    const csrf = inMemoryCsrfToken || cookieValue("hide_csrf");
    if (csrf) headers.set("x-csrf-token", csrf);
  }
  const response = await fetch(apiUrl(path), { ...init, headers, credentials: "include" });
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
  const response = await fetch(apiUrl(path), { headers, cache: "no-store", credentials: "include" });
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

function isAssistantConversationPage(value: unknown): value is AssistantConversationPage {
  if (!value || typeof value !== "object") return false;
  const page = value as Partial<AssistantConversationPage>;
  return Array.isArray(page.items)
    && (page.nextCursor === null || typeof page.nextCursor === "string");
}

async function fetchAssistantConversationPage({
  lifecycle = "active",
  query = "",
  cursor = null,
  limit = 30,
}: {
  lifecycle?: ConversationLifecycleView;
  query?: string;
  cursor?: string | null;
  limit?: number;
} = {}): Promise<AssistantConversationPage> {
  const parameters = new URLSearchParams({ lifecycle, limit: String(limit) });
  if (query.trim()) parameters.set("q", query.trim());
  if (cursor) parameters.set("cursor", cursor);
  try {
    const page = await apiFetch<unknown>(
      `/api/v2/assistant/conversations?${parameters.toString()}`,
    );
    if (isAssistantConversationPage(page)) return page;
  } catch (reason) {
    if (!(reason instanceof ApiFailure) || reason.status !== 404) throw reason;
  }

  // Compatibility for a rolling deployment where the new PWA reaches an old
  // Control API. V1 is unpaged, so the fallback is intentionally finite and
  // disappears once every workstation exposes the V2 cursor contract.
  const legacyQuery = lifecycle === "active"
    ? ""
    : lifecycle === "archived"
      ? "?includeArchived=true"
      : "?includeArchived=true&includeDeleted=true";
  const legacy = await apiFetch<AssistantConversation[]>(
    `/api/v1/assistant/conversations${legacyQuery}`,
  );
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const filtered = (Array.isArray(legacy) ? legacy : []).filter((conversation) => {
    if (lifecycle === "active" && (conversation.archived || conversation.deleted)) return false;
    if (lifecycle === "archived" && (!conversation.archived || conversation.deleted)) return false;
    if (lifecycle === "deleted" && !conversation.deleted) return false;
    if (!normalizedQuery) return true;
    const firstUserMessage = conversation.messages?.find((message) => message.role === "user")?.content ?? "";
    return `${conversation.title ?? ""}\n${firstUserMessage}`.toLocaleLowerCase().includes(normalizedQuery);
  });
  return { items: filtered.slice(0, limit), nextCursor: null };
}

async function fetchSpeechReply(text: string, signal: AbortSignal): Promise<Response> {
  const headers = new Headers({
    accept: "application/octet-stream",
    "content-type": "application/json",
  });
  if (API_TOKEN) headers.set("authorization", `Bearer ${API_TOKEN}`);
  else if (csrfToken()) headers.set("x-csrf-token", csrfToken());
  const response = await fetch(apiUrl("/api/v2/speech/tts"), {
    method: "POST",
    headers,
    credentials: "include",
    body: JSON.stringify({ text, locale: "zh-CN" }),
    signal,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = apiErrorDetail(await response.json(), detail); } catch { /* Binary errors may be empty. */ }
    throw new ApiFailure(detail || `语音回复请求失败 (${response.status})`, response.status);
  }
  return response;
}

function makeId(prefix: string): string {
  const random = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}-${random}`;
}

interface PendingMissionInterventionEntry {
  fingerprint: string;
  interventionId: string;
}

const PENDING_MISSION_INTERVENTION_PREFIX = "h-ide.pending-mission-interventions.v1";

function pendingMissionInterventionStorageKey(userId: string, missionId: string): string {
  return `${PENDING_MISSION_INTERVENTION_PREFIX}:${encodeURIComponent(userId)}:${encodeURIComponent(missionId)}`;
}

function pendingMissionInterventionEntries(storageKey: string): PendingMissionInterventionEntry[] {
  try {
    const raw = window.sessionStorage.getItem(storageKey);
    if (raw === null) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const entries: PendingMissionInterventionEntry[] = [];
    for (const rawEntry of parsed) {
      if (!rawEntry || typeof rawEntry !== "object") return [];
      const entry = rawEntry as Record<string, unknown>;
      if (typeof entry.fingerprint !== "string" || !entry.fingerprint
        || typeof entry.interventionId !== "string" || !entry.interventionId
        || entry.interventionId !== entry.interventionId.trim()
        || entry.interventionId.length > 256) return [];
      entries.push({
        fingerprint: entry.fingerprint,
        interventionId: entry.interventionId,
      });
    }
    return entries;
  } catch {
    return [];
  }
}

function storedMissionInterventionId(storageKey: string, fingerprint: string): string | null {
  const matching = pendingMissionInterventionEntries(storageKey).filter(
    (entry) => entry.fingerprint === fingerprint,
  );
  return matching.length === 1 ? matching[0].interventionId : null;
}

function storeMissionIntervention(
  storageKey: string,
  entry: PendingMissionInterventionEntry,
): void {
  try {
    const entries = pendingMissionInterventionEntries(storageKey)
      .filter((item) => item.fingerprint !== entry.fingerprint);
    window.sessionStorage.setItem(storageKey, JSON.stringify([...entries, entry].slice(-8)));
  } catch {
    // The in-memory retry identity remains available when browser storage is unavailable.
  }
}

function clearStoredMissionIntervention(
  storageKey: string,
  entry: PendingMissionInterventionEntry,
): void {
  try {
    const entries = pendingMissionInterventionEntries(storageKey);
    const remaining = entries.filter((item) => !(
      item.fingerprint === entry.fingerprint && item.interventionId === entry.interventionId
    ));
    if (remaining.length === entries.length) return;
    if (remaining.length) window.sessionStorage.setItem(storageKey, JSON.stringify(remaining));
    else window.sessionStorage.removeItem(storageKey);
  } catch {
    // A stale browser-storage entry is harmless: the server still binds the id to its payload.
  }
}

function wsUrl(after: number): string {
  if (API_BASE) {
    const parsed = new URL(API_BASE, window.location.href);
    parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
    parsed.pathname = `${parsed.pathname.replace(/\/$/, "")}/api/v1/events`;
    const params = new URLSearchParams({ after: String(after) });
    if (API_TOKEN) params.set("token", API_TOKEN);
    parsed.search = params.toString();
    return parsed.toString();
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const token = API_TOKEN ? `&token=${encodeURIComponent(API_TOKEN)}` : "";
  return `${protocol}//${window.location.host}/api/v1/events?after=${after}${token}`;
}

function friendlyError(error: unknown): string {
  if (error instanceof ApiFailure) {
    const normalized = error.message.toLowerCase();
    if (normalized.includes("assistant plan has expired") || normalized.includes("status=expired")) {
      return "计划已过期，未发送任何机器人命令。请根据原任务重新规划后再审核。";
    }
    return error.message;
  }
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "麦克风权限被拒绝；请在浏览器站点设置中允许 H-IDE 使用麦克风";
    }
    if (error.name === "NotFoundError") return "没有检测到可用麦克风";
    if (error.name === "NotReadableError") return "麦克风正被其他应用占用，暂时无法录音";
  }
  if (error instanceof Error) return error.message;
  return "请求失败，请检查工作站连接";
}

function assistantUserFacingError(error: unknown): string {
  if (error instanceof ApiFailure) {
    if (error.status === 401 || error.status === 403) return "会话权限已失效，请重新登录后重试";
    if (error.status === 409) return "当前任务状态已变化，请刷新后重试";
    if (error.status === 429) return "服务暂时繁忙，请稍后重试";
    if (error.status >= 500) return "服务暂时不可用，请稍后重试";
  }
  return "暂时无法完成，请重试";
}

function conversationHasActiveWork(status: string | undefined): boolean {
  return new Set([
    "planning",
    "queued",
    "running",
    "review",
    "approved",
    "ready",
    "observing",
    "dispatching",
    "executing",
    "verifying",
    "needs_input",
    "awaiting_review",
    "suspended",
    "awaiting_user",
    "teaching",
  ]).has(status ?? "");
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

function robotReadyForDemo(robot: RobotState | undefined): boolean {
  return Boolean(
    robot
    && robot.connected
    && (robot.status === "online" || robot.status === "degraded"),
  );
}

function preferredDemoRobotId(robots: RobotState[], current: string): string {
  const selected = robots.find((robot) => robot.robotId === current);
  if (robotReadyForDemo(selected)) return selected!.robotId;
  const simulation = robots.find(
    (robot) => robot.robotId === "g1-sim" && robotReadyForDemo(robot),
  );
  return simulation?.robotId ?? preferredRobotId(robots, "");
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
  if (pathname === "/developer/scene-memory" || pathname.startsWith("/developer/scene-memory/")) return { kind: "scene-memory", shell: "developer" };
  if (pathname === "/app/scene-memory" || pathname.startsWith("/app/scene-memory/")) return { kind: "scene-memory", shell: "user" };
  if (pathname === "/developer/diagnostics" || pathname.startsWith("/developer/diagnostics/")) return { kind: "diagnostics", shell: "developer" };
  if (pathname === "/developer/robots" || pathname.startsWith("/developer/robots/")) return { kind: "robot-monitor", shell: "developer" };
  if (pathname === "/developer/simulation" || pathname.startsWith("/developer/simulation/")) return { kind: "simulation", shell: "developer" };
  if (pathname === "/developer/interfaces" || pathname.startsWith("/developer/interfaces/")) return { kind: "interfaces", shell: "developer" };
  const developerTask = pathname.match(/^\/developer\/tasks\/([^/]+)$/);
  if (developerTask) return { kind: "chat", shell: "developer", conversationId: decodeURIComponent(developerTask[1]) };
  if (pathname === "/developer" || pathname === "/developer/") return { kind: "chat", shell: "developer" };
  const task = pathname.match(/^\/(?:app\/)?tasks\/([^/]+)$/);
  if (task) return { kind: "chat", shell: "user", conversationId: decodeURIComponent(task[1]) };
  // PCD/scenes is intentionally absent from the product router. Legacy URLs
  // land in the assistant instead of exposing the archived workspace.
  return { kind: "chat", shell: "user" };
}

function chatPath(shell: "user" | "developer", conversationId?: string): string {
  const root = shell === "developer" ? "/developer" : "/app";
  return conversationId ? `${root}/tasks/${encodeURIComponent(conversationId)}` : root;
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

function AuthScreen({ onAuthenticated }: { onAuthenticated: (session: AuthSessionResponse) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void (async () => {
      setBusy(true);
      setError("");
      try {
        if (mode === "register") {
          await apiFetch<{ user: AuthPrincipal }>("/api/v1/auth/register", {
            method: "POST",
            body: JSON.stringify({ username, displayName: displayName.trim() || username, password }),
          });
        }
        const session = await apiFetch<AuthSessionResponse>("/api/v1/auth/login", {
          method: "POST",
          body: JSON.stringify({ username, password }),
        });
        if (!session?.principal?.userId) throw new Error("登录响应缺少用户身份");
        setCsrfToken(session.csrfToken);
        onAuthenticated(session);
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setBusy(false);
      }
    })();
  };

  return <main className="auth-shell">
    <section className="auth-intro"><span className="auth-logo"><img src="/assets/leader-demo/brand/h-ide-logo.png" alt="H-IDE" /></span><p className="eyebrow">H-IDE ASSISTANT</p><h1>让人与机器人自然协作</h1><p>聊天、视觉观察与机器人任务都在同一个连续会话里；真正的行动始终受权限与安全边界约束。</p><RobotAvatar speech="你好，我是 H-IDE。准备好开始今天的协作了吗？" /></section>
    <section className="auth-card" aria-label={mode === "login" ? "登录 H-IDE" : "注册 H-IDE"}>
      <div className="auth-tabs"><button className={mode === "login" ? "active" : ""} type="button" onClick={() => setMode("login")}>登录</button><button className={mode === "register" ? "active" : ""} type="button" onClick={() => setMode("register")}>注册</button></div>
      <form onSubmit={submit}>
        <div><h2>{mode === "login" ? "欢迎回来" : "创建账户"}</h2><p>{mode === "login" ? "使用工作站账户继续" : "注册后会以普通用户身份登录"}</p></div>
        <label>用户名<input name="username" autoComplete="username" required minLength={2} maxLength={64} value={username} onChange={(event) => setUsername(event.target.value)} /></label>
        {mode === "register" ? <label>显示名称<input name="displayName" autoComplete="name" maxLength={80} value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label> : null}
        <label>密码<input name="password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} required minLength={12} maxLength={256} value={password} onChange={(event) => setPassword(event.target.value)} /></label>
        {error ? <p className="auth-error" role="alert"><Icon name="warning" />{error}</p> : null}
        <button className="auth-submit" type="submit" disabled={busy}>{busy ? "请稍候…" : mode === "login" ? "登录" : "注册并登录"}</button>
      </form>
      <p className="auth-security">使用 HttpOnly 会话 Cookie 与 CSRF 双重提交保护；浏览器不会保存 API Bearer Token。</p>
    </section>
  </main>;
}

function App() {
  const [principal, setPrincipal] = useState<AuthPrincipal | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let cancelled = false;
    void apiFetch<AuthSessionResponse>("/api/v1/auth/me")
      .then((session) => {
        if (!cancelled && session?.principal?.userId) {
          setCsrfToken(session.csrfToken);
          setPrincipal(session.principal);
        }
      })
      .catch(() => {
        if (!cancelled) setPrincipal(null);
      })
      .finally(() => {
        if (!cancelled) setChecking(false);
      });
    return () => { cancelled = true; };
  }, []);

  const logout = useCallback(() => {
    void apiFetch<void>("/api/v1/auth/logout", { method: "POST" })
      .catch(() => undefined)
      .finally(() => {
        setCsrfToken("");
        setPrincipal(null);
        window.history.replaceState({}, "", "/app");
      });
  }, []);

  if (checking) return <div className="auth-loading" role="status"><span className="planning-pulse" />正在连接 H-IDE…</div>;
  if (!principal) return <AuthScreen onAuthenticated={(session) => {
    setPrincipal(session.principal);
  }} />;
  return <WorkspaceApp principal={principal} onLogout={logout} />;
}

function WorkspaceApp({ principal, onLogout }: { principal: AuthPrincipal; onLogout: () => void }) {
  const [route, navigate] = useAppRoute();
  const canUseSceneMemory = principal.scopes.includes("scene-memory:use");
  const canDevelop = (principal.role === "DEVELOPER" || principal.role === "ADMIN")
    && principal.scopes.includes("diagnostics:read");
  const developerMode = canDevelop && route.shell === "developer";
  const [robots, setRobots] = useState<RobotState[]>([]);
  const [selectedRobotId, setSelectedRobotId] = useState("");
  const [selectedDisplayRobotId, setSelectedDisplayRobotId] = useState<RobotDisplayOption["id"]>(() => {
    try {
      const value = window.localStorage.getItem("h-ide-leader-demo-robot");
      return value === "agibot" || value === "deeprobotics" || value === "unitree" ? value : "unitree";
    } catch {
      return "unitree";
    }
  });
  const [selectedSkill, setSelectedSkill] = useState<LeaderSkillId | null>(null);
  const [skillGreeting, setSkillGreeting] = useState("");
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
  const [assistantConversationsLoaded, setAssistantConversationsLoaded] = useState(false);
  const [assistantConversation, setAssistantConversation] = useState<AssistantConversation | null>(null);
  const [assistantText, setAssistantText] = useState("");
  const [assistantTurn, setAssistantTurn] = useState<AssistantTurn | null>(null);
  const [assistantBusy, setAssistantBusy] = useState(false);
  const [assistantThinking, setAssistantThinking] = useState(false);
  const [assistantInlineError, setAssistantInlineError] = useState("");
  const [assistantRetryRequest, setAssistantRetryRequest] = useState<AssistantRetryRequest | null>(null);
  const [assistantClientIssue, setAssistantClientIssue] = useState<AssistantClientIssue | null>(null);
  const [transcriptionId, setTranscriptionId] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const [voiceCaptureMode, setVoiceCaptureMode] = useState<VoiceCaptureMode>(() => {
    try {
      return window.localStorage.getItem("h-ide-offline-wav-fallback") === "true"
        ? "offline_wav"
        : "streaming";
    } catch {
      return "streaming";
    }
  });
  const [voiceReplyEnabled, setVoiceReplyEnabled] = useState(() => {
    try { return window.localStorage.getItem("h-ide-voice-replies") === "true"; } catch { return false; }
  });
  const [voiceUi, setVoiceUi] = useState<VoiceUiState>({
    phase: "idle",
    partial: "",
    metrics: { framesSent: 0, audioMsSent: 0 },
  });
  const [reviewFeedback, setReviewFeedback] = useState("");
  const [draft, setDraft] = useState<AssistantDraft | null>(null);
  const [missionDraft, setMissionDraft] = useState<MissionDraftRecordV2 | null>(null);
  const [missionRecord, setMissionRecord] = useState<MissionRecordV2 | null>(null);
  const [missionMessages, setMissionMessages] = useState<AssistantMessage[]>([]);
  const [missionHistory, setMissionHistory] = useState<MissionRecordV2[]>([]);
  const [missionDraftHistory, setMissionDraftHistory] = useState<MissionDraftRecordV2[]>([]);
  const [missionHistoryLoaded, setMissionHistoryLoaded] = useState(false);
  const [visualMission, setVisualMission] = useState<VisualMissionRecord | null>(null);
  const activeSceneContext: ActiveSceneContext | null = null;
  const [toolMenuOpen, setToolMenuOpen] = useState(false);
  const [controlPanelOpen, setControlPanelOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyLifecycle, setHistoryLifecycle] = useState<ConversationLifecycleView>("active");
  const [historyConversations, setHistoryConversations] = useState<AssistantConversation[]>([]);
  const [historyNextCursor, setHistoryNextCursor] = useState<string | null>(null);
  const [historySearch, setHistorySearch] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [deletedConversationUndo, setDeletedConversationUndo] = useState<AssistantConversation | null>(null);
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
  const streamingSpeechRef = useRef<BrowserStreamingSpeechSession | null>(null);
  const speechPlayerRef = useRef<StreamingPcmPlayer | null>(null);
  const voiceReplyEnabledRef = useRef(voiceReplyEnabled);
  voiceReplyEnabledRef.current = voiceReplyEnabled;
  const selectedSkillRef = useRef<LeaderSkillId | null>(selectedSkill);
  selectedSkillRef.current = selectedSkill;
  const finalTranscriptRef = useRef(new Set<string>());
  const pendingMissionInterventionRef = useRef(new Map<string, string>());
  // A poll that was already in flight when the user pressed cancel must not
  // resurrect the old running state after the cancellation response arrives.
  const cancelledAssistantTasksRef = useRef(new Set<string>());
  const deletedConversationUndoTimerRef = useRef<number | undefined>(undefined);
  const restoredConversationRef = useRef(false);
  const restoredMissionRef = useRef(false);
  const assistantConversationRef = useRef<AssistantConversation | null>(assistantConversation);
  assistantConversationRef.current = assistantConversation;
  const selectedRobotRef = useRef(selectedRobotId);
  selectedRobotRef.current = selectedRobotId;
  const selectedRobotManualRef = useRef(false);

  const selectedRobot = useMemo(
    () => robots.find((robot) => robot.robotId === selectedRobotId) ?? null,
    [robots, selectedRobotId],
  );

  const loadRobots = useCallback(async () => {
    const list = await apiFetch<RobotState[]>("/api/v1/robots");
    setRobots(list);
    setSelectedRobotId((current) => {
      // A first discovery response can contain only the stale g1-dev record;
      // once the simulation record arrives, prefer the connected g1-sim for
      // the leader demo unless the operator explicitly chose a robot.
      if (!selectedRobotManualRef.current) {
        return preferredDemoRobotId(list, current);
      }
      return preferredRobotId(list, current);
    });
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
    setMissionDraft(null);
    setMissionRecord(null);
    setMissionMessages([]);
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
    try {
      const page = await fetchAssistantConversationPage({ limit: 30 });
      setAssistantConversations(page.items);
      return page.items;
    } finally {
      setAssistantConversationsLoaded(true);
    }
  }, []);

  const loadConversationHistory = useCallback(async (
    lifecycle: ConversationLifecycleView,
    query: string,
    cursor: string | null = null,
    append = false,
  ) => {
    setHistoryLoading(true);
    try {
      const page = await fetchAssistantConversationPage({
        lifecycle,
        query,
        cursor,
        limit: 30,
      });
      setHistoryConversations((current) => append
        ? [
          ...current,
          ...page.items.filter((item) => !current.some(
            (existing) => existing.conversationId === item.conversationId,
          )),
        ]
        : page.items);
      setHistoryNextCursor(page.nextCursor);
      if (lifecycle === "active" && !query.trim() && !cursor) {
        setAssistantConversations(page.items);
      }
      return page;
    } finally {
      setHistoryLoading(false);
    }
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
    const latestMissionId = detail.activeMissionId ?? detail.missionIds?.at(-1) ?? null;
    const latestMissionDraftId = [...(detail.missionLinks ?? [])].reverse().find(
      (item) => item.draftStatus === "draft",
    )?.missionDraftId
      ?? detail.missionDraftIds?.at(-1)
      ?? detail.turns?.map((item) => item.result?.missionDraftId).filter(
        (item): item is string => typeof item === "string" && Boolean(item),
      ).at(-1)
      ?? null;
    if (latestMissionId) {
      try {
        const mission = await apiFetch<MissionRecordV2>(
          `/api/v2/missions/${encodeURIComponent(latestMissionId)}`,
        );
        setMissionRecord(mission);
        setMissionHistory((current) => [
          mission,
          ...current.filter((item) => item.missionId !== mission.missionId),
        ]);
      } catch (reason) {
        if (!(reason instanceof ApiFailure) || reason.status !== 404) throw reason;
      }
    }
    if (latestMissionDraftId && latestMissionDraftId !== latestMissionId) {
      try {
        const missionDraftRecord = await apiFetch<MissionDraftRecordV2>(
          `/api/v2/missions/drafts/${encodeURIComponent(latestMissionDraftId)}`,
        );
        setMissionDraft(missionDraftRecord);
        setMissionDraftHistory((current) => [
          missionDraftRecord,
          ...current.filter((item) => item.draftId !== missionDraftRecord.draftId),
        ]);
      } catch (reason) {
        if (!(reason instanceof ApiFailure) || reason.status !== 404) throw reason;
      }
    }
    return detail;
  }, [applyConversationDetail, loadVisualMission]);

  useEffect(() => {
    let cancelled = false;
    const developerRequests = developerMode
      ? Promise.all([loadMappings(), apiFetch<ControlEvent[]>("/api/v1/events")])
      : Promise.resolve<[void, ControlEvent[]]>([undefined, []]);
    void Promise.all([
      loadRobots(),
      loadMissionHistory(),
      developerRequests,
    ])
      .then(([, , [, initialEvents]]) => {
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
  }, [developerMode, loadMappings, loadMissionHistory, loadRobots]);

  useEffect(() => {
    if (!selectedRobot || !developerMode) {
      setCapabilities(null);
      return;
    }
    void loadCapabilities(selectedRobot.robotId);
  }, [developerMode, loadCapabilities, selectedRobot?.robotId]);

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
        const active = conversations.find((item) => conversationHasActiveWork(item.status));
        if (active) navigate(chatPath(initialRoute.shell, active.conversationId), { replace: true });
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
    if (!historyOpen) return;
    void loadConversationHistory(historyLifecycle, historySearch)
      .catch((reason) => setError(friendlyError(reason)));
  }, [historyLifecycle, historyOpen, historySearch, loadConversationHistory]);

  useEffect(() => {
    if (!missionHistoryLoaded || !assistantConversationsLoaded || restoredMissionRef.current) return;
    restoredMissionRef.current = true;
    const initialRoute = parseAppRoute(window.location.pathname);
    if (initialRoute.kind !== "chat" || initialRoute.conversationId) return;
    const activeMission = missionHistory.find(
      (item) => !["completed", "failed", "cancelled"].includes(item.snapshot.status),
    );
    const activeDraft = missionDraftHistory.find((item) => item.status === "draft");
    const linkedConversation = activeMission
      ? assistantConversations.find((item) => (
        item.activeMissionId === activeMission.missionId
        || item.missionIds?.includes(activeMission.missionId)
      ))
      : activeDraft
        ? assistantConversations.find((item) => item.missionDraftIds?.includes(activeDraft.draftId))
        : null;
    const identifier = linkedConversation?.conversationId
      ?? activeMission?.missionId
      ?? activeDraft?.draftId;
    if (identifier) navigate(chatPath(initialRoute.shell, identifier), { replace: true });
  }, [assistantConversations, assistantConversationsLoaded, missionDraftHistory, missionHistory, missionHistoryLoaded, navigate]);

  useEffect(() => {
    if (route.kind !== "chat") return;
    if (!route.conversationId) {
      setAssistantConversation(null);
      setAssistantTurn(null);
      setAssistantThinking(false);
      setAssistantInlineError("");
      setAssistantRetryRequest(null);
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
    void streamingSpeechRef.current?.cancel();
    streamingSpeechRef.current = null;
    void speechPlayerRef.current?.close();
    speechPlayerRef.current = null;
    if (deletedConversationUndoTimerRef.current !== undefined) {
      window.clearTimeout(deletedConversationUndoTimerRef.current);
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        "h-ide-offline-wav-fallback",
        String(voiceCaptureMode === "offline_wav"),
      );
    } catch {
      // Local preferences are optional in private browsing.
    }
  }, [voiceCaptureMode]);

  useEffect(() => {
    try { window.localStorage.setItem("h-ide-voice-replies", String(voiceReplyEnabled)); } catch {
      // Local preferences are optional in private browsing.
    }
  }, [voiceReplyEnabled]);

  useEffect(() => {
    try {
      window.localStorage.setItem("h-ide-leader-demo-robot", selectedDisplayRobotId);
    } catch {
      // Local preferences are optional in private browsing.
    }
  }, [selectedDisplayRobotId]);

  const selectLeaderSkill = useCallback((skill: LeaderSkillDefinition) => {
    if (skill.id === "inspection.office" && !assistantConversation && !missionDraft && !missionRecord) {
      const current = robots.find((item) => item.robotId === selectedRobotId);
      const demoRobotId = preferredDemoRobotId(robots, selectedRobotId);
      if (demoRobotId && demoRobotId !== selectedRobotId && !robotReadyForDemo(current)) {
        selectedRobotManualRef.current = false;
        setSelectedRobotId(demoRobotId);
        setNotice("已切换到在线仿真机器人 G1");
      }
    }
    setSelectedSkill(skill.id);
    setSkillGreeting(skill.prompt);
    setAssistantInlineError("");
    if (skill.id === "assistant.voice") {
      // The leader-demo voice path uses the browser's zh-CN SpeechSynthesis
      // contract, so it is available even when the optional streaming TTS
      // service is not configured.
      setVoiceReplyEnabled(true);
    }
  }, [assistantConversation, missionDraft, missionRecord, robots, selectedRobotId]);

  const clearLeaderSkill = useCallback(() => {
    setSelectedSkill(null);
    setSkillGreeting("");
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
      navigate(chatPath(route.shell, conversation.conversationId));
    }
    return conversation;
  }, [assistantConversation, navigate, route.shell]);

  const refreshConversation = useCallback(async (conversationId: string, robotId: string) => {
    const detail = await apiFetch<AssistantConversation>(
      `/api/v1/assistant/conversations/${encodeURIComponent(conversationId)}`,
    );
    if (detail?.conversationId && detail.robotId === robotId && selectedRobotRef.current === robotId) {
      setAssistantConversation((current) => {
        if (current?.conversationId !== detail.conversationId) return detail;
        const messages = mergeConversationMessages(
          detail.messages ?? [],
          current.messages ?? [],
        );
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
    let transientFailures = 0;
    for (let attempt = 0; attempt < 250; attempt += 1) {
      let turn: AssistantTurn;
      try {
        turn = await apiFetch<AssistantTurn>(`/api/v1/assistant/turns/${encodeURIComponent(taskId)}`);
        transientFailures = 0;
      } catch (reason) {
        const status = reason instanceof ApiFailure ? reason.status : undefined;
        if (status !== 401 && status !== 403 && status !== 404 && transientFailures < 5) {
          transientFailures += 1;
          await new Promise((resolve) => window.setTimeout(resolve, 400 * transientFailures));
          attempt -= 1;
          continue;
        }
        throw reason;
      }
      if (selectedRobotRef.current !== robotId) return turn;
      if (cancelledAssistantTasksRef.current.has(taskId)) return turn;
      setAssistantTurn(turn);
      if (isAssistantTurnTerminal(turn.status)) {
        if (turn.result?.missionDraftId) {
          const mission = await apiFetch<MissionDraftRecordV2>(
            `/api/v2/missions/drafts/${encodeURIComponent(turn.result.missionDraftId)}`,
          );
          setMissionDraft(mission);
          setMissionDraftHistory((current) => [
            mission,
            ...current.filter((item) => item.draftId !== mission.draftId),
          ]);
          setDraft(null);
          setNotice("行动方案已准备好，请确认后再执行");
        } else if (turn.draft) {
          setDraft(turn.draft);
          setNotice("多 Agent 计划已生成，等待人工审核");
        } else if (turn.status === "completed") {
          setNotice("");
        } else if (turn.status === "needs_input") {
          setNotice(turn.result?.assistantMessage ?? "调度 Agent 需要补充信息");
        } else if (turn.status === "cannot_plan") {
          setNotice(turn.result?.assistantMessage ?? "当前能力不足，无法规划任务");
        } else if (["failed", "unknown"].includes(turn.status)) {
          const detail = turn.error ?? "Agent 规划失败";
          setAssistantInlineError(
            turn.failure?.userMessage
              ?? "本轮没有完成，机器人没有移动；可以安全重试。",
          );
          setAssistantClientIssue({
            occurredAt: new Date().toISOString(),
            operation: "assistant.turn",
            detail,
            taskId: turn.taskId,
          });
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

  const speakAssistantReply = useCallback((text: string | undefined) => {
    const reply = text?.trim();
    if (!reply) return;
    if (selectedSkillRef.current === "assistant.voice") {
      // Keep the demo path local and deterministic: browser SpeechSynthesis
      // speaks the same text rendered in the transcript. It does not require
      // a cloud TTS credential and never changes robot-control authority.
      browserSpeakChinese(reply);
      return;
    }
    if (!voiceReplyEnabledRef.current) return;
    const player = speechPlayerRef.current ?? new StreamingPcmPlayer();
    speechPlayerRef.current = player;
    void player.speak(reply, fetchSpeechReply, {
      onFirstByte: (milliseconds) => setVoiceUi((current) => ({
        ...current,
        metrics: { ...current.metrics, ttsFirstByteMs: milliseconds },
      })),
      onError: (message) => setError(`语音回复未播放：${message}`),
    });
  }, []);

  const triggerLeaderWave = useCallback(async (robotId: string) => {
    if (selectedSkillRef.current !== "assistant.voice") return;
    try {
      await apiFetch<{ status: string }>("/api/v1/leader-demo/gestures/wave", {
        method: "POST",
        body: JSON.stringify({ robotId }),
      });
      setNotice("语音助手已回答，机器人正在挥手");
    } catch {
      // Speech remains a successful assistant interaction when the optional
      // local MuJoCo gesture channel is not running.
      setNotice("语音助手已回答（仿真挥手通道未连接）");
    }
  }, []);

  const sendAssistantMessage = useCallback(async (
    rawText: string,
    modality: "text" | "voice" = "text",
    sourceTranscriptionId: string | null = null,
  ) => {
    const text = rawText.trim();
    if (!text || assistantBusy) return;
    const retirePreviousMissionUi = Boolean(
      (missionDraft && missionDraft.status !== "draft")
      || (
        missionRecord
        && ["completed", "failed", "cancelled"].includes(
          missionRecord.snapshot.status,
        )
      ),
    );
    if (connection !== "live") {
      setAssistantInlineError("工作站当前离线，请恢复连接后重试");
      setAssistantRetryRequest({ text, modality, transcriptionId: sourceTranscriptionId, route: "conversation" });
      return;
    }
    setAssistantBusy(true);
    setAssistantThinking(true);
    setAssistantInlineError("");
    setAssistantRetryRequest({ text, modality, transcriptionId: sourceTranscriptionId, route: "conversation" });
    setError("");
    try {
      let robotId = selectedRobotId;
      if (selectedSkillRef.current === "inspection.office") {
        const selected = robots.find((item) => item.robotId === robotId);
        const demoRobotId = preferredDemoRobotId(robots, robotId);
        // A stale g1-dev record can survive a previous developer session. Do
        // not send a known Office inspection to an unbound robot; transparently
        // create the next conversation on the connected simulation G1.
        if (
          demoRobotId
          && demoRobotId !== robotId
          && !robotReadyForDemo(selected)
        ) {
          robotId = demoRobotId;
          selectedRobotManualRef.current = false;
          selectedRobotRef.current = robotId;
          setSelectedRobotId(robotId);
          setAssistantConversation(null);
          setAssistantTurn(null);
          setAssistantInlineError("");
          setNotice("已切换到在线仿真机器人 G1");
        }
      }
      const conversation = await ensureAssistantConversation(robotId);
      if (selectedRobotRef.current !== robotId) {
        throw new Error("机器人选择已变化，请重新提交任务");
      }
      const accepted = await apiFetch<{
        taskId: string;
        conversationId: string;
        status: string;
        userMessage?: AssistantMessage;
      }>(
        `/api/v1/assistant/conversations/${encodeURIComponent(conversation.conversationId)}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            text,
            modality,
            transcriptionId: sourceTranscriptionId,
            skillId: selectedSkillRef.current,
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
      // Mission cards are projections of one planning turn, not permanent
      // conversation chrome. Once the next turn is durably accepted, retire
      // terminal artifacts from the previous turn so a rejected/completed
      // plan cannot be mistaken for the response currently being prepared.
      // The durable draft, Mission, messages and evidence remain available in
      // conversation history and are not deleted here.
      if (retirePreviousMissionUi) {
        setMissionDraft(null);
        setMissionRecord(null);
        setMissionMessages([]);
      }
      setDraft(null);
      setReviewFeedback("");
      setAssistantConversation((current) => current ? {
        ...current,
        status: "planning",
        messages: mergeConversationMessages(
          current.messages ?? [],
          [accepted.userMessage ?? {
            messageId: makeId("message-local"),
            taskId: accepted.taskId,
            role: "user",
            content: text,
            modality,
            transcriptionId: sourceTranscriptionId,
            createdAt: new Date().toISOString(),
          }],
        ),
      } : current);
      setAssistantConversations((current) => current.map((item) => item.conversationId === conversation.conversationId
        ? { ...item, status: "planning", updatedAt: new Date().toISOString() }
        : item));
      const terminal = await monitorAssistantTurn(accepted.taskId, conversation.robotId);
      speakAssistantReply(terminal.result?.assistantMessage);
      if (terminal.status === "completed" && terminal.result?.assistantMessage) {
        void triggerLeaderWave(conversation.robotId);
      }
      if (!["failed", "unknown"].includes(terminal.status)) {
        setAssistantRetryRequest(null);
      }
    } catch (reason) {
      setAssistantInlineError(assistantUserFacingError(reason));
      setAssistantClientIssue({
        occurredAt: new Date().toISOString(),
        operation: "assistant.submit",
        detail: friendlyError(reason),
        taskId: assistantTurn?.taskId,
      });
    } finally {
      setAssistantThinking(false);
      setAssistantBusy(false);
    }
  }, [assistantBusy, assistantTurn?.taskId, connection, ensureAssistantConversation, missionDraft, missionRecord, monitorAssistantTurn, selectedRobotId, speakAssistantReply, triggerLeaderWave]);

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
    const officeReport = currentMissionOfficeReport(advanced.mission);
    const message = status === "completed"
      ? officeReport?.narrativeZhCN
        ?? "任务已完成。视觉模型只提供语义假设，发送给 Robot Agent 的目标点来自 RGB-D / 场景几何事实。"
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
    speakAssistantReply(message);
    return advanced;
  }, [speakAssistantReply]);

  const sendMissionMessage = useCallback(async (
    rawText: string,
    modality: "text" | "voice" = "text",
    sourceTranscriptionId: string | null = null,
    candidateSelection: {
      candidateId: string;
      selectionSourceAnalysisId: string;
    } | null = null,
    retrying = false,
  ) => {
    const text = rawText.trim();
    if (!text || assistantBusy) return;
    const activeMission = missionRecord
      && !["completed", "failed", "cancelled"].includes(missionRecord.snapshot.status)
      ? missionRecord
      : null;
    // Only an already-running Mission may bypass the conversation router for
    // an answer/revision. Every new message must return to ConversationGateway
    // so chat and QA can never accidentally become a robot plan.
    if (!activeMission) {
      await sendAssistantMessage(text, modality, sourceTranscriptionId);
      return;
    }
    if (connection !== "live") {
      setAssistantInlineError("工作站当前离线，请恢复连接后重试");
      setAssistantRetryRequest({ text, modality, transcriptionId: sourceTranscriptionId, route: "mission", candidateSelection });
      return;
    }
    setAssistantBusy(true);
    setAssistantThinking(true);
    setAssistantInlineError("");
    setAssistantRetryRequest({ text, modality, transcriptionId: sourceTranscriptionId, route: "mission", candidateSelection });
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
    const kind = activeMission.snapshot.status === "needs_input" ? "answer" : "modify_goal";
    const interventionFingerprint = JSON.stringify({
      missionId: activeMission.missionId,
      kind,
      text,
      candidateSelection,
    });
    const storageKey = pendingMissionInterventionStorageKey(
      principal.userId,
      activeMission.missionId,
    );
    const pendingKey = `${storageKey}\u0000${interventionFingerprint}`;
    const interventionId = pendingMissionInterventionRef.current.get(pendingKey)
      ?? storedMissionInterventionId(storageKey, interventionFingerprint)
      ?? makeId("intervention");
    pendingMissionInterventionRef.current.set(pendingKey, interventionId);
    storeMissionIntervention(storageKey, {
      fingerprint: interventionFingerprint,
      interventionId,
    });
    if (!retrying) setMissionMessages((current) => [...current, userMessage]);
    try {
      const response = await apiFetch<{
        decision: { kind?: string; reason?: string };
        mission: MissionRecordV2;
      }>(`/api/v2/missions/${encodeURIComponent(activeMission.missionId)}/messages`, {
        method: "POST",
        body: JSON.stringify({
          requestedBy: "operator",
          text,
          kind,
          interventionId,
          ...(candidateSelection ?? {}),
        }),
      });
      if (pendingMissionInterventionRef.current.get(pendingKey) === interventionId) {
        pendingMissionInterventionRef.current.delete(pendingKey);
      }
      clearStoredMissionIntervention(storageKey, {
        fingerprint: interventionFingerprint,
        interventionId,
      });
      setMissionRecord(response.mission);
      setMissionHistory((current) => [
        response.mission,
        ...current.filter((item) => item.missionId !== response.mission.missionId),
      ]);
      const reply = response.decision?.kind === "replan"
        ? "已结合你的补充生成持续任务修订；请审核变化后继续。"
        : response.decision?.reason ?? "已记录补充信息，等待人工处理。";
      setMissionMessages((current) => [
        ...current,
        {
          messageId: makeId("mission-message"),
          role: "assistant",
          content: reply,
          modality: "text",
          createdAt: new Date().toISOString(),
        },
      ]);
      speakAssistantReply(reply);
      setAssistantRetryRequest(null);
    } catch (reason) {
      setAssistantInlineError(assistantUserFacingError(reason));
      setAssistantClientIssue({
        occurredAt: new Date().toISOString(),
        operation: "mission.message",
        detail: friendlyError(reason),
      });
    } finally {
      setAssistantThinking(false);
      setAssistantBusy(false);
    }
  }, [assistantBusy, connection, missionRecord, principal.userId, sendAssistantMessage, speakAssistantReply]);

  const submitAssistantMessage = useCallback((event: FormEvent) => {
    event.preventDefault();
    if (missionRecord?.snapshot.status === "needs_input") {
      void sendMissionMessage(assistantText, "text", null);
      return;
    }
    void sendAssistantMessage(assistantText, "text", null);
  }, [assistantText, missionRecord?.snapshot.status, sendAssistantMessage, sendMissionMessage]);

  const retryAssistantRequest = useCallback(() => {
    if (!assistantRetryRequest || assistantBusy) return;
    setAssistantInlineError("");
    const request = assistantRetryRequest;
    if (request.route === "mission") {
      void sendMissionMessage(
        request.text,
        request.modality,
        request.transcriptionId,
        request.candidateSelection ?? null,
        true,
      );
    } else {
      void sendAssistantMessage(
        request.text,
        request.modality,
        request.transcriptionId,
      );
    }
  }, [assistantBusy, assistantRetryRequest, sendAssistantMessage, sendMissionMessage]);

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
    const taskId = assistantTurn.taskId;
    cancelledAssistantTasksRef.current.add(taskId);
    void (async () => {
      try {
        const turn = await apiFetch<AssistantTurn>(
          `/api/v1/assistant/turns/${encodeURIComponent(taskId)}/cancel`,
          { method: "POST" },
        );
        setAssistantTurn(turn);
        setAssistantThinking(false);
        setAssistantInlineError("");
        setAssistantRetryRequest(null);
        setNotice("规划已取消；人工控制仍可继续使用");
      } catch (reason) {
        setError(friendlyError(reason));
      } finally {
        setAssistantThinking(false);
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
        // Barge-in is deterministic: microphone intent cancels every queued
        // or playing TTS source before requesting capture permission.
        speechPlayerRef.current?.stop();
        setVoiceUi((current) => ({ ...current, partial: "" }));
        setVoiceBusy(true);
        try {
          if (voiceCaptureMode === "offline_wav") {
            const recorder = new BrowserWavRecorder();
            await recorder.start();
            recorderRef.current = recorder;
            setVoiceUi((current) => ({ ...current, phase: "listening" }));
            setRecording(true);
            setVoiceBusy(false);
            setNotice("离线 WAV 降级已启用；再次点击后批量转写并发送");
            return;
          }
          if (API_TOKEN) {
            throw new Error("浏览器流式语音要求 Cookie 登录；Bearer 工程模式只能显式使用离线 WAV 降级");
          }
          const csrf = csrfToken();
          if (!csrf) throw new Error("语音会话缺少 CSRF 凭证，请重新登录 H-IDE");
          const sessionKey = `speech-${Date.now()}-${Math.random().toString(16).slice(2)}`;
          let session: BrowserStreamingSpeechSession;
          session = new BrowserStreamingSpeechSession({
            url: speechWebSocketUrl(API_BASE),
            csrfToken: csrf,
            locale: "zh-CN",
            callbacks: {
              onPhase: (phase) => {
                setVoiceUi((current) => ({ ...current, phase }));
                if (["closed", "error"].includes(phase)) setRecording(false);
              },
              onPartial: ({ text }) => {
                // Partial text is intentionally ephemeral. It never mutates
                // the composer and therefore cannot be edited into, or sent
                // around, the ConversationGateway / PlanDraft boundary.
                setVoiceUi((current) => ({ ...current, partial: text }));
              },
              onFinal: ({ text, utteranceId, transcriptionId: issuedTranscriptionId }) => {
                const finalKey = `${sessionKey}:${utteranceId || text}`;
                if (finalTranscriptRef.current.has(finalKey)) return;
                if (!issuedTranscriptionId?.trim()) {
                  setError("流式语音最终转写缺少服务端签发的 transcriptionId，未提交消息");
                  setRecording(false);
                  setVoiceBusy(false);
                  void session.cancel();
                  return;
                }
                finalTranscriptRef.current.add(finalKey);
                setRecording(false);
                setVoiceBusy(true);
                setVoiceUi((current) => ({ ...current, phase: "finalizing", partial: text }));
                setNotice("语音已识别并直接发送给 H-IDE；机器人行动仍受 PlanDraft 审核约束");
                void session.finish().catch((reason) => setError(friendlyError(reason)));
                void (missionRecord?.snapshot.status === "needs_input"
                  ? sendMissionMessage(text, "voice", issuedTranscriptionId)
                  : sendAssistantMessage(text, "voice", issuedTranscriptionId))
                  .finally(() => {
                    setVoiceBusy(false);
                    setVoiceUi((current) => ({ ...current, partial: "" }));
                  });
              },
              onMetrics: (metrics) => setVoiceUi((current) => ({ ...current, metrics })),
              onError: (message) => {
                setError(message);
                setRecording(false);
                setVoiceBusy(false);
              },
            },
          });
          streamingSpeechRef.current = session;
          await session.start();
          setRecording(true);
          setVoiceBusy(false);
          setNotice("正在流式聆听；识别结果会直接发送，不会写入文本框");
        } catch (reason) {
          await streamingSpeechRef.current?.cancel().catch(() => undefined);
          streamingSpeechRef.current = null;
          setVoiceUi((current) => ({ ...current, phase: "error", partial: "" }));
          setVoiceBusy(false);
          setError(friendlyError(reason));
        }
        return;
      }

      setVoiceBusy(true);
      setRecording(false);
      try {
        if (voiceCaptureMode === "streaming") {
          const session = streamingSpeechRef.current;
          if (!session) throw new Error("流式语音会话已经关闭");
          const finalsBefore = finalTranscriptRef.current.size;
          await session.finish();
          streamingSpeechRef.current = null;
          if (finalTranscriptRef.current.size === finalsBefore) {
            setNotice("这次没有识别到有效语音，请靠近麦克风后重试");
            setVoiceUi((current) => ({ ...current, partial: "" }));
          }
          return;
        }

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
        setNotice("离线语音已批量转写并发送；任何机器人动作仍受 PlanDraft 约束");
        if (missionRecord?.snapshot.status === "needs_input") {
          await sendMissionMessage(transcript.text, "voice", transcript.transcriptionId);
        } else {
          await sendAssistantMessage(transcript.text, "voice", transcript.transcriptionId);
        }
      } catch (reason) {
        await recorderRef.current?.cancel();
        recorderRef.current = null;
        await streamingSpeechRef.current?.cancel().catch(() => undefined);
        streamingSpeechRef.current = null;
        setError(friendlyError(reason));
      } finally {
        setVoiceBusy(false);
      }
    })();
  }, [
    connection,
    missionRecord?.snapshot.status,
    recording,
    sendAssistantMessage,
    sendMissionMessage,
    voiceCaptureMode,
  ]);

  const toggleVoiceReplies = useCallback(() => {
    void (async () => {
      if (voiceReplyEnabled) {
        speechPlayerRef.current?.stop();
        setVoiceReplyEnabled(false);
        setNotice("语音回复已关闭");
        return;
      }
      try {
        const player = speechPlayerRef.current ?? new StreamingPcmPlayer();
        speechPlayerRef.current = player;
        await player.unlock();
        setVoiceReplyEnabled(true);
        setNotice("语音回复已开启；开始说话会立即打断播放");
      } catch (reason) {
        setError(`无法开启语音回复：${friendlyError(reason)}`);
      }
    })();
  }, [voiceReplyEnabled]);

  const toggleOfflineVoiceFallback = useCallback((enabled: boolean) => {
    void recorderRef.current?.cancel();
    recorderRef.current = null;
    void streamingSpeechRef.current?.cancel();
    streamingSpeechRef.current = null;
    setRecording(false);
    setVoiceBusy(false);
    setVoiceUi((current) => ({ ...current, phase: "idle", partial: "" }));
    setVoiceCaptureMode(enabled ? "offline_wav" : "streaming");
    setNotice(enabled
      ? "已显式启用离线 WAV 批量转写；它不是默认语音链路"
      : "已恢复 AudioWorklet 流式语音");
  }, []);

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
    setAssistantThinking(false);
    setAssistantInlineError("");
    setAssistantRetryRequest(null);
    setDraft(null);
    setMissionDraft(null);
    setMissionRecord(null);
    setMissionMessages([]);
    setAssistantText("");
    setTranscriptionId(null);
    setReviewFeedback("");
    setSelectedSkill(null);
    setSkillGreeting("");
    setError("");
    navigate(chatPath(route.shell));
  }, [navigate, route.shell, selectedRobotId]);

  const selectRobotForNewTask = useCallback((robotId: string) => {
    selectedRobotManualRef.current = true;
    if (robotId === selectedRobotId) return;
    startNewTask(robotId);
    setNotice(`已为 ${robotId} 打开新的任务会话；原会话上下文未被修改`);
  }, [selectedRobotId, startNewTask]);

  const openConversation = useCallback((conversationId: string) => {
    setHistoryOpen(false);
    navigate(chatPath(route.shell, conversationId));
  }, [navigate, route.shell]);

  const renameConversation = useCallback((conversation: AssistantConversation) => {
    const requested = window.prompt(
      "重命名会话",
      conversation.title ?? conversationLabel(conversation, conversation),
    );
    if (requested === null) return;
    const title = requested.trim();
    if (!title) {
      setError("会话名称不能为空");
      return;
    }
    void runRequest(async () => {
      const updated = await apiFetch<AssistantConversation>(
        `/api/v1/assistant/conversations/${encodeURIComponent(conversation.conversationId)}`,
        { method: "PATCH", body: JSON.stringify({ title }) },
      );
      setAssistantConversations((current) => current.map((item) => (
        item.conversationId === updated.conversationId ? { ...item, ...updated } : item
      )));
      setHistoryConversations((current) => current.map((item) => (
        item.conversationId === updated.conversationId ? { ...item, ...updated } : item
      )));
      setAssistantConversation((current) => current?.conversationId === updated.conversationId
        ? { ...current, ...updated }
        : current);
      setNotice(`会话已重命名为“${updated.title ?? title}”`);
    });
  }, [runRequest]);

  const hideConversationAfterLifecycleChange = useCallback((conversationId: string) => {
    setAssistantConversations((current) => current.filter(
      (item) => item.conversationId !== conversationId,
    ));
    setHistoryConversations((current) => current.filter(
      (item) => item.conversationId !== conversationId,
    ));
    if (route.kind === "chat" && route.conversationId === conversationId) startNewTask();
  }, [route, startNewTask]);

  const offerConversationUndo = useCallback((conversation: AssistantConversation) => {
    if (deletedConversationUndoTimerRef.current !== undefined) {
      window.clearTimeout(deletedConversationUndoTimerRef.current);
    }
    setDeletedConversationUndo(conversation);
    deletedConversationUndoTimerRef.current = window.setTimeout(() => {
      deletedConversationUndoTimerRef.current = undefined;
      setDeletedConversationUndo(null);
    }, 10_000);
  }, []);

  const archiveConversation = useCallback((conversation: AssistantConversation) => {
    void runRequest(async () => {
      await apiFetch<AssistantConversation>(
        `/api/v1/assistant/conversations/${encodeURIComponent(conversation.conversationId)}`,
        { method: "PATCH", body: JSON.stringify({ archived: true }) },
      );
      hideConversationAfterLifecycleChange(conversation.conversationId);
      setNotice("会话已归档；数据仍保留，可通过归档查询恢复");
    });
  }, [hideConversationAfterLifecycleChange, runRequest]);

  const deleteConversation = useCallback((conversation: AssistantConversation) => {
    void runRequest(async () => {
      if (
        conversation.activeMissionId
        || conversation.lifecycleBlockedReason === "active_mission"
      ) {
        throw new Error("机器人任务仍在执行，请先在会话中取消 Mission 并等待终止回执");
      }
      if (
        conversation.lifecycleBlockedReason === "active_turn"
        || ["planning", "queued", "running"].includes(conversation.status)
      ) {
        if (!window.confirm("当前消息仍在处理中。取消本次处理并将会话移入回收站？")) return;
        const detail = assistantConversation?.conversationId === conversation.conversationId
          ? assistantConversation
          : await apiFetch<AssistantConversation>(
            `/api/v1/assistant/conversations/${encodeURIComponent(conversation.conversationId)}`,
          );
        const activeTurn = detail.turns?.find((item) => ["queued", "running"].includes(item.status));
        if (activeTurn) {
          await apiFetch<AssistantTurn>(
            `/api/v1/assistant/turns/${encodeURIComponent(activeTurn.taskId)}/cancel`,
            { method: "POST" },
          );
        }
      } else if (!window.confirm("将这个会话移入回收站？消息、任务和视觉凭证都会保留。")) {
        return;
      }
      await apiFetch<AssistantConversation>(
        `/api/v1/assistant/conversations/${encodeURIComponent(conversation.conversationId)}`,
        { method: "DELETE" },
      );
      hideConversationAfterLifecycleChange(conversation.conversationId);
      offerConversationUndo(conversation);
      setNotice("会话已移入回收站");
    });
  }, [assistantConversation, hideConversationAfterLifecycleChange, offerConversationUndo, runRequest]);

  const restoreConversation = useCallback((conversation: AssistantConversation) => {
    void runRequest(async () => {
      const restored = await apiFetch<AssistantConversation>(
        `/api/v1/assistant/conversations/${encodeURIComponent(conversation.conversationId)}`,
        { method: "PATCH", body: JSON.stringify({ deleted: false, archived: false }) },
      );
      setAssistantConversations((current) => [
        restored,
        ...current.filter((item) => item.conversationId !== restored.conversationId),
      ]);
      setHistoryConversations((current) => historyLifecycle === "active" && !historySearch
        ? [
          restored,
          ...current.filter((item) => item.conversationId !== restored.conversationId),
        ]
        : current.filter((item) => item.conversationId !== restored.conversationId));
      setDeletedConversationUndo(null);
      if (deletedConversationUndoTimerRef.current !== undefined) {
        window.clearTimeout(deletedConversationUndoTimerRef.current);
        deletedConversationUndoTimerRef.current = undefined;
      }
      setNotice("会话已恢复");
    });
  }, [historyLifecycle, historySearch, runRequest]);

  const bulkDeleteConversations = useCallback((conversations: AssistantConversation[]) => {
    const conversationIds = conversations.map((item) => item.conversationId);
    if (!conversationIds.length) return;
    if (!window.confirm(`将选中的 ${conversationIds.length} 个已结束会话移入回收站？`)) return;
    void runRequest(async () => {
      const result = await apiFetch<AssistantConversationBulkDeleteResult>(
        "/api/v2/assistant/conversations/bulk-delete",
        { method: "POST", body: JSON.stringify({ conversationIds }) },
      );
      const deleted = new Set(result.deletedConversationIds);
      setAssistantConversations((current) => current.filter(
        (item) => !deleted.has(item.conversationId),
      ));
      setHistoryConversations((current) => current.filter(
        (item) => !deleted.has(item.conversationId),
      ));
      if (route.kind === "chat" && route.conversationId && deleted.has(route.conversationId)) {
        startNewTask();
      }
      setNotice(result.skipped.length
        ? `已清理 ${deleted.size} 个会话；${result.skipped.length} 个活动会话已跳过`
        : `已清理 ${deleted.size} 个会话`);
    });
  }, [route, runRequest, startNewTask]);

  const currentMissionEntry = useMemo<AssistantConversation | null>(() => {
    const current = missionRecord ?? missionDraft;
    if (!current) return null;
    const plan = current.plan;
    const conversationId = "missionId" in current ? current.missionId : current.draftId;
    const status = "snapshot" in current ? current.snapshot.status : current.status;
    const updatedAt = "snapshot" in current
      ? current.snapshot.updatedAt
      : current.updatedAt ?? plan.createdAt;
    return {
      conversationId,
      robotId: current.robotId,
      requestedBy: "operator",
      locale: "zh-CN",
      title: plan.objective,
      status,
      createdAt: plan.createdAt,
      updatedAt,
    };
  }, [missionDraft, missionRecord]);
  // The product history has one row per Assistant conversation. A current
  // legacy/orphan Mission may still occupy the active tab for safe recovery,
  // but it never becomes a duplicate history row.
  const allConversationEntries = assistantConversations;
  const currentConversationEntry = assistantConversation ?? currentMissionEntry;
  const conversationPath = chatPath(
    route.shell,
    missionRecord?.missionId ?? missionDraft?.draftId ?? assistantConversation?.conversationId,
  );

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

  if (route.shell === "developer" && !canDevelop) {
    return <main className="full-workspace"><div className="workspace-content"><section className="robot-monitor" role="alert">
      <h2>当前账号无开发者权限</h2>
      <p>机器人监控台需要开发者或管理员账号，以及诊断读取权限。请使用有权限的账号登录。</p>
      <button type="button" onClick={() => navigate("/app", { replace: true })}>返回普通对话</button>
    </section></div></main>;
  }

  return (
    <div className={`hide-app-shell shell-${developerMode ? "developer" : "user"} ${historyOpen ? "history-drawer-open" : ""}`}>
      <AppTitlebar
        current={currentConversationEntry}
        route={route}
        onBackToWorkspace={() => navigate(conversationPath)}
        onOpenConversation={openConversation}
        onOpenHistory={() => setHistoryOpen((value) => !value)}
        onOpenTools={() => setToolMenuOpen((value) => !value)}
        principal={principal}
        developerMode={developerMode}
        canDevelop={canDevelop}
        canUseTools={developerMode || canUseSceneMemory}
        robot={selectedRobot}
        connection={connection}
        taskStatus={activeTaskStatus}
        onStop={() => submitCommand("stop", {})}
        stopDisabled={!selectedRobot || connection !== "live" || busy || !capabilities?.capabilities.includes("stop")}
        onModeChange={(mode) => navigate(chatPath(mode, currentConversationEntry?.conversationId))}
        onLogout={onLogout}
      />

      {connection === "offline" ? <div className="global-connection-banner" role="alert">工作站连接已中断。离线期间不会排队发送规划、审核或控制命令。</div> : null}
      {error ? <div className="global-error-banner" role="alert"><Icon name="warning" /><span><strong>操作未完成</strong>{error}</span><button type="button" onClick={() => setError("")} aria-label="关闭错误"><Icon name="close" /></button></div> : null}

      {route.kind === "chat" ? <ChatWorkspace
        conversation={assistantConversation}
        missionMessages={missionMessages}
        robots={robots}
        robot={selectedRobot}
        selectedRobotId={selectedRobotId}
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
        selectedSkill={selectedSkill}
        skillGreeting={skillGreeting}
        selectedDisplayId={selectedDisplayRobotId}
        busy={assistantBusy}
        thinking={assistantThinking}
        inlineError={assistantInlineError}
        canCancelThinking={Boolean(assistantTurn && ["queued", "running"].includes(assistantTurn.status))}
        voiceBusy={voiceBusy}
        recording={recording}
        voiceUi={voiceUi}
        voiceCaptureMode={voiceCaptureMode}
        voiceReplyEnabled={voiceReplyEnabled}
        connection={connection}
        onSubmit={submitAssistantMessage}
        onVoice={toggleVoiceRecording}
        onToggleVoiceReplies={toggleVoiceReplies}
        onCancelPlanning={cancelAssistantTurn}
        onRetry={retryAssistantRequest}
        onConfirm={confirmDraft}
        onConfirmMission={confirmMissionDraft}
        onRejectMission={rejectMissionDraft}
        onContinueMission={continueMission}
        onCancelMission={cancelMission}
        onApproveMissionRevision={approveMissionRevision}
        onSelectMissionCandidate={(candidateId, analysisId, label) => {
          void sendMissionMessage(
            `选择候选：${label}`,
            "text",
            null,
            { candidateId, selectionSourceAnalysisId: analysisId },
          );
        }}
        onReject={rejectDraft}
        onRevise={reviseDraft}
        onReplan={replanExpiredDraft}
        onRunVisualMission={runVisualMission}
        onSelectVisualCandidate={selectVisualMissionCandidate}
        onCancelVisualMission={cancelVisualMission}
        onLoadVisualEvidence={loadVisualEvidence}
        onPrompt={setAssistantText}
        onSelectSkill={selectLeaderSkill}
        onClearSkill={clearLeaderSkill}
        onDisplayRobotChange={setSelectedDisplayRobotId}
        developerMode={developerMode}
        onRobotChange={selectRobotForNewTask}
      /> : null}

      {route.kind === "scene-memory" ? <WorkspacePage
        title="巡检与记忆"
        subtitle="用纯 Qwen VLM 观察环境，由人确认后保存场景记忆"
        onBack={() => navigate(conversationPath)}
      >
        <SceneMemoryWorkspace
          principal={principal}
          robots={robots}
          selectedRobotId={selectedRobotId}
          developerMode={developerMode}
          apiFetch={apiFetch}
          apiFetchBlob={apiFetchBlob}
          onBackToChat={() => navigate(conversationPath)}
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

      {developerMode && route.kind === "diagnostics" ? <WorkspacePage
        title="系统诊断"
        subtitle="只读检查 H-IDE、Agent、Vision、Robot Agent、Nav2 与仿真执行链"
        onBack={() => navigate(conversationPath)}
      >
        <DeveloperDiagnostics apiFetch={apiFetch} robotId={selectedRobotId || undefined} clientIssue={assistantClientIssue} />
      </WorkspacePage> : null}

      {developerMode && route.kind === "robot-monitor" ? <WorkspacePage
        title="机器人监控台"
        subtitle="所有已注册机器人配置的只读可用性检查"
        onBack={() => navigate(conversationPath)}
      >
        <RobotMonitor apiFetch={apiFetch} />
      </WorkspacePage> : null}

      {toolMenuOpen && (developerMode || canUseSceneMemory) ? <ToolMenu
        developerMode={developerMode}
        canUseSceneMemory={canUseSceneMemory}
        advancedMode={advancedMode}
        offlineVoiceFallback={voiceCaptureMode === "offline_wav"}
        onAdvancedModeChange={setAdvancedMode}
        onOfflineVoiceFallbackChange={toggleOfflineVoiceFallback}
        onClose={() => setToolMenuOpen(false)}
        onSceneMemory={() => {
          setToolMenuOpen(false);
          navigate(route.shell === "developer" ? "/developer/scene-memory" : "/app/scene-memory");
        }}
        onControl={() => { setToolMenuOpen(false); setControlPanelOpen(true); }}
        onSimulation={() => { setToolMenuOpen(false); navigate("/developer/simulation"); }}
        onDiagnostics={() => { setToolMenuOpen(false); navigate("/developer/diagnostics"); }}
        onRobotMonitor={() => { setToolMenuOpen(false); navigate("/developer/robots"); }}
        onHistory={() => { setToolMenuOpen(false); setHistoryOpen(true); }}
        onInterfaces={() => { setToolMenuOpen(false); navigate("/developer/interfaces"); }}
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

      {historyOpen ? <SidePanel title="任务历史" subtitle={`${historyConversations.length} 个会话${historyNextCursor ? " · 还有更多" : ""}`} onClose={() => setHistoryOpen(false)} side="left" showClose={false}>
        <HistoryPanel
          conversations={historyConversations}
          currentId={currentConversationEntry?.conversationId}
          onOpen={openConversation}
          onNew={() => { startNewTask(); }}
          onRename={renameConversation}
          onArchive={archiveConversation}
          onDelete={deleteConversation}
          onRestore={restoreConversation}
          onBulkDelete={bulkDeleteConversations}
          lifecycle={historyLifecycle}
          onLifecycleChange={setHistoryLifecycle}
          search={historySearch}
          onSearch={setHistorySearch}
          loading={historyLoading}
          hasMore={Boolean(historyNextCursor)}
          onLoadMore={() => {
            if (!historyNextCursor) return;
            void loadConversationHistory(
              historyLifecycle,
              historySearch,
              historyNextCursor,
              true,
            ).catch((reason) => setError(friendlyError(reason)));
          }}
          developerMode={developerMode}
        />
      </SidePanel> : null}

      {notice ? <div className="app-toast" role="status"><Icon name="check" /><span>{notice}</span>{deletedConversationUndo && notice === "会话已移入回收站" ? <button className="toast-action" type="button" onClick={() => restoreConversation(deletedConversationUndo)}>撤销</button> : null}<button type="button" onClick={() => { setNotice(""); setDeletedConversationUndo(null); }} aria-label="关闭提示"><Icon name="close" /></button></div> : null}
    </div>
  );
}

type IconName =
  | "logo" | "home" | "plus" | "history" | "tools" | "robot" | "map" | "stop"
  | "close" | "warning" | "check" | "mic" | "send" | "attach" | "control"
  | "interface" | "refresh" | "back" | "spark" | "chevron" | "activity" | "volume";

function Icon({ name }: { name: IconName }) {
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  const paths: Record<IconName, React.ReactNode> = {
    logo: <><path d="M6 7.5 12 4l6 3.5v7L12 18l-6-3.5z" /><path d="m9 9 3-1.7L15 9v3.5L12 14l-3-1.5z" /></>,
    home: <><path d="m3.5 10.5 8.5-7 8.5 7" /><path d="M5.5 9.5V20h13V9.5M9.5 20v-6h5v6" /></>,
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
    volume: <><path d="M4 10v4h4l5 4V6l-5 4z" /><path d="M16 9a4 4 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11" /></>,
  };
  return <svg className="app-icon" viewBox="0 0 24 24" aria-hidden="true" {...common}>{paths[name]}</svg>;
}

function conversationLabel(conversation: AssistantConversation, current: AssistantConversation | null): string {
  const detail = current?.conversationId === conversation.conversationId ? current : conversation;
  const title = detail.title?.trim();
  if (title) return title;
  const firstUserMessage = detail.messages?.find((message) => message.role === "user")?.content.trim();
  if (firstUserMessage) return firstUserMessage.length > 22 ? `${firstUserMessage.slice(0, 22)}…` : firstUserMessage;
  return `${conversation.robotId} 的新任务`;
}

function AppTitlebar({
  current,
  route,
  onBackToWorkspace,
  onOpenConversation,
  onOpenHistory,
  onOpenTools,
  principal,
  developerMode,
  canDevelop,
  canUseTools,
  robot,
  connection,
  taskStatus,
  onStop,
  stopDisabled,
  onModeChange,
  onLogout,
}: {
  current: AssistantConversation | null;
  route: AppRoute;
  onBackToWorkspace: () => void;
  onOpenConversation: (conversationId: string) => void;
  onOpenHistory: () => void;
  onOpenTools: () => void;
  principal: AuthPrincipal;
  developerMode: boolean;
  canDevelop: boolean;
  canUseTools: boolean;
  robot: RobotState | null;
  connection: ConnectionState;
  taskStatus: string;
  onStop: () => void;
  stopDisabled: boolean;
  onModeChange: (mode: "user" | "developer") => void;
  onLogout: () => void;
}) {
  const pose = poseFreshness(robot);
  const [routeTitleLeaving, setRouteTitleLeaving] = useState(false);
  const workspaceTitles: Partial<Record<AppRoute["kind"], [string, string]>> = {
    "scene-memory": ["巡检与记忆", "用纯 Qwen VLM 观察环境，由人确认后保存场景记忆"],
    simulation: ["仿真观测台", "MuJoCo + SONIC、Nav2 联调与连续任务回放"],
    interfaces: ["接口中心", "逻辑动作与代理接口的版本化映射"],
    diagnostics: ["系统诊断", "只读检查 H-IDE、Agent、Vision、Robot Agent、Nav2 与仿真执行链"],
    "robot-monitor": ["机器人监控台", "所有已注册机器人配置的只读可用性检查"],
  };
  const workspaceTitle = workspaceTitles[route.kind];
  useEffect(() => {
    setRouteTitleLeaving(false);
  }, [route.kind]);
  const leaveWorkspace = () => {
    if (routeTitleLeaving) return;
    setRouteTitleLeaving(true);
    window.setTimeout(onBackToWorkspace, 180);
  };
  return <header className="app-titlebar">
    <button className="brand-button" type="button" onClick={onOpenHistory} aria-label="切换任务历史" title="切换任务历史"><Icon name="home" /></button>
    {workspaceTitle ? <div className={`route-titlebar${routeTitleLeaving ? " is-leaving" : ""}`}>
      <button className="icon-button" type="button" onClick={leaveWorkspace} aria-label="返回会话"><Icon name="back" /></button>
      <div><strong>{workspaceTitle[0]}</strong></div>
    </div> : <nav className="mission-tabs" aria-label="活动任务">
      {current ? <button
        className={route.kind === "chat" ? "active" : ""}
        type="button"
        onClick={() => onOpenConversation(current.conversationId)}
      ><span className={`tab-state tab-${current.status}`} />{conversationLabel(current, current)}</button> : null}
    </nav>}
    {developerMode ? <div className="titlebar-contexts" aria-label="当前控制上下文">
      <span className={`titlebar-context connection-${connection}`}><span className="context-dot" />{connection === "live" ? "工作站在线" : connection === "connecting" ? "正在连接" : "工作站离线"}</span>
      <span className={`titlebar-context ${pose.stale ? "stale" : "ready"}`}><Icon name="activity" />{pose.label}</span>
      <span className="titlebar-context task-context">任务 <StatusPill value={taskStatus} /></span>
      <button className="global-stop-button titlebar-stop-button" type="button" onClick={onStop} disabled={stopDisabled}><Icon name="stop" />停止机器人</button>
    </div> : null}
    <div className="titlebar-actions">
      {canDevelop ? <div className="shell-switch" aria-label="界面模式">
        <button className={!developerMode ? "active" : ""} type="button" onClick={() => onModeChange("user")}>用户</button>
        <button className={developerMode ? "active" : ""} type="button" onClick={() => onModeChange("developer")}>开发者</button>
      </div> : null}
      {canUseTools ? <button className="icon-button titlebar-tools-button" type="button" onClick={onOpenTools} aria-label="打开工具" title="工具">
        <Icon name="tools" /><span>工具</span>
      </button> : null}
      <span className="account-chip" title={`${principal.username} · ${principal.role}`}><span>{principal.displayName.slice(0, 1).toUpperCase()}</span><b>{principal.displayName}</b></span>
      <button className="logout-button" type="button" onClick={onLogout}>退出</button>
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

function WorkspacePage({ title, subtitle, onBack, children }: { title: string; subtitle: string; onBack: () => void; children: React.ReactNode }) {
  return <main className="full-workspace">
    <div className="workspace-content">{children}</div>
  </main>;
}

function ToolMenu({
  developerMode,
  canUseSceneMemory,
  advancedMode,
  offlineVoiceFallback,
  onAdvancedModeChange,
  onOfflineVoiceFallbackChange,
  onClose,
  onSceneMemory,
  onControl,
  onSimulation,
  onDiagnostics,
  onRobotMonitor,
  onHistory,
  onInterfaces,
  onRefresh,
}: {
  developerMode: boolean;
  canUseSceneMemory: boolean;
  advancedMode: boolean;
  offlineVoiceFallback: boolean;
  onAdvancedModeChange: (value: boolean) => void;
  onOfflineVoiceFallbackChange: (value: boolean) => void;
  onClose: () => void;
  onSceneMemory: () => void;
  onControl: () => void;
  onSimulation: () => void;
  onDiagnostics: () => void;
  onRobotMonitor: () => void;
  onHistory: () => void;
  onInterfaces: () => void;
  onRefresh: () => void;
}) {
  return <><button className="popover-scrim" type="button" onClick={onClose} aria-label="关闭工具" /><aside className="tool-menu" aria-label="H-IDE 工具">
    <div className="tool-menu-header"><span>工具</span><button className="icon-button" type="button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button></div>
    {canUseSceneMemory ? <button type="button" onClick={onSceneMemory}><span className="tool-icon"><Icon name="map" /></span><span><strong>巡检与记忆</strong><small>带机器人认识环境，确认它应该记住什么</small></span><Icon name="chevron" /></button> : null}
    {developerMode ? <button type="button" onClick={onControl}><span className="tool-icon"><Icon name="control" /></span><span><strong>手动控制</strong><small>测试高层动作与停止</small></span><Icon name="chevron" /></button> : null}
    {developerMode ? <button type="button" onClick={onSimulation}><span className="tool-icon"><Icon name="activity" /></span><span><strong>仿真观测台</strong><small>连续任务、障碍与回放</small></span><Icon name="chevron" /></button> : null}
    {developerMode ? <button type="button" onClick={onDiagnostics}><span className="tool-icon"><Icon name="activity" /></span><span><strong>系统诊断</strong><small>执行链存活、就绪度与具体故障</small></span><Icon name="chevron" /></button> : null}
    {developerMode ? <button type="button" onClick={onRobotMonitor}><span className="tool-icon"><Icon name="activity" /></span><span><strong>机器人监控台</strong><small>查看所有机器人配置与可用状态</small></span><Icon name="chevron" /></button> : null}
    <button type="button" onClick={onHistory}><span className="tool-icon"><Icon name="history" /></span><span><strong>任务历史</strong><small>恢复规划、审核和执行记录</small></span><Icon name="chevron" /></button>
    {developerMode ? <button type="button" onClick={onRefresh}><span className="tool-icon"><Icon name="refresh" /></span><span><strong>刷新状态</strong><small>重新读取机器人与接口快照</small></span><Icon name="chevron" /></button> : null}
    <div className="tool-separator" />
    <label className="advanced-toggle"><span><strong>离线 WAV 转写降级</strong><small>显式启用批量 Whisper；默认使用低延迟流式语音</small></span><input type="checkbox" checked={offlineVoiceFallback} onChange={(event) => onOfflineVoiceFallbackChange(event.target.checked)} /><i /></label>
    {developerMode ? <label className="advanced-toggle"><span><strong>工程模式</strong><small>显示接口与调试工具</small></span><input type="checkbox" checked={advancedMode} onChange={(event) => onAdvancedModeChange(event.target.checked)} /><i /></label> : null}
    {developerMode && advancedMode ? <button type="button" onClick={onInterfaces}><span className="tool-icon"><Icon name="interface" /></span><span><strong>接口中心</strong><small>映射版本、探测与回滚</small></span><Icon name="chevron" /></button> : null}
  </aside></>;
}

function SidePanel({ title, subtitle, onClose, side = "right", showClose = true, children }: { title: string; subtitle: string; onClose: () => void; side?: "left" | "right"; showClose?: boolean; children: React.ReactNode }) {
  return <div className="side-panel-layer"><button className="side-panel-scrim" type="button" onClick={onClose} aria-label={`关闭${title}`} /><aside className={`side-panel side-${side}`} aria-label={title}>
    <header><div><h2>{title}</h2><p>{subtitle}</p></div>{showClose ? <button className="icon-button" type="button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button> : null}</header>
    <div className="side-panel-content">{children}</div>
  </aside></div>;
}

const BULK_DELETABLE_CONVERSATION_STATUSES = new Set([
  "idle",
  "completed",
  "failed",
  "cancelled",
  "rejected",
  "cannot_plan",
  "expired",
]);

function HistoryPanel({
  conversations,
  currentId,
  onOpen,
  onNew,
  onRename,
  onArchive,
  onDelete,
  onRestore,
  onBulkDelete,
  lifecycle,
  onLifecycleChange,
  search,
  onSearch,
  loading,
  hasMore,
  onLoadMore,
  developerMode,
}: {
  conversations: AssistantConversation[];
  currentId?: string;
  onOpen: (conversationId: string) => void;
  onNew: () => void;
  onRename: (conversation: AssistantConversation) => void;
  onArchive: (conversation: AssistantConversation) => void;
  onDelete: (conversation: AssistantConversation) => void;
  onRestore: (conversation: AssistantConversation) => void;
  onBulkDelete: (conversations: AssistantConversation[]) => void;
  lifecycle: ConversationLifecycleView;
  onLifecycleChange: (lifecycle: ConversationLifecycleView) => void;
  search: string;
  onSearch: (value: string) => void;
  loading: boolean;
  hasMore: boolean;
  onLoadMore: () => void;
  developerMode: boolean;
}) {
  const [searchDraft, setSearchDraft] = useState(search);
  const [managing, setManaging] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  useEffect(() => setSearchDraft(search), [search]);
  useEffect(() => {
    setManaging(false);
    setSelected(new Set());
  }, [lifecycle]);
  const deletable = conversations.filter((conversation) => (
    conversation.lifecycleMutable !== false
    && BULK_DELETABLE_CONVERSATION_STATUSES.has(conversation.status)
  ));
  const selectedConversations = conversations.filter((conversation) => selected.has(conversation.conversationId));
  const toggleSelected = (conversationId: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(conversationId)) next.delete(conversationId);
      else next.add(conversationId);
      return next;
    });
  };
  return <div className="history-panel">
    <button className="new-history-task" type="button" onClick={onNew}><Icon name="plus" />开始新任务</button>
    <div className="history-view-tabs" role="tablist" aria-label="历史记录分类">
      {(["active", "archived", "deleted"] as const).map((value) => <button
        className={lifecycle === value ? "active" : ""}
        type="button"
        role="tab"
        aria-selected={lifecycle === value}
        onClick={() => onLifecycleChange(value)}
        key={value}
      >{value === "active" ? "会话" : value === "archived" ? "归档" : "回收站"}</button>)}
    </div>
    <form className="history-search" onSubmit={(event) => { event.preventDefault(); onSearch(searchDraft.trim()); }}>
      <input aria-label="搜索会话标题" value={searchDraft} onChange={(event) => setSearchDraft(event.target.value)} maxLength={120} placeholder="搜索标题或消息…" />
      <button type="submit">搜索</button>
      {search ? <button type="button" onClick={() => { setSearchDraft(""); onSearch(""); }}>清除</button> : null}
    </form>
    <p className="history-contract-note">{lifecycle === "deleted"
      ? "回收站记录不会参与普通会话列表；恢复后消息、任务和视觉凭证仍保持关联。"
      : lifecycle === "archived"
        ? "归档会话保留完整上下文，可随时恢复到普通会话列表。"
        : "每行代表一条完整会话；Mission、草案和视觉凭证不再重复占用历史列表。"}</p>
    {lifecycle === "active" ? <div className="history-manage-bar">
      <button type="button" onClick={() => { setManaging((value) => !value); setSelected(new Set()); }}>{managing ? "完成" : "管理历史"}</button>
      {managing ? <><button type="button" onClick={() => setSelected(new Set(deletable.map((item) => item.conversationId)))}>选择所有已结束</button><button className="history-delete-selected" type="button" disabled={!selectedConversations.length} onClick={() => onBulkDelete(selectedConversations)}>删除所选 ({selectedConversations.length})</button></> : null}
    </div> : null}
    <div className="history-list" aria-busy={loading}>
      {conversations.length ? conversations.map((conversation) => {
        const bulkEligible = conversation.lifecycleMutable !== false
          && BULK_DELETABLE_CONVERSATION_STATUSES.has(conversation.status);
        return <div className={`history-row ${conversation.conversationId === currentId ? "active" : ""}`} key={conversation.conversationId}>
          {managing ? <label className="history-select"><input type="checkbox" checked={selected.has(conversation.conversationId)} disabled={!bulkEligible} onChange={() => toggleSelected(conversation.conversationId)} /><span /></label> : null}
          <button className="history-open" type="button" onClick={() => lifecycle === "active" ? onOpen(conversation.conversationId) : onRestore(conversation)}>
            <span className={`history-state tab-${conversation.status}`}><Icon name="spark" /></span><span><strong>{conversationLabel(conversation, conversation.conversationId === currentId ? conversation : null)}</strong><small>{developerMode ? `${conversation.robotId} · ` : ""}{formatDateTime(conversation.updatedAt ?? conversation.createdAt)}</small></span><StatusPill value={conversation.status} />
          </button>
          {!managing ? <details className="history-actions"><summary role="button" aria-label={`管理会话：${conversationLabel(conversation, null)}`}>•••</summary><div>{lifecycle === "active" ? <><button type="button" onClick={() => onRename(conversation)}>重命名</button><button type="button" onClick={() => onArchive(conversation)}>归档</button><button className="danger" type="button" onClick={() => onDelete(conversation)}>删除</button></> : <button type="button" onClick={() => onRestore(conversation)}>恢复会话</button>}</div></details> : null}
        </div>;
      }) : !loading ? <EmptyState>{search ? "没有匹配的会话。" : lifecycle === "deleted" ? "回收站为空。" : lifecycle === "archived" ? "还没有归档会话。" : "还没有任务会话。发送第一条消息后，会话会保存在这里。"}</EmptyState> : null}
      {loading ? <div className="history-loading" role="status"><span className="planning-pulse" />正在读取会话…</div> : null}
    </div>
    {hasMore ? <button className="history-load-more" type="button" onClick={onLoadMore} disabled={loading}>加载更多</button> : null}
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

function ConversationVisualEvidence({ turn, developerMode }: { turn: AssistantTurn; developerMode: boolean }) {
  const [imageUrl, setImageUrl] = useState("");
  const [loadError, setLoadError] = useState("");
  const evidence = turn.result?.evidence ?? [];
  const image = evidence.find((item) => {
    const mediaType = typeof item.mediaType === "string" ? item.mediaType : "";
    return mediaType.startsWith("image/") && typeof item.proxyUrl === "string";
  });
  const proxyUrl = typeof image?.proxyUrl === "string" ? image.proxyUrl : "";
  const expectedPrefix = `/api/v1/assistant/turns/${encodeURIComponent(turn.taskId)}/visual-evidence/`;

  useEffect(() => {
    if (!proxyUrl || !proxyUrl.startsWith(expectedPrefix)) return undefined;
    let disposed = false;
    let objectUrl = "";
    setLoadError("");
    void apiFetchBlob(proxyUrl)
      .then((blob) => {
        if (disposed) return;
        objectUrl = URL.createObjectURL(blob);
        setImageUrl(objectUrl);
      })
      .catch((reason) => {
        if (!disposed) setLoadError(friendlyError(reason));
      });
    return () => {
      disposed = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [expectedPrefix, proxyUrl]);

  if (!evidence.length && !turn.result?.visualResult) return null;
  const visual = turn.result?.visualResult;
  const hypotheses = visual && Array.isArray(visual.hypotheses)
    ? visual.hypotheses.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    : [];
  const labels = hypotheses.flatMap((item) => typeof item.label === "string" ? [item.label] : []).slice(0, 8);
  return <section className="conversation-evidence-card" aria-label="视觉证据">
    <header><span><Icon name="activity" /></span><div><strong>我看到的画面</strong><small>{labels.length ? `识别到：${labels.join("、")}` : "视觉观察凭证"}</small></div></header>
    {imageUrl ? <img src={imageUrl} alt="机器人本次视觉观察画面" /> : <div className="evidence-loading">{loadError || "正在载入视觉凭证…"}</div>}
    {developerMode ? <details><summary>视觉处理详情</summary><pre>{JSON.stringify({ taskId: turn.taskId, evidence, visualResult: visual }, null, 2)}</pre></details> : null}
  </section>;
}

function ThinkingIndicator({ canCancel, onCancel }: { canCancel: boolean; onCancel: () => void }) {
  return <div className="thinking-indicator" role="status" aria-live="polite" aria-label="H-IDE 正在思考">
    <span className="thinking-orbit" aria-hidden="true"><Icon name="spark" /><i /></span>
    <span>Thinking…</span>
    {canCancel ? <button type="button" onClick={onCancel}>取消</button> : null}
  </div>;
}

function AssistantInlineFailure({ message, onRetry }: { message: string; onRetry: () => void }) {
  return <div className="assistant-inline-failure" role="alert">
    <Icon name="warning" />
    <span>{message}</span>
    <button type="button" onClick={onRetry}>重试</button>
  </div>;
}

function ChatWorkspace({
  conversation,
  missionMessages,
  robots,
  robot,
  selectedRobotId,
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
  selectedSkill,
  skillGreeting,
  selectedDisplayId,
  busy,
  thinking,
  inlineError,
  canCancelThinking,
  voiceBusy,
  recording,
  voiceUi,
  voiceCaptureMode,
  voiceReplyEnabled,
  connection,
  onSubmit,
  onVoice,
  onToggleVoiceReplies,
  onCancelPlanning,
  onRetry,
  onConfirm,
  onConfirmMission,
  onRejectMission,
  onContinueMission,
  onCancelMission,
  onApproveMissionRevision,
  onSelectMissionCandidate,
  onReject,
  onRevise,
  onReplan,
  onRunVisualMission,
  onSelectVisualCandidate,
  onCancelVisualMission,
  onLoadVisualEvidence,
  onPrompt,
  onSelectSkill,
  onClearSkill,
  onDisplayRobotChange,
  developerMode,
  onRobotChange,
}: {
  conversation: AssistantConversation | null;
  missionMessages: AssistantMessage[];
  robots: RobotState[];
  robot: RobotState | null;
  selectedRobotId: string;
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
  selectedSkill: LeaderSkillId | null;
  skillGreeting: string;
  selectedDisplayId: RobotDisplayOption["id"];
  busy: boolean;
  thinking: boolean;
  inlineError: string;
  canCancelThinking: boolean;
  voiceBusy: boolean;
  recording: boolean;
  voiceUi: VoiceUiState;
  voiceCaptureMode: VoiceCaptureMode;
  voiceReplyEnabled: boolean;
  connection: ConnectionState;
  onSubmit: (event: FormEvent) => void;
  onVoice: () => void;
  onToggleVoiceReplies: () => void;
  onCancelPlanning: () => void;
  onRetry: () => void;
  onConfirm: () => void;
  onConfirmMission: () => void;
  onRejectMission: () => void;
  onContinueMission: () => void;
  onCancelMission: () => void;
  onApproveMissionRevision: () => void;
  onSelectMissionCandidate: (
    candidateId: string,
    analysisId: string,
    label: string,
  ) => void;
  onReject: () => void;
  onRevise: () => void;
  onReplan: () => void;
  onRunVisualMission: () => void;
  onSelectVisualCandidate: (candidateId: string) => void;
  onCancelVisualMission: () => void;
  onLoadVisualEvidence: (missionId: string, sha256: string) => Promise<string>;
  onPrompt: (prompt: string) => void;
  onSelectSkill: (skill: LeaderSkillDefinition) => void;
  onClearSkill: () => void;
  onDisplayRobotChange: (displayId: RobotDisplayOption["id"]) => void;
  developerMode: boolean;
  onRobotChange: (robotId: string) => void;
}) {
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const missionLink = (conversation?.missionLinks ?? []).find((item) => (
    mission && item.missionId === mission.missionId
  )) ?? null;
  const missionDraftLink = (conversation?.missionLinks ?? []).find((item) => (
    missionDraft && item.missionDraftId === missionDraft.draftId
  )) ?? null;
  const missionTaskId = missionLink?.taskId
    ?? (turn && turn.result?.missionDraftId === mission?.missionId ? turn.taskId : null);
  const missionDraftTaskId = missionDraftLink?.taskId
    ?? (turn && turn.result?.missionDraftId === missionDraft?.draftId ? turn.taskId : null);
  const legacyTaskId = draft?.taskId ?? null;
  const activityTaskId = turn?.taskId
    ?? missionTaskId
    ?? missionDraftTaskId
    ?? legacyTaskId;
  const localMessages = missionMessages.map((message) => ({
    ...message,
    taskId: message.taskId ?? missionTaskId,
    turnSequence: message.turnSequence ?? missionLink?.turnSequence ?? null,
  }));
  const timeline = buildConversationTimeline(conversation, turn, localMessages);
  const messages = timeline.flatMap((item) => item.messages);
  const missionAttached = missionTaskId !== null
    && timeline.some((item) => item.taskId === missionTaskId);
  const missionDraftAttached = missionDraftTaskId !== null
    && timeline.some((item) => item.taskId === missionDraftTaskId);
  const legacyAttached = legacyTaskId !== null
    && timeline.some((item) => item.taskId === legacyTaskId);
  const empty = !timeline.length && !draft && !missionDraft && !mission;
  useEffect(() => {
    const element = timelineRef.current;
    if (!element) return;
    if (typeof element.scrollTo === "function") element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
    else element.scrollTop = element.scrollHeight;
  }, [busy, draft?.draftId, inlineError, messages.length, mission?.snapshot.lastSequence, missionDraft?.draftId, thinking, turn?.status]);
  const suggestions = ["你好，介绍一下你自己", "机器人现在状态怎么样？", "只观察当前画面，告诉我看到了什么", "去会议桌旁边"];
  const activeSkill = LEADER_SKILLS.find((item) => item.id === selectedSkill) ?? null;
  const latestAssistantSpeech = [...messages]
    .reverse()
    .find((message) => message.role === "assistant")?.content
    ?? skillGreeting
    ?? "你好，我是 H-IDE。";
  const activeMissionBlocksDraft = Boolean(
    mission
    && !["completed", "failed", "cancelled"].includes(mission.snapshot.status),
  );
  const renderMissionDraft = () => missionDraft ? <MissionPlanCard
      draft={missionDraft}
      robot={robot}
      busy={busy}
      feedback={feedback}
      setFeedback={setFeedback}
      onConfirm={onConfirmMission}
      onReject={onRejectMission}
      developerMode={developerMode}
      blockedByMissionId={activeMissionBlocksDraft ? mission?.missionId ?? null : null}
    /> : null;
  const renderMission = () => mission ? <MissionExecutionCard
      mission={mission}
      busy={busy}
      onContinue={onContinueMission}
      onCancel={onCancelMission}
      onApproveRevision={onApproveMissionRevision}
      onSelectCandidate={onSelectMissionCandidate}
      developerMode={developerMode}
    /> : null;
  const renderLegacyArtifacts = () => !missionDraft && !mission ? <>
    {draft && isVisualMissionPlan(draft.plan) ? <VisualMissionCard
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
    /> : draft ? <PlanDraftCard
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
      developerMode={developerMode}
    /> : null}
    {draft?.status === "confirmed" && lastReceipt && !isVisualMissionPlan(draft.plan) ? <ExecutionCard receipt={lastReceipt} /> : null}
  </> : null;
  const renderLeaderRobotPicker = () => (
    <div className="leader-skill-robot-selector">
      <span className="leader-skill-robot-label">选择机器人</span>
      <RobotPicker
        robots={robots}
        selectedRobotId={selectedRobotId}
        selectedDisplayId={selectedDisplayId}
        onRobotChange={onRobotChange}
        onDisplayChange={onDisplayRobotChange}
        compact={developerMode}
      />
    </div>
  );
  return <main className="chat-workspace">
    <aside className="leader-avatar-rail" aria-label="H-IDE 机器人助手">
      <RobotAvatar compact speech={selectedSkill ? latestAssistantSpeech : undefined} />
    </aside>
    <div className="chat-timeline" ref={timelineRef} aria-live="polite">
      <div className="chat-thread">
        {empty ? <section className="chat-empty-state">
          <h1>今天想和我做什么？</h1>
          <span className="sr-only">今天想和机器人做什么？</span>
          <p>先选择一个 Skill，再用中文告诉我目标。巡检会进入办公室并汇总信息；语音助手会直接用声音回应。</p>
          {renderLeaderRobotPicker()}
          <SkillSelection selected={selectedSkill} onSelect={onSelectSkill} />
          <div className="suggestion-grid">{suggestions.map((prompt) => <button key={prompt} type="button" onClick={() => onPrompt(prompt)}>{prompt}<Icon name="chevron" /></button>)}</div>
        </section> : null}
        {activeSkill ? <div className="leader-session-context">{empty ? null : renderLeaderRobotPicker()}<SkillModeBanner skill={activeSkill} onClear={onClearSkill} />{skillGreeting ? <article className="leader-skill-greeting"><div className="message-author"><span className="agent-avatar"><Icon name="spark" /></span>H-IDE</div><div className="message-body">{skillGreeting}</div></article> : null}</div> : null}
        {timeline.map((item) => <section
          className="conversation-turn"
          data-task-id={item.taskId ?? undefined}
          data-turn-sequence={item.turnSequence ?? undefined}
          key={item.key}
        >
          {item.messages.map((message) => <article className={`chat-message message-${message.role}`} key={message.messageId}>
            <div className="message-author">{message.role === "user" ? "你" : message.role === "assistant" ? <><span className="agent-avatar"><Icon name="spark" /></span>H-IDE</> : "系统"}<time>{formatTime(message.createdAt)}</time></div>
            <div className="message-body">{message.content}</div>
            {message.modality === "voice" ? <span className="message-modality"><Icon name="mic" />语音转写</span> : null}
          </article>)}
          {(activityTaskId !== null
            ? item.taskId === activityTaskId
            : item.key === "turn:legacy-orphans") && !thinking && inlineError
            ? <AssistantInlineFailure message={inlineError} onRetry={onRetry} />
            : null}
          {item.turn?.result?.assistantMessage && !item.messages.some((message) => message.role === "assistant") ? <article className="chat-message message-assistant"><div className="message-author"><span className="agent-avatar"><Icon name="spark" /></span>H-IDE</div><div className="message-body">{item.turn.result.assistantMessage}</div></article> : null}
          {item.turn && Boolean(item.turn.result?.evidence?.length || item.turn.result?.visualResult) ? <ConversationVisualEvidence turn={item.turn} developerMode={developerMode} /> : null}
          {missionTaskId !== null && item.taskId === missionTaskId ? renderMission() : null}
          {missionDraftTaskId !== null && item.taskId === missionDraftTaskId ? renderMissionDraft() : null}
          {legacyTaskId !== null && item.taskId === legacyTaskId ? renderLegacyArtifacts() : null}
        </section>)}
        {thinking ? <section className="conversation-turn transient-thinking-turn"><ThinkingIndicator canCancel={canCancelThinking} onCancel={onCancelPlanning} /></section> : null}
        {!missionAttached ? renderMission() : null}
        {!missionDraftAttached ? renderMissionDraft() : null}
        {!legacyAttached ? renderLegacyArtifacts() : null}
      </div>
    </div>
    <div className="composer-region">
      <form className="chat-composer" onSubmit={onSubmit}>
        <div className="composer-contexts">{developerMode ? <span className={`composer-health ${robot?.connected ? "connected" : "offline"}`}>{robot?.connected ? "代理在线" : "代理离线"}</span> : null}<button className={voiceReplyEnabled ? "attached" : ""} type="button" onClick={onToggleVoiceReplies} aria-label={voiceReplyEnabled ? "关闭语音回复" : "开启语音回复"}><Icon name="volume" />{voiceReplyEnabled ? "语音回复开" : "语音回复关"}</button>{developerMode ? <span>{voiceCaptureMode === "streaming" ? "流式 ASR" : "离线 WAV 降级"}</span> : null}</div>
        <textarea id="assistant-task" aria-label="任务描述" value={text} onChange={(event) => setText(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} rows={1} maxLength={4000} placeholder={connection === "live" ? "描述任务，或点击麦克风说话…" : "工作站离线，任务草稿不会发送"} disabled={connection !== "live"} />
        {voiceUi.partial ? <div className="voice-live-transcript" role="status" aria-live="polite"><span className="voice-wave" /> <b>{voiceUi.phase === "finalizing" ? "已识别" : "正在听"}</b><span>{voiceUi.partial}</span></div> : null}
        <div className="composer-actions"><div><button className={`composer-icon-button voice-button ${recording ? "recording" : ""}`} type="button" onClick={onVoice} disabled={voiceBusy || busy || connection !== "live"} aria-label={recording ? "停止录音并发送" : "开始语音输入"}><Icon name="mic" /></button><span className="composer-hint">{recording ? (voiceCaptureMode === "streaming" ? "流式聆听中，再次点击结束" : "离线录音中，再次点击批量转写") : voiceBusy ? "正在识别并发送…" : "Enter 发送 · Shift+Enter 换行"}</span></div><button className="send-button" type="submit" disabled={busy || !text.trim() || connection !== "live"} aria-label="发送给 H-IDE"><Icon name="send" /></button></div>
        {developerMode ? <details className="speech-metrics"><summary>语音链路指标</summary><span>状态 {voiceUi.phase}</span><span>首个 partial {voiceUi.metrics.firstPartialMs === undefined ? "—" : `${Math.round(voiceUi.metrics.firstPartialMs)} ms`}</span><span>停说→final {voiceUi.metrics.endToFinalMs === undefined ? "—" : `${Math.round(voiceUi.metrics.endToFinalMs)} ms`}</span><span>TTS 首包 {voiceUi.metrics.ttsFirstByteMs === undefined ? "—" : `${Math.round(voiceUi.metrics.ttsFirstByteMs)} ms`}</span><span>{voiceUi.metrics.framesSent} 帧 · {voiceUi.metrics.audioMsSent} ms 音频</span></details> : null}
      </form>
      <p className="composer-disclaimer">聊天和问答不会控制机器人；行动计划会在获得相应授权后执行。</p>
    </div>
  </main>;
}

function MissionPlanCard({ draft, robot, busy, feedback, setFeedback, onConfirm, onReject, developerMode, blockedByMissionId = null }: {
  draft: MissionDraftRecordV2;
  robot: RobotState | null;
  busy: boolean;
  feedback: string;
  setFeedback: (value: string) => void;
  onConfirm: () => void;
  onReject: () => void;
  developerMode: boolean;
  blockedByMissionId?: string | null;
}) {
  const plan = draft.plan;
  const reviewable = draft.status === "draft"
    && !busy
    && blockedByMissionId === null
    && plan.robotId === robot?.robotId;
  const typedPlaces = plan.nodes.flatMap((node) => {
    const intent = node.intent && typeof node.intent === "object"
      ? node.intent as Record<string, unknown>
      : null;
    const place = intent?.place && typeof intent.place === "object"
      ? intent.place as Record<string, unknown>
      : null;
    const completion = intent?.completion && typeof intent.completion === "object"
      ? intent.completion as Record<string, unknown>
      : null;
    return place && typeof place.locationId === "string"
      ? [{ node, place, completion }]
      : [];
  });
  const finalPlace = [...typedPlaces].reverse().find(({ place }) => (
    place.anchorRole === "interior" || place.anchorRole === "observation_vantage"
  )) ?? typedPlaces.at(-1);
  const finalPlaceName = typeof finalPlace?.place.locationName === "string"
    ? finalPlace.place.locationName
    : null;
  const finalAnchorRole = finalPlace?.place.anchorRole === "observation_vantage"
    ? "内部观察点"
    : finalPlace?.place.anchorRole === "interior"
      ? "内部站立点"
      : finalPlace?.place.anchorRole === "entry"
        ? "入口点"
        : null;
  const candidateCount = Array.isArray(finalPlace?.place.candidateTargetRecordIds)
    ? finalPlace.place.candidateTargetRecordIds.length
    : 0;
  const completionLabel = finalPlace?.completion?.predicate === "inside_region"
    ? "机器人进入目标区域并靠近选定内部点"
    : finalPlace?.completion?.predicate === "at_anchor"
      ? "机器人到达选定入口点"
      : null;
  return <section className="plan-draft-card mission-v2-card" aria-label="Mission V2 计划审核">
    <header>
      <div><span className="card-kicker"><Icon name="spark" />MISSION V2 · SUPERVISOR–WORKER</span><h2>{plan.objective}</h2></div>
      <StatusPill value={draft.status} />
    </header>
    {finalPlaceName ? <div className="place-plan-summary" aria-label="地点任务完成条件">
      <span><small>目标地点</small><strong>{finalPlaceName}</strong></span>
      <span><small>目标角色</small><strong>{finalAnchorRole ?? "已审核地点"}</strong></span>
      <span><small>备用点</small><strong>{Math.max(0, candidateCount - 1)} 个</strong></span>
      <span><small>完成条件</small><strong>{completionLabel ?? "完成全部审核步骤"}</strong></span>
    </div> : null}
    {developerMode ? <div className="plan-context-grid">
      <span><small>目标机器人</small><strong>{plan.robotId}</strong></span>
      <span><small>风险上限</small><strong>{plan.envelope.maximumRiskLevel}</strong></span>
      <span><small>单段 / 累计距离</small><strong>{plan.envelope.maximumGoalDistanceM}m / {plan.envelope.maximumCumulativeDistanceM}m</strong></span>
      <span><small>审核有效期</small><strong>{formatDraftExpiry(plan.expiresAt, Date.now())}</strong></span>
    </div> : <p className="user-plan-summary">我会按下面的顺序完成任务。执行中如果环境发生变化，我会停下并告诉你。</p>}
    <p className="card-warning"><Icon name="warning" />请确认周围环境安全，并留意机器人移动。</p>
    {blockedByMissionId ? <p className="mission-queue-notice"><Icon name="activity" />当前机器人仍在执行上一项 Mission；这份计划已排在对应对话轮次下方，待上一任务结束后即可审核。</p> : null}
    <ol className="plan-step-list">
      {plan.nodes.map((node, index) => <li key={node.nodeId}>
        <span>{index + 1}</span>
        <div><strong>{developerMode ? `${node.skillId}@${node.skillVersion}` : node.purpose}</strong>{developerMode ? <><p>{node.purpose}</p><small>{node.dependsOn.length ? `依赖：${node.dependsOn.join("、")}` : "起始节点"}</small></> : <small>{index === 0 ? "开始后先完成这一步" : "完成上一步后继续"}</small>}</div>
        <StatusPill value={node.riskLevel} />
      </li>)}
    </ol>
    {developerMode ? <details className="plan-technical-details">
      <summary><span>本次授权边界</span><Icon name="chevron" /></summary>
      <div>
        <section><strong>能力快照</strong><p>{plan.envelope.allowedSkillIds.join("、")}</p></section>
        <section><strong>固定身份</strong><p>boot: {plan.envelope.robotBootId}<br />epoch: {plan.envelope.localizationEpochId}</p></section>
        <section><strong>自动修订</strong><p>最多 {plan.envelope.maximumAutoRevisions} 次；超出授权边界必须重新审核。</p></section>
        <section><strong>场景先验</strong><p>{plan.envelope.sceneRefs.length ? plan.envelope.sceneRefs.map((item) => `${item.sceneId}@${item.revision}`).join("、") : "未附加场景先验"}</p></section>
        {finalPlace ? <section><strong>强类型地点绑定</strong><p>{String(finalPlace.place.locationId)} · {String(finalPlace.place.anchorRole)}<br />{String(finalPlace.place.targetRecordId)}</p></section> : null}
      </div>
    </details> : null}
    {draft.status === "draft" ? <div className="plan-review-controls">
      <label htmlFor="mission-review-feedback">审核备注（拒绝时会保存原因）</label>
      <textarea id="mission-review-feedback" value={feedback} onChange={(event) => setFeedback(event.target.value)} placeholder="例如：目标描述不清楚，先补充视觉参照物" />
      <div className="plan-review-actions"><button className="reject-plan" type="button" onClick={onReject} disabled={!reviewable}>拒绝</button><HoldConfirmButton disabled={!reviewable} onConfirm={onConfirm} /></div>
    </div> : <p className="frozen-plan-note"><Icon name="check" />该草案已{statusLabel(draft.status)}，没有发送未审核动作。</p>}
  </section>;
}

function MissionExecutionCard({ mission, busy, onContinue, onCancel, onApproveRevision, onSelectCandidate, developerMode }: {
  mission: MissionRecordV2;
  busy: boolean;
  onContinue: () => void;
  onCancel: () => void;
  onApproveRevision: () => void;
  onSelectCandidate: (candidateId: string, analysisId: string, label: string) => void;
  developerMode: boolean;
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
  const currentVisualAmbiguity = currentMissionVisualAmbiguity(mission);
  const officeReport = currentMissionOfficeReport(mission);
  const userProgress = missionUserProgress(mission);
  const presentedVisualResult = status === "needs_input"
    ? currentVisualAmbiguity ?? visualResult
    : visualResult;
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
    {developerMode ? <div className="mission-runtime-summary">
      <span><small>当前节点</small><strong>{mission.snapshot.activeNodeId ?? "—"}</strong></span>
      <span><small>活动修订</small><strong>r{mission.activeRevision.sequence}</strong></span>
      <span><small>视觉假设</small><strong>{hypotheses}</strong></span>
      <span><small>可信空间事实</small><strong>{facts}</strong></span>
    </div> : <p className="user-mission-status">{status === "completed" ? "任务已经完成。" : status === "failed" ? "任务没有完成，我已停止执行。" : status === "cancelled" ? "任务已取消。" : status === "needs_input" ? "任务已停止推进，需要补充目标信息。" : status === "suspended" ? "任务已暂停，场景、定位或授权状态需要重新确认。" : status === "awaiting_review" ? "任务修订正在等待你的审核。" : status === "ready" ? "下一节点的目标已经解析，正在继续执行。" : "机器人正在按计划执行，我会持续汇报进展。"}</p>}
    {userProgress.length ? <section className="mission-user-progress" aria-label="机器人执行进度">
      <strong>实时进度</strong>
      <ol>{userProgress.map((item) => <li className={`progress-${item.state}`} key={item.sequence}>
        <span />
        <p>{item.text}</p>
      </li>)}</ol>
    </section> : null}
    {presentedVisualResult ? <MissionVisualSummary
      result={presentedVisualResult}
      observationOnly={observationOnly}
      onOpen={() => setVisualEvidenceOpen(true)}
    /> : null}
    {officeReport ? <MissionOfficeReportCard report={officeReport} /> : null}
    {mission.snapshot.detail ? <p className={`mission-runtime-notice notice-${status}`}><Icon name={status === "needs_input" ? "warning" : "activity"} />{mission.snapshot.detail}</p> : null}
    <ol className="plan-step-list mission-node-list">
      {mission.activeRevision.nodes.map((node, index) => {
        const state = states.get(node.nodeId);
        const nodeStatus = state?.status ?? "pending";
        return <li key={node.nodeId} className={`node-${nodeStatus}`}>
          <span>{nodeStatus === "succeeded" ? "✓" : nodeStatus === "skipped" ? "–" : index + 1}</span>
          <div><strong>{developerMode ? node.skillId : node.purpose}</strong>{developerMode ? <p>{node.purpose}</p> : null}<small>{developerMode ? `尝试 ${state?.attempt ?? 0} 次 · ` : ""}{statusLabel(nodeStatus)}</small></div>
          <StatusPill value={nodeStatus} />
        </li>;
      })}
    </ol>
    {developerMode && latestEvents.length ? <details className="plan-technical-details mission-event-details">
      <summary><span>事件时间线（{mission.events.length}）</span><Icon name="chevron" /></summary>
      <div>{latestEvents.map((event) => <section key={event.eventId}><strong>#{event.sequence} · {event.eventType}</strong><p>{event.nodeId ?? "Mission"} · {formatTime(event.occurredAt)}</p></section>)}</div>
    </details> : null}
    {currentVisualAmbiguity ? <MissionVisualCandidateSelection
      result={currentVisualAmbiguity}
      busy={busy}
      onSelect={onSelectCandidate}
      missionId={mission.missionId}
      onLoadEvidence={loadMissionVisualEvidence}
    /> : null}
    {status === "needs_input" ? <p className="mission-input-hint">{currentVisualAmbiguity ? "可以在下方用文字或语音选择一个视觉候选；系统会先重新观察验证，不会直接移动。" : "请根据上方的具体阻塞原因补充地点信息，或取消后重新规划；系统不会退化到入口或任意可信点。"}</p> : null}
    {status === "awaiting_review" ? <div className="mission-revision-review"><p><strong>修订需要审核</strong><span>{Array.isArray(pendingRevision?.changeSummary) ? pendingRevision.changeSummary.join("；") : "持续规划改变了任务步骤或授权范围。"}</span></p><button type="button" onClick={onApproveRevision} disabled={busy}>审核修订并继续</button></div> : null}
    {!['completed', 'failed', 'cancelled'].includes(status) ? <div className="mission-runtime-actions">
      <button className="mission-cancel-button" type="button" onClick={onCancel} disabled={busy}>取消 Mission</button>
      {status === "ready" ? <button type="button" onClick={onContinue} disabled={busy}>继续执行就绪节点</button> : null}
    </div> : null}
    {status === "completed" ? <p className="frozen-plan-note"><Icon name="check" />Mission 已完成；每个目标与执行回执均保留在事件记录中。</p> : null}
  </section>
  {visualEvidenceOpen && presentedVisualResult ? <MissionVisualEvidenceDrawer
    missionId={mission.missionId}
    result={presentedVisualResult}
    observationOnly={observationOnly}
    onClose={() => setVisualEvidenceOpen(false)}
    onLoadEvidence={loadMissionVisualEvidence}
  /> : null}
  </>;
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

function PlanDraftCard({ draft, robot, activeSceneContext, busy, feedback, setFeedback, onConfirm, onReject, onRevise, onReplan, developerMode }: {
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
  developerMode: boolean;
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
    {developerMode ? <div className="plan-context-grid"><span><small>目标机器人</small><strong>{draft.plan.robotId}</strong></span><span><small>场景版本</small><strong>{activeSceneContext ? `r${activeSceneContext.revision}` : "未绑定"}</strong></span><span><small>起点位姿</small><strong>{robot?.pose ? `${robot.pose.x.toFixed(2)}, ${robot.pose.y.toFixed(2)}` : "未知"}</strong></span><span><small>审核有效期</small><strong className={expired ? "expired" : ""}>{expiry}</strong></span></div> : <p className="user-plan-summary">请确认这项机器人行动是否符合你的意思。</p>}
    {draft.plan.robotId !== robot?.robotId ? <p className="card-warning"><Icon name="warning" />当前机器人与草案不一致，不能执行。</p> : null}
    {review?.warnings?.length ? <div className="plan-warnings"><strong><Icon name="warning" />执行前确认</strong>{review.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div> : null}
    <ol className="plan-step-list">{draft.plan.steps.map((step, index) => {
      const detail = review?.steps.find((item) => item.stepId === step.stepId);
      const goal = navigationGoal(step.payload);
      return <li key={step.stepId}><span>{index + 1}</span><div><strong>{developerMode ? step.commandKey : (detail?.rationale ?? `完成第 ${index + 1} 步`)}</strong>{developerMode && detail ? <p>{detail.rationale}</p> : null}{developerMode && goal ? <small>目标坐标 · {goal.x.toFixed(2)}, {goal.y.toFixed(2)}, {goal.z.toFixed(2)} · yaw {goal.yaw.toFixed(2)}</small> : null}{developerMode && step.dependsOn?.length ? <small>等待步骤：{step.dependsOn.join("、")}</small> : null}</div><StatusPill value="pending" /></li>;
    })}</ol>
    {developerMode && (review?.contributions?.length || review?.spatialChecks?.length) ? <details className="plan-technical-details"><summary>查看 Agent 贡献与空间依据 <Icon name="chevron" /></summary><div>{review?.contributions?.map((worker) => <section key={worker.agentId}><strong>{worker.agentId}</strong><p>{worker.summary}</p></section>)}{review?.spatialChecks?.map((check) => <section key={check.stepId}><strong>{check.stepId} · {({ observed_free: "未发现障碍", occupied: "障碍警告", unknown: "观测不足", outside: "地图外" } as const)[check.check.classification]}</strong><p>map {check.goalPose.x.toFixed(2)}, {check.goalPose.y.toFixed(2)} · 净空 {check.check.clearanceMeters.toFixed(2)} m</p></section>)}</div></details> : null}
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
              aria-label={scenePanelOpen ? "收起归档场景" : "打开归档场景"}
            >
              <span className="input-mode-icon">◫</span><strong>归档场景</strong><small>产品界面暂不提供</small>
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
              aria-label={recording ? "停止流式语音" : "开始语音输入"}
            >
              <span className="input-mode-icon">◉</span><strong>{recording ? "停止流式语音" : "语音助手"}</strong><small>{voiceBusy ? "正在发送…" : "AudioWorklet · 流式 ASR"}</small>
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

export {
  App,
  apiFetch,
  ApiFailure,
  isDraftExpired,
  isDraftReviewableForRobot,
  preferredDemoRobotId,
  preferredRobotId,
};

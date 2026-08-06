import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type {
  ApiError,
  CapabilitySnapshot,
  CommandReceipt,
  ControlEvent,
  MappingSnapshot,
  PlanDraft,
  RobotState,
} from "./contracts";
import { isTerminal, statusLabel } from "./contracts";
import "./app.css";

type View = "workspace" | "assistant" | "interfaces";
type ConnectionState = "connecting" | "live" | "offline";

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

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("accept", "application/json");
  if (init.body && !headers.has("content-type")) headers.set("content-type", "application/json");
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
    const detail = typeof body === "object" && body !== null && "detail" in body
      ? String((body as ApiError).detail ?? response.statusText)
      : typeof body === "string" ? body : response.statusText;
    throw new ApiFailure(detail || `请求失败 (${response.status})`, response.status);
  }
  return body as T;
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
  if (error instanceof ApiFailure) return error.message;
  if (error instanceof Error) return error.message;
  return "请求失败，请检查工作站连接";
}

function StatusPill({ value, tone }: { value: string | undefined; tone?: string }) {
  const normalized = (value ?? "unknown").toLowerCase().replace(/[^a-z0-9-]/g, "-");
  return <span className={`status-pill ${tone ?? normalized}`}>{statusLabel(value)}</span>;
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="empty-state">{children}</div>;
}

function App() {
  const [view, setView] = useState<View>("workspace");
  const [robots, setRobots] = useState<RobotState[]>([]);
  const [selectedRobotId, setSelectedRobotId] = useState("robot-1");
  const [capabilities, setCapabilities] = useState<CapabilitySnapshot | null>(null);
  const [events, setEvents] = useState<ControlEvent[]>([]);
  const [mappings, setMappings] = useState<MappingSnapshot[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [lastReceipt, setLastReceipt] = useState<CommandReceipt | null>(null);
  const [lastCommandId, setLastCommandId] = useState<string | null>(null);
  const [planJson, setPlanJson] = useState(
    JSON.stringify({
      robotId: "robot-1",
      steps: [{ stepId: "step-1", commandKey: "move.forward", payload: {} }],
    }, null, 2),
  );
  const [draft, setDraft] = useState<{ draftId: string; status: string; plan: PlanDraft } | null>(null);
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

  const selectedRobot = useMemo(
    () => robots.find((robot) => robot.robotId === selectedRobotId) ?? null,
    [robots, selectedRobotId],
  );

  const loadRobots = useCallback(async () => {
    const list = await apiFetch<RobotState[]>("/api/v1/robots");
    setRobots(list);
    if (list.length && !list.some((robot) => robot.robotId === selectedRobotId)) {
      setSelectedRobotId(list[0].robotId);
    }
  }, [selectedRobotId]);

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

  useEffect(() => {
    let cancelled = false;
    void Promise.all([loadRobots(), loadMappings(), apiFetch<ControlEvent[]>("/api/v1/events")])
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
  }, [loadMappings, loadRobots]);

  useEffect(() => {
    void loadCapabilities(selectedRobotId);
  }, [loadCapabilities, selectedRobotId]);

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

  const createDraft = useCallback((event: FormEvent) => {
    event.preventDefault();
    void runRequest(async () => {
      let plan: PlanDraft;
      try {
        plan = JSON.parse(planJson) as PlanDraft;
      } catch {
        throw new Error("计划 JSON 格式不正确");
      }
      const result = await apiFetch<{ draftId: string; status: string; plan: PlanDraft }>("/api/v1/assistant/drafts", {
        method: "POST",
        body: JSON.stringify({ ...plan, draftId: plan.draftId ?? plan.planId ?? makeId("draft") }),
      });
      setDraft(result);
      setNotice("计划已生成，等待人工确认");
    });
  }, [planJson, runRequest]);

  const confirmDraft = useCallback(() => {
    if (!draft) return;
    void runRequest(async () => {
      const result = await apiFetch<{ draftId: string; status: string; receipts?: CommandReceipt[] }>(`/api/v1/assistant/drafts/${encodeURIComponent(draft.draftId)}/confirm`, {
        method: "POST",
        body: JSON.stringify({ confirmed: true }),
      });
      setDraft((current) => current ? { ...current, status: result.status } : current);
      if (result.receipts?.length) {
        const receipt = result.receipts[result.receipts.length - 1];
        setLastReceipt(receipt);
        setLastCommandId(receipt.commandId);
      }
      setNotice(`计划 ${statusLabel(result.status)}`);
    });
  }, [draft, runRequest]);

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

  const latestEvents = [...events].reverse().slice(0, 12);
  const motionDisabled = busy || !selectedRobot || selectedRobot.safetyMode === "estop" || selectedRobot.status === "offline";

  return (
    <div className="console-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">◈</span>
          <div>
            <p className="eyebrow">WORKSTATION CONTROL PLANE</p>
            <h1>Robot Action Console</h1>
          </div>
        </div>
        <div className="topbar-actions">
          <span className="connection-indicator" data-state={connection}>
            <span className="connection-dot" /> {connection === "live" ? "实时连接" : connection === "connecting" ? "连接中" : "离线缓存"}
          </span>
          <button className="button button-ghost" type="button" onClick={refresh} disabled={busy}>刷新状态</button>
        </div>
      </header>

      <main className="main-content">
        <section className="fleet-strip" aria-label="机器人 fleet">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">FLEET</p>
              <h2>机器人编队</h2>
            </div>
            <span className="muted">{robots.length} 台已登记</span>
          </div>
          <div className="robot-cards">
            {robots.length === 0 ? <EmptyState>正在等待工作站返回机器人列表…</EmptyState> : robots.map((robot) => (
              <button
                className={`robot-card ${robot.robotId === selectedRobotId ? "selected" : ""}`}
                key={robot.robotId}
                type="button"
                onClick={() => setSelectedRobotId(robot.robotId)}
              >
                <span className="robot-card-top"><span className="robot-avatar">{robot.robotId.slice(-2).toUpperCase()}</span><StatusPill value={robot.status} /></span>
                <strong>{robot.name ?? robot.robotId}</strong>
                <span className="robot-meta">{robot.connected ? "链路正常" : "等待重连"} · {statusLabel(robot.safetyMode)}</span>
                <span className="battery-line"><span style={{ width: `${Math.max(0, Math.min(100, robot.battery ?? 0))}%` }} />{robot.battery == null ? "—" : `${robot.battery}%`}</span>
              </button>
            ))}
          </div>
        </section>

        <div className="workspace-grid">
          <aside className="sidebar panel">
            <p className="eyebrow">ROBOT WORKSPACE</p>
            <nav className="view-tabs" aria-label="工作区导航">
              <button className={view === "workspace" ? "active" : ""} type="button" role="tab" aria-selected={view === "workspace"} onClick={() => setView("workspace")}>控制工作台</button>
              <button className={view === "assistant" ? "active" : ""} type="button" role="tab" aria-selected={view === "assistant"} onClick={() => setView("assistant")}>Agent 计划</button>
              <button className={view === "interfaces" ? "active" : ""} type="button" role="tab" aria-selected={view === "interfaces"} onClick={() => setView("interfaces")}>接口中心</button>
            </nav>
            <div className="safety-card">
              <div className="safety-card-title"><span className="safety-icon">!</span><strong>安全状态</strong></div>
              <StatusPill value={selectedRobot?.safetyMode ?? "unknown"} tone={selectedRobot?.safetyMode === "normal" ? "normal" : "warning"} />
              <p>物理急停与机器人本地安全监控始终优先于远程链路。</p>
            </div>
            <div className="sidebar-note"><span className="lock-icon">⌁</span><span>浏览器只发送高层命令<br />不暴露 ROS 话题或速度接口</span></div>
          </aside>

          <section className="content-column">
            {error && <div className="alert alert-error" role="alert"><strong>连接/操作异常</strong><span>{error}</span><button type="button" onClick={() => setError("")} aria-label="关闭错误">×</button></div>}
            {notice && <div className="alert alert-success" role="status"><span>{notice}</span><button type="button" onClick={() => setNotice("")} aria-label="关闭提示">×</button></div>}
            {view === "workspace" && (
              <WorkspaceView
                robot={selectedRobot}
                capabilities={capabilities}
                lastReceipt={lastReceipt}
                lastCommandId={lastCommandId}
                disabled={motionDisabled}
                busy={busy}
                onCommand={submitCommand}
                onCancel={cancelLast}
              />
            )}
            {view === "assistant" && (
              <AssistantView
                planJson={planJson}
                setPlanJson={setPlanJson}
                draft={draft}
                busy={busy}
                onCreate={createDraft}
                onConfirm={confirmDraft}
              />
            )}
            {view === "interfaces" && (
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
            )}
          </section>

          <aside className="timeline-column panel">
            <div className="section-heading compact"><div><p className="eyebrow">EVENT STREAM</p><h2>执行时间线</h2></div><span className="live-badge">{connection === "live" ? "LIVE" : "CACHE"}</span></div>
            <div className="timeline" aria-live="polite">
              {latestEvents.length === 0 ? <EmptyState>暂无事件。手动命令和 Agent 计划的每个状态都会记录在这里。</EmptyState> : latestEvents.map((event, index) => <TimelineItem event={event} key={`${event.eventId ?? "local"}-${index}`} />)}
            </div>
          </aside>
        </div>
      </main>
      <footer className="footer-note"><span>控制面 v0.1 · SQLite local profile</span><span>Stop / Cancel 保持在控制链路内，不依赖场景、视频或助手服务</span></footer>
    </div>
  );
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
        <ActionButton label="立即停止" hint="stop · safety" icon="■" danger disabled={busy || !robot} onClick={() => onCommand("stop", {})} />
        <button className="action-button secondary-action" type="button" disabled={!lastCommandId || busy} onClick={onCancel}><span className="action-icon">×</span><span><strong>取消最近命令</strong><small>{lastCommandId ?? "暂无可取消命令"}</small></span></button>
      </div>
      {(!robot || robot.status === "offline") && <p className="inline-warning">机器人不在线，运动动作已锁定；停止与状态恢复仍可用。</p>}
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

function AssistantView({ planJson, setPlanJson, draft, busy, onCreate, onConfirm }: { planJson: string; setPlanJson: (value: string) => void; draft: { draftId: string; status: string; plan: PlanDraft } | null; busy: boolean; onCreate: (event: FormEvent) => void; onConfirm: () => void }) {
  return <section className="panel assistant-panel"><div className="section-heading"><div><p className="eyebrow">AGENT GATE</p><h2>Agent 计划确认</h2></div><span className="muted">规划与执行分离</span></div><div className="assistant-layout"><form className="stack-form" onSubmit={onCreate}><label htmlFor="plan-json">计划 JSON</label><textarea id="plan-json" value={planJson} onChange={(event) => setPlanJson(event.target.value)} rows={13} spellCheck={false} /><div className="form-help">Agent 只能生成结构化草案；创建草案不会调用机器人。确认前可编辑、审阅和撤回。</div><button className="button button-primary" type="submit" disabled={busy}>生成待确认草案</button></form><div className="draft-review"><p className="eyebrow">CONFIRMATION GATE</p>{!draft ? <><EmptyState>还没有待确认计划。提交左侧草案后，在这里检查步骤和目标机器人。</EmptyState><button className="button button-confirm" type="button" disabled>确认并执行计划</button></> : <><div className="draft-header"><div><span className="muted">待确认：{draft.draftId}</span><h3>{draft.plan.steps.length} 个高层步骤</h3></div><StatusPill value={draft.status} /></div><div className="plan-target">目标机器人 <strong>{draft.plan.robotId}</strong></div><ol className="plan-steps">{draft.plan.steps.map((step) => <li key={step.stepId}><span>{step.stepId}</span><strong>{step.commandKey}</strong>{step.dependsOn?.length ? <small>依赖：{step.dependsOn.join(", ")}</small> : null}</li>)}</ol><button className="button button-confirm" type="button" onClick={onConfirm} disabled={busy || draft.status !== "draft"}>确认并执行计划</button><p className="form-help">仅点击确认后才会进入 ActionRuntime；accepted/running 的不确定结果会暂停计划，禁止自动重放。</p></>}</div></div></section>;
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

function formatTime(value?: string): string {
  if (!value) return "刚刚";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

if (typeof document !== "undefined") {
  const root = document.getElementById("root");
  if (root && !root.hasAttribute("data-test-root")) createRoot(root).render(<App />);
}

export { App, apiFetch, ApiFailure };

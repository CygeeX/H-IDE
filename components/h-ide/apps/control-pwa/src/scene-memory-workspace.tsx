import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import type {
  AuthPrincipal,
  PoseMap,
  RobotState,
  SceneInspectionSession,
  SceneLandmark,
  SceneLandmarkAlias,
  SceneLandmarkStatus,
  SceneMemoryCapabilities,
} from "./contracts";
import "./scene-memory-workspace.css";

type JsonFetcher = <T>(path: string, init?: RequestInit) => Promise<T>;
type BlobFetcher = (path: string) => Promise<Blob>;

export interface SceneMemoryWorkspaceProps {
  principal: AuthPrincipal;
  robots: RobotState[];
  selectedRobotId: string;
  developerMode: boolean;
  apiFetch: JsonFetcher;
  apiFetchBlob: BlobFetcher;
  onBackToChat: () => void;
}

const STATUS_LABELS: Record<SceneLandmarkStatus, string> = {
  candidate: "待确认",
  confirmed: "已记住",
  superseded: "已作废",
};

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function roughLandmarkLocation(
  mapPose: PoseMap | null | undefined,
  robotPose: PoseMap | null | undefined,
): string {
  if (!mapPose || !robotPose || ![mapPose.x, mapPose.y, robotPose.x, robotPose.y, robotPose.yaw].every(finite)) {
    return "已保存大致位置";
  }
  const dx = mapPose.x - robotPose.x;
  const dy = mapPose.y - robotPose.y;
  const distance = Math.hypot(dx, dy);
  if (distance < 0.45) return "在本次记录位置附近";
  const forward = Math.cos(robotPose.yaw) * dx + Math.sin(robotPose.yaw) * dy;
  const left = -Math.sin(robotPose.yaw) * dx + Math.cos(robotPose.yaw) * dy;
  const parts: string[] = [];
  if (Math.abs(forward) > Math.max(0.35, Math.abs(left) * 0.35)) parts.push(forward >= 0 ? "前方" : "后方");
  if (Math.abs(left) > Math.max(0.35, Math.abs(forward) * 0.35)) parts.push(left >= 0 ? "左侧" : "右侧");
  const rounded = Math.max(0.5, Math.round(distance * 2) / 2);
  return `在本次记录位置${parts.join("偏") || "附近"}约 ${rounded.toFixed(1)} 米`;
}

function dateTime(value?: string | null): string {
  if (!value) return "时间未知";
  const parsed = new Date(value);
  if (!Number.isFinite(parsed.valueOf())) return value;
  return parsed.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function safeEvidencePath(value?: string | null): string {
  if (!value || !value.startsWith("/api/") || value.startsWith("//") || value.includes("..")) return "";
  return value;
}

function EvidenceThumbnail({
  reference,
  label,
  load,
}: {
  reference?: string | null;
  label: string;
  load: BlobFetcher;
}) {
  const path = safeEvidencePath(reference);
  const [source, setSource] = useState("");
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!path) {
      setSource("");
      setFailed(false);
      return undefined;
    }
    let disposed = false;
    let objectUrl = "";
    setSource("");
    setFailed(false);
    void load(path)
      .then((blob) => {
        if (disposed) return;
        if (typeof URL.createObjectURL !== "function") throw new Error("object URL unavailable");
        objectUrl = URL.createObjectURL(blob);
        setSource(objectUrl);
      })
      .catch(() => {
        if (!disposed) setFailed(true);
      });
    return () => {
      disposed = true;
      if (objectUrl && typeof URL.revokeObjectURL === "function") URL.revokeObjectURL(objectUrl);
    };
  }, [load, path]);

  if (source) return <img className="memory-evidence-image" src={source} alt={`${label}的视觉证据`} />;
  return <div className={`memory-evidence-placeholder ${failed ? "failed" : ""}`} aria-label={`${label}的视觉证据缩略图不可用`}>
    <span aria-hidden="true">◉</span>
    <small>{path && !failed ? "载入视觉凭证…" : failed ? "凭证暂时无法载入" : "暂无可访问缩略图"}</small>
  </div>;
}

function statusClass(status: SceneLandmarkStatus): string {
  return `memory-status memory-${status}`;
}

function DeveloperProvenance({ landmark }: { landmark: SceneLandmark }) {
  const observation = landmark.currentObservation;
  if (!observation) return <p className="memory-developer-empty">该记录没有可显示的来源详情。</p>;
  return <details className="memory-developer-provenance">
    <summary>开发者来源、坐标与定位纪元</summary>
    <dl>
      <div><dt>map 坐标</dt><dd>x {observation.mapPose.x.toFixed(3)} · y {observation.mapPose.y.toFixed(3)} · z {observation.mapPose.z.toFixed(3)} · yaw {observation.mapPose.yaw.toFixed(3)}</dd></div>
      <div><dt>定位纪元</dt><dd><code>{observation.localizationEpochId}</code></dd></div>
      <div><dt>视觉来源</dt><dd>{observation.vlmProvider} · {observation.vlmModel}</dd></div>
      <div><dt>证据 SHA-256</dt><dd><code>{observation.evidenceSha256}</code></dd></div>
      <div><dt>TF provenance</dt><dd><code>{JSON.stringify(observation.tfProvenance)}</code></dd></div>
      <div><dt>Depth provenance</dt><dd><code>{JSON.stringify(observation.depth)}</code></dd></div>
    </dl>
  </details>;
}

function LandmarkCard({
  landmark,
  developerMode,
  busy,
  apiFetchBlob,
  onConfirm,
  onRename,
  onAddAlias,
  onRemoveAlias,
  onInvalidate,
}: {
  landmark: SceneLandmark;
  developerMode: boolean;
  busy: boolean;
  apiFetchBlob: BlobFetcher;
  onConfirm: (landmark: SceneLandmark) => void;
  onRename: (landmark: SceneLandmark) => void;
  onAddAlias: (landmark: SceneLandmark) => void;
  onRemoveAlias: (landmark: SceneLandmark, alias: SceneLandmarkAlias) => void;
  onInvalidate: (landmark: SceneLandmark) => void;
}) {
  const observation = landmark.currentObservation;
  const activeAliases = landmark.aliases.filter((alias) => alias.status === "active");
  return <article className={`memory-landmark-card landmark-${landmark.status}`}>
    <EvidenceThumbnail
      reference={observation?.evidenceRef}
      label={landmark.canonicalName}
      load={apiFetchBlob}
    />
    <div className="memory-landmark-body">
      <header>
        <div><h3>{landmark.canonicalName}</h3><p>{landmark.description || roughLandmarkLocation(observation?.mapPose, observation?.robotPose)}</p></div>
        <span className={statusClass(landmark.status)}>{STATUS_LABELS[landmark.status]}</span>
      </header>
      {landmark.description ? <p className="memory-rough-location">{roughLandmarkLocation(observation?.mapPose, observation?.robotPose)}</p> : null}
      <div className="memory-landmark-meta">
        <time>{dateTime(observation?.createdAt ?? landmark.createdAt)}</time>
        {finite(observation?.confidence) ? <span>可信度 {Math.round(observation.confidence * 100)}%</span> : null}
        {observation?.supportCount ? <span>{observation.supportCount} 帧支持</span> : null}
      </div>
      {activeAliases.length ? <div className="memory-aliases" aria-label={`${landmark.canonicalName}的别名`}>
        {activeAliases.map((alias) => <span key={alias.aliasId}>{alias.alias}{landmark.status !== "superseded" ? <button type="button" disabled={busy} onClick={() => onRemoveAlias(landmark, alias)} aria-label={`移除别名 ${alias.alias}`}>×</button> : null}</span>)}
      </div> : null}
      {developerMode ? <DeveloperProvenance landmark={landmark} /> : null}
      {landmark.executionEligible === false ? <p className="memory-safety-boundary">这条记忆只辅助后续理解；不会直接驱动机器人，行动仍需 Mission 与实时视觉复核。</p> : null}
      {landmark.status !== "superseded" ? <footer>
        {landmark.status === "candidate" ? <button className="memory-primary" type="button" disabled={busy} onClick={() => onConfirm(landmark)}>确认记住</button> : null}
        <button type="button" disabled={busy} onClick={() => onRename(landmark)}>修改名称</button>
        <button type="button" disabled={busy} onClick={() => onAddAlias(landmark)}>添加别名</button>
        <button className="memory-danger" type="button" disabled={busy} onClick={() => onInvalidate(landmark)}>作废</button>
      </footer> : null}
    </div>
  </article>;
}

export function SceneMemoryWorkspace({
  principal,
  robots,
  selectedRobotId,
  developerMode,
  apiFetch,
  apiFetchBlob,
  onBackToChat,
}: SceneMemoryWorkspaceProps) {
  const canUse = principal.scopes.includes("scene-memory:use");
  const [capabilities, setCapabilities] = useState<SceneMemoryCapabilities | null>(null);
  const [inspections, setInspections] = useState<SceneInspectionSession[]>([]);
  const [selectedInspection, setSelectedInspection] = useState<SceneInspectionSession | null>(null);
  const [landmarks, setLandmarks] = useState<SceneLandmark[]>([]);
  const [robotId, setRobotId] = useState(selectedRobotId);
  const [purpose, setPurpose] = useState("办公室巡检");
  const [filter, setFilter] = useState<"all" | SceneLandmarkStatus>("all");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [unavailable, setUnavailable] = useState("");

  const selectedRobot = useMemo(
    () => robots.find((robot) => robot.robotId === robotId) ?? null,
    [robotId, robots],
  );
  const localizationEpochId = typeof selectedRobot?.details?.localizationEpochId === "string"
    ? selectedRobot.details.localizationEpochId
    : "";

  useEffect(() => {
    if (!robotId && selectedRobotId) setRobotId(selectedRobotId);
  }, [robotId, selectedRobotId]);

  const loadAll = useCallback(async (preferredInspectionId?: string) => {
    if (!canUse) {
      setUnavailable("当前账户没有场景记忆权限。请联系管理员授予 scene-memory:use。");
      setLoading(false);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const [capabilityValue, inspectionValues, landmarkValues] = await Promise.all([
        apiFetch<SceneMemoryCapabilities>("/api/v2/scene-memory/capabilities", { cache: "no-store" }),
        apiFetch<SceneInspectionSession[]>("/api/v2/scene-memory/inspections?includeClosed=true&limit=100", { cache: "no-store" }),
        apiFetch<SceneLandmark[]>("/api/v2/scene-memory/landmarks?includeSuperseded=true&limit=100", { cache: "no-store" }),
      ]);
      if (capabilityValue.semanticProvider !== "qwen" || capabilityValue.executionEligible !== false) {
        throw new Error("场景记忆服务没有声明纯 Qwen、只读观察安全边界");
      }
      setCapabilities(capabilityValue);
      setInspections(Array.isArray(inspectionValues) ? inspectionValues : []);
      setLandmarks(Array.isArray(landmarkValues) ? landmarkValues : []);
      setUnavailable("");
      const preferred = inspectionValues.find((item) => item.inspectionId === preferredInspectionId)
        ?? inspectionValues.find((item) => item.status === "open" && item.robotId === robotId)
        ?? inspectionValues[0];
      if (preferred) {
        const detail = await apiFetch<SceneInspectionSession>(`/api/v2/scene-memory/inspections/${encodeURIComponent(preferred.inspectionId)}`, { cache: "no-store" });
        setSelectedInspection(detail);
      } else {
        setSelectedInspection(null);
      }
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "场景记忆服务不可用";
      setCapabilities(null);
      setUnavailable(`巡检与记忆服务当前不可用：${message}`);
    } finally {
      setLoading(false);
    }
  }, [apiFetch, canUse, robotId]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  const run = useCallback(async (operation: () => Promise<void>, success: string) => {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await operation();
      setNotice(success);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "操作失败，请稍后重试");
    } finally {
      setBusy(false);
    }
  }, []);

  const startInspection = (event: FormEvent) => {
    event.preventDefault();
    if (!selectedRobot?.connected) {
      setError("所选机器人未连接，不能开始巡检。");
      return;
    }
    if (!localizationEpochId) {
      setError("Robot Agent 尚未提供 localizationEpochId；不能伪造定位纪元。");
      return;
    }
    void run(async () => {
      const created = await apiFetch<SceneInspectionSession>("/api/v2/scene-memory/inspections", {
        method: "POST",
        body: JSON.stringify({
          robotId: selectedRobot.robotId,
          localizationEpochId,
          mapFrameId: "map",
          purpose: purpose.trim() || undefined,
        }),
      });
      await loadAll(created.inspectionId);
    }, "巡检已创建。请回到对话，通过普通 Mission / Nav2 把机器人移动到要记录的位置。");
  };

  const chooseInspection = (inspection: SceneInspectionSession) => {
    setBusy(true);
    setError("");
    void apiFetch<SceneInspectionSession>(`/api/v2/scene-memory/inspections/${encodeURIComponent(inspection.inspectionId)}`, { cache: "no-store" })
      .then(setSelectedInspection)
      .catch((reason) => setError(reason instanceof Error ? reason.message : "无法读取巡检"))
      .finally(() => setBusy(false));
  };

  const confirmLandmark = (landmark: SceneLandmark) => {
    if (!window.confirm(`确认把“${landmark.canonicalName}”保存为场景记忆？它不会直接触发机器人运动。`)) return;
    void run(async () => {
      await apiFetch(`/api/v2/scene-memory/landmarks/${encodeURIComponent(landmark.landmarkId)}/confirm`, {
        method: "POST",
        body: JSON.stringify({ expectedRecordVersion: landmark.recordVersion }),
      });
      await loadAll(selectedInspection?.inspectionId);
    }, `已记住“${landmark.canonicalName}”`);
  };

  const renameLandmark = (landmark: SceneLandmark) => {
    const value = window.prompt("修改名称", landmark.canonicalName)?.trim();
    if (!value || value === landmark.canonicalName) return;
    if (!window.confirm(`把“${landmark.canonicalName}”更名为“${value}”？`)) return;
    void run(async () => {
      await apiFetch(`/api/v2/scene-memory/landmarks/${encodeURIComponent(landmark.landmarkId)}`, {
        method: "PATCH",
        body: JSON.stringify({ expectedRecordVersion: landmark.recordVersion, canonicalName: value }),
      });
      await loadAll(selectedInspection?.inspectionId);
    }, `名称已改为“${value}”`);
  };

  const addAlias = (landmark: SceneLandmark) => {
    const alias = window.prompt(`为“${landmark.canonicalName}”添加别名`)?.trim();
    if (!alias) return;
    if (!window.confirm(`确认添加别名“${alias}”？`)) return;
    void run(async () => {
      await apiFetch(`/api/v2/scene-memory/landmarks/${encodeURIComponent(landmark.landmarkId)}/aliases`, {
        method: "POST",
        body: JSON.stringify({ alias }),
      });
      await loadAll(selectedInspection?.inspectionId);
    }, `已添加别名“${alias}”`);
  };

  const removeAlias = (landmark: SceneLandmark, alias: SceneLandmarkAlias) => {
    if (!window.confirm(`确认移除别名“${alias.alias}”？历史审计仍会保留。`)) return;
    void run(async () => {
      await apiFetch(`/api/v2/scene-memory/landmarks/${encodeURIComponent(landmark.landmarkId)}/aliases/${encodeURIComponent(alias.aliasId)}`, { method: "DELETE" });
      await loadAll(selectedInspection?.inspectionId);
    }, `别名“${alias.alias}”已停用`);
  };

  const invalidateLandmark = (landmark: SceneLandmark) => {
    const reason = window.prompt(`为什么要作废“${landmark.canonicalName}”？`)?.trim();
    if (!reason) return;
    if (!window.confirm(`确认作废“${landmark.canonicalName}”？它会保留在历史记录中。`)) return;
    void run(async () => {
      await apiFetch(`/api/v2/scene-memory/landmarks/${encodeURIComponent(landmark.landmarkId)}/invalidate`, {
        method: "POST",
        body: JSON.stringify({ expectedRecordVersion: landmark.recordVersion, reason }),
      });
      await loadAll(selectedInspection?.inspectionId);
    }, `“${landmark.canonicalName}”已作废`);
  };

  const finishInspection = (cancel: boolean) => {
    if (!selectedInspection || selectedInspection.status !== "open") return;
    const action = cancel ? "取消" : "完成";
    if (!window.confirm(`${action}这次巡检？${cancel ? "未完成的采集点会取消，已确认记忆仍保留。" : "至少需要一条已确认记忆。"}`)) return;
    void run(async () => {
      await apiFetch(`/api/v2/scene-memory/inspections/${encodeURIComponent(selectedInspection.inspectionId)}/${cancel ? "cancel" : "complete"}`, {
        method: "POST",
        body: JSON.stringify({ expectedRecordVersion: selectedInspection.recordVersion, reason: cancel ? "用户从巡检界面取消" : undefined }),
      });
      await loadAll(selectedInspection.inspectionId);
    }, `巡检已${action}`);
  };

  const filteredLandmarks = landmarks.filter((landmark) => filter === "all" || landmark.status === filter);
  const counts = landmarks.reduce<Record<SceneLandmarkStatus, number>>((current, item) => {
    current[item.status] += 1;
    return current;
  }, { candidate: 0, confirmed: 0, superseded: 0 });
  const checkpoints = selectedInspection?.checkpoints ?? [];

  return <section className="scene-memory-workspace" aria-label="巡检与记忆">
    <header className="memory-hero">
      <div><p className="memory-eyebrow">SCENE TEACH</p><h2>巡检与记忆</h2><p>带机器人认识环境，再由你确认它应该记住什么。</p></div>
      <button type="button" className="memory-chat-button" onClick={onBackToChat}>返回对话移动机器人</button>
    </header>

    {loading ? <div className="memory-loading" role="status"><span />正在读取巡检与场景记忆…</div> : null}
    {unavailable ? <div className="memory-unavailable" role="alert"><strong>暂时无法使用</strong><p>{unavailable}</p><button type="button" onClick={() => void loadAll()}>重新检查</button></div> : null}
    {error ? <div className="memory-message memory-error" role="alert">{error}</div> : null}
    {notice ? <div className="memory-message memory-success" role="status">{notice}</div> : null}

    {!unavailable && capabilities ? <>
      <section className="memory-flow" aria-label="巡检步骤">
        <article className={selectedInspection?.status === "open" ? "done" : "active"}><span>1</span><div><strong>开始巡检</strong><small>绑定当前机器人与定位纪元</small></div></article>
        <article className={selectedInspection?.status === "open" ? "active" : ""}><span>2</span><div><strong>移动并停稳</strong><small>回到普通对话，由 Mission / Nav2 移动</small></div></article>
        <article className={checkpoints.some((item) => ["captured", "named", "confirmed"].includes(item.status)) ? "done" : "blocked"}><span>3</span><div><strong>观察场景</strong><small>Qwen VLM + RGB-D + TF 生成可信候选</small></div></article>
        <article className={counts.confirmed ? "done" : counts.candidate ? "active" : ""}><span>4</span><div><strong>命名并确认</strong><small>人工确认后才写入场景记忆</small></div></article>
      </section>

      <div className="memory-layout">
        <aside className="memory-session-panel">
          <h3>本次巡检</h3>
          <form onSubmit={startInspection}>
            <label>机器人<select value={robotId} onChange={(event) => setRobotId(event.target.value)}>
              {robots.map((robot) => <option key={robot.robotId} value={robot.robotId}>{robot.name ?? robot.robotId}{robot.connected ? " · 在线" : " · 离线"}</option>)}
            </select></label>
            <label>巡检说明<input value={purpose} maxLength={500} onChange={(event) => setPurpose(event.target.value)} placeholder="例如：首次办公室巡检" /></label>
            <p className={localizationEpochId ? "memory-preflight ready" : "memory-preflight"}>{selectedRobot?.connected ? "机器人在线" : "机器人离线"} · {localizationEpochId ? "定位上下文已就绪" : "缺少定位上下文"}</p>
            <button className="memory-primary" type="submit" disabled={busy || !selectedRobot?.connected || !localizationEpochId || inspections.some((item) => item.robotId === robotId && item.status === "open")}>开始新的巡检</button>
          </form>
          <div className="memory-session-list">
            <header><strong>巡检记录</strong><span>{inspections.length}</span></header>
            {inspections.map((inspection) => <button className={selectedInspection?.inspectionId === inspection.inspectionId ? "selected" : ""} type="button" key={inspection.inspectionId} onClick={() => chooseInspection(inspection)}>
              <span><strong>{inspection.purpose || "场景巡检"}</strong><small>{robots.find((robot) => robot.robotId === inspection.robotId)?.name ?? "机器人"}{developerMode ? ` · ${inspection.robotId}` : ""} · {dateTime(inspection.updatedAt)}</small></span><i>{inspection.status === "open" ? "进行中" : inspection.status === "completed" ? "已完成" : "已取消"}</i>
            </button>)}
            {!inspections.length ? <p>还没有巡检记录。</p> : null}
          </div>
          {developerMode && selectedInspection ? <details className="memory-session-provenance"><summary>开发者巡检上下文</summary><dl><div><dt>inspection</dt><dd><code>{selectedInspection.inspectionId}</code></dd></div><div><dt>robot</dt><dd><code>{selectedInspection.robotId}</code></dd></div><div><dt>epoch</dt><dd><code>{selectedInspection.localizationEpochId}</code></dd></div><div><dt>map frame</dt><dd><code>{selectedInspection.mapFrameId}</code></dd></div></dl></details> : null}
          {selectedInspection?.status === "open" ? <div className="memory-session-actions"><button type="button" disabled={busy} onClick={() => finishInspection(false)}>完成巡检</button><button className="memory-danger" type="button" disabled={busy} onClick={() => finishInspection(true)}>取消巡检</button></div> : null}
        </aside>

        <main className="memory-content">
          <section className="memory-capture-boundary" aria-label="可信采集状态">
            <div><strong>移动机器人后，在要记录的位置停稳</strong><p>移动必须回到对话，通过普通 Mission / Nav2 执行。场景记忆本身没有运动权限。</p></div>
            <button type="button" disabled>等待可信采集桥</button>
            <p className="memory-bridge-warning"><b>当前缺少服务端一键采集编排端点。</b> 浏览器不会伪造停稳位姿、Qwen VLM、深度与 TF 凭证；可信观测管线写入候选后，会自动出现在下方供你命名和确认。</p>
          </section>

          <section className="memory-library">
            <header><div><p className="memory-eyebrow">LANDMARK MEMORY</p><h3>场景记忆</h3></div><button type="button" disabled={busy} onClick={() => void loadAll(selectedInspection?.inspectionId)}>刷新</button></header>
            <nav aria-label="记忆状态筛选">
              {(["all", "candidate", "confirmed", "superseded"] as const).map((status) => <button className={filter === status ? "active" : ""} type="button" key={status} onClick={() => setFilter(status)}>{status === "all" ? "全部" : STATUS_LABELS[status]} <span>{status === "all" ? landmarks.length : counts[status]}</span></button>)}
            </nav>
            <div className="memory-landmark-list">
              {filteredLandmarks.map((landmark) => <LandmarkCard
                key={landmark.landmarkId}
                landmark={landmark}
                developerMode={developerMode}
                busy={busy}
                apiFetchBlob={apiFetchBlob}
                onConfirm={confirmLandmark}
                onRename={renameLandmark}
                onAddAlias={addAlias}
                onRemoveAlias={removeAlias}
                onInvalidate={invalidateLandmark}
              />)}
              {!filteredLandmarks.length ? <div className="memory-empty"><span>✦</span><strong>还没有{filter === "all" ? "" : STATUS_LABELS[filter]}记忆</strong><p>创建巡检并等待可信视觉采集管线生成候选。</p></div> : null}
            </div>
          </section>
        </main>
      </div>
    </> : null}
  </section>;
}

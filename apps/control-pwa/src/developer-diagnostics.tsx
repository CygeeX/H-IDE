import { useCallback, useEffect, useState } from "react";
import "./developer-diagnostics.css";

type Availability = {
  configured?: boolean;
  alive?: boolean;
  ready?: boolean;
  standby?: boolean;
  error?: string | null;
  latencyMs?: number | null;
};

type DiagnosticClassification = "physical_fault" | "authorization_hold" | "ready" | "active" | "unknown" | "degraded";

type DiagnosticSnapshot = {
  schemaVersion: "h-ide.developer-diagnostics/1";
  observedAt: string;
  target: { robotId: string; kind: "simulation" | "physical" };
  summary: {
    status: "standby" | "ready" | "active" | "degraded";
    classification: DiagnosticClassification;
    ready: boolean;
    authorizationRequired: boolean;
    motionAuthorized: boolean;
    holdReason?: string | null;
    fault?: string | null;
    processAliveIsNotReadiness: true;
    detail: string;
  };
  release: {
    hIde: string;
    agent: string;
    vision: string;
    simulation: string;
    rosDistro: string;
  };
  components: {
    hIde: Availability & { detail?: string };
    agent: Availability & { reasons?: string[] };
    vision: Availability & {
      provider?: string | null;
      model?: string | null;
      segmentation?: string | null;
      profile?: string | null;
      latestAnalysis?: {
        recordedAt: string;
        latencyMs?: number | null;
        status: string;
        evidenceCount: number;
        evidenceKinds: string[];
      } | null;
    };
    gateway: Availability & {
      connected?: boolean;
      identityMatches?: boolean;
      protocol?: string | null;
      executionBackend?: string | null;
      controlState?: string | null;
      observationStatus?: string | null;
      fault?: string | null;
    };
    robotAgent: Availability & {
      identityMatches?: boolean;
      commandReadiness?: { ready?: boolean; mode?: string | null; reason?: string | null };
      transport?: { ready?: boolean; reason?: string | null };
      control?: {
        ready?: boolean;
        state?: string | null;
        mode?: string | null;
        active?: boolean | null;
        standby?: boolean;
        canAcquire?: boolean | null;
        authorityRevoked?: boolean | null;
        safeStopState?: string | null;
        safeStopEpoch?: number | null;
        leaseValid?: boolean;
        leaseRemainingMs?: number | null;
        activeCommandCount?: number | null;
        reason?: string | null;
      };
      safeStop?: {
        ready?: boolean | null;
        authorityRevoked?: boolean | null;
        state?: string | null;
        epoch?: number | null;
        consistent?: boolean;
      };
    };
    nav2: Availability & {
      reason?: string | null;
      actions?: Record<string, { ready: boolean; reason?: string | null }>;
    };
    localization: {
      configured?: boolean;
      poseReady?: boolean;
      poseAgeMs?: number | null;
      poseMaximumAgeMs?: number | null;
      tfReady?: boolean | null;
      tfReason?: string | null;
      reason?: string | null;
    };
    observation: {
      configured?: boolean;
      ready?: boolean;
      status?: string | null;
      reason?: string | null;
      captureAgeMs?: number | null;
      healthAgeMs?: number | null;
      gatewayAckAgeMs?: number | null;
      maximumAgeMs?: number | null;
    };
    sonic: Availability & {
      motionReady?: boolean;
      mode?: string | null;
      nav2State?: string | null;
      controlClock?: string | null;
      staticHoldActive?: boolean | null;
      commandZero?: boolean | null;
      navigationTaskActive?: boolean | null;
      stopPending?: boolean | null;
      supportMode?: string | null;
      supportSafetyLatched?: boolean | null;
      faultTracePresent?: boolean | null;
      faultTraceSampleCount?: number | null;
      lastErrorKnown?: boolean;
      lastError?: string | null;
      fault?: string | null;
    };
    plant: Availability & {
      status?: string | null;
      captureSequence?: number | null;
      rgbReady?: boolean;
      depthReady?: boolean;
      rgbFrames?: number | null;
      depthFrames?: number | null;
      supportMode?: string | null;
      safetyStopLatched?: boolean;
      faultKnown?: boolean;
      fault?: string | null;
    };
  };
};

export type DeveloperDiagnosticsProps = {
  apiFetch: <T>(path: string, init?: RequestInit) => Promise<T>;
  robotId?: string;
  pollIntervalMs?: number;
};

function ms(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value)} ms` : "未报告";
}

function availabilityTone(value: Availability): "ready" | "standby" | "degraded" | "failed" | "unknown" {
  if (value.configured === false) return "unknown";
  if (value.ready && value.standby) return "standby";
  if (value.ready) return "ready";
  if (value.alive) return "degraded";
  return "failed";
}

function ReadinessBadge({ value, unknown = false }: { value: boolean | null | undefined; unknown?: boolean }) {
  if (unknown || value == null) return <span className="diag-badge unknown"><i />未知</span>;
  return <span className={`diag-badge ${value ? "ready" : "failed"}`}><i />{value ? "就绪" : "未就绪"}</span>;
}

function Fact({ label, value, bad = false }: { label: string; value: string; bad?: boolean }) {
  return <div className={`diag-fact ${bad ? "bad" : ""}`}><dt>{label}</dt><dd>{value}</dd></div>;
}

function ComponentCard({ title, subtitle, value, children }: {
  title: string;
  subtitle: string;
  value: Availability;
  children?: React.ReactNode;
}) {
  const tone = availabilityTone(value);
  return <article className={`diag-component-card tone-${tone}`}>
    <header><div><span>{subtitle}</span><h3>{title}</h3></div><span className={`diag-badge ${tone}`}><i />{
      tone === "ready" ? "就绪" : tone === "standby" ? "安全待命" : tone === "degraded" ? "存活 / 未就绪" : tone === "failed" ? "不可用" : "未配置"
    }</span></header>
    <div className="diag-process-line"><span>进程响应</span><b className={value.alive ? "ok" : "bad"}>{value.alive ? "ALIVE" : "NO RESPONSE"}</b>{value.latencyMs != null ? <small>{ms(value.latencyMs)}</small> : null}</div>
    {value.error ? <p className="diag-fault"><strong>故障</strong>{value.error}</p> : null}
    {children}
  </article>;
}

export function DeveloperDiagnostics({
  apiFetch,
  robotId,
  pollIntervalMs = 5_000,
}: DeveloperDiagnosticsProps) {
  const [snapshot, setSnapshot] = useState<DiagnosticSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const query = robotId ? `?robotId=${encodeURIComponent(robotId)}` : "";
      const next = await apiFetch<DiagnosticSnapshot>(`/api/v2/developer/diagnostics${query}`);
      setSnapshot(next);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法读取开发者诊断");
    } finally {
      setLoading(false);
    }
  }, [apiFetch, robotId]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, Math.max(2_000, pollIntervalMs));
    return () => window.clearInterval(timer);
  }, [pollIntervalMs, refresh]);

  if (!snapshot && loading) return <div className="developer-diagnostics diag-loading" role="status"><span />正在聚合只读诊断…</div>;
  if (!snapshot) return <div className="developer-diagnostics diag-empty" role="alert"><strong>诊断暂不可用</strong><p>{error}</p><button type="button" onClick={() => { setLoading(true); void refresh(); }}>重试</button></div>;

  const { components } = snapshot;
  const control = components.robotAgent.control;
  const vision = components.vision;
  const lastAnalysis = vision.latestAnalysis;
  const actions = Object.entries(components.nav2.actions ?? {});
  const authorizationHold = snapshot.summary.classification === "authorization_hold";
  const physicalFault = snapshot.summary.classification === "physical_fault";
  const heroTitle = authorizationHold
    ? "安全待命，执行前需授权"
    : snapshot.summary.classification === "active"
      ? "机器人正在执行"
      : snapshot.summary.classification === "ready"
        ? "执行链路已授权"
        : physicalFault
          ? "检测到物理执行故障"
          : snapshot.summary.classification === "unknown"
            ? "执行状态存在未知项"
            : "执行链路存在阻塞";
  const heroDescription = authorizationHold
    ? "物理安全停止已经逐层确认；当前没有运动权限，也没有活动命令。"
    : snapshot.summary.ready
      ? "所有必需组件都报告了相互一致的健康状态。"
      : "至少一个必需组件未知、故障或相互矛盾，请查看下方红色项目。";
  const overallLabel = authorizationHold
    ? "STANDBY"
    : snapshot.summary.classification === "active"
      ? "ACTIVE"
      : snapshot.summary.ready
        ? "READY"
        : "DEGRADED";

  return <section className="developer-diagnostics" aria-label="开发者系统诊断">
    <header className={`diag-hero ${snapshot.summary.status}`}>
      <div><span className="diag-kicker">READ-ONLY SYSTEM DIAGNOSTICS</span><h2>{heroTitle}</h2><p>{heroDescription}</p></div>
      <div className="diag-hero-actions"><span className={`diag-overall ${snapshot.summary.status}`}><i />{overallLabel}</span><button type="button" onClick={() => void refresh()} disabled={loading}>{loading ? "刷新中…" : "刷新只读状态"}</button></div>
      <p className="diag-readiness-contract"><strong>重要：</strong>process alive ≠ ready，ready ≠ 已获运动权限。当前授权：{snapshot.summary.motionAuthorized ? "已授权" : "未授权"}；{snapshot.summary.authorizationRequired ? "执行前必须走既有授权流程。" : "授权状态以任务运行时为准。"}</p>
      {authorizationHold && snapshot.summary.holdReason ? <p className="diag-hold-reason"><strong>待命原因</strong>{snapshot.summary.holdReason}</p> : null}
      {physicalFault && snapshot.summary.fault ? <p className="diag-summary-fault"><strong>物理故障</strong>{snapshot.summary.fault}</p> : null}
    </header>

    {error ? <div className="diag-stale-warning" role="alert"><strong>刷新失败，当前显示上一份快照</strong>{error}</div> : null}

    <div className="diag-context-bar">
      <span><small>目标</small><strong>{snapshot.target.robotId}</strong><em>{snapshot.target.kind === "simulation" ? "仿真" : "真机"}</em></span>
      <span><small>ROS</small><strong>{snapshot.release.rosDistro}</strong></span>
      <span><small>H-IDE</small><strong>{snapshot.release.hIde}</strong></span>
      <span><small>Agent</small><strong>{snapshot.release.agent}</strong></span>
      <span><small>Vision</small><strong>{snapshot.release.vision}</strong></span>
      <span><small>采样时间</small><strong>{new Date(snapshot.observedAt).toLocaleTimeString()}</strong></span>
    </div>

    <div className="diag-chain" aria-label="系统通信链路">
      <span className={components.hIde.ready ? "ready" : "failed"}>H-IDE</span><i>→</i>
      <span className={components.agent.ready ? "ready" : "failed"}>Agent</span><i>→</i>
      <span className={components.vision.ready ? "ready" : "failed"}>Vision</span><i>→</i>
      <span className={components.gateway.ready ? "ready" : "failed"}>Gateway</span><i>→</i>
      <span className={components.robotAgent.standby ? "standby" : components.robotAgent.ready ? "ready" : "failed"}>Robot Agent</span><i>→</i>
      <span className={components.nav2.ready ? "ready" : "failed"}>Nav2</span><i>→</i>
      <span className={components.sonic.standby ? "standby" : components.sonic.ready ? "ready" : "failed"}>SONIC</span><i>→</i>
      <span className={components.plant.standby ? "standby" : components.plant.ready ? "ready" : "failed"}>MuJoCo</span>
    </div>

    <div className="diag-component-grid">
      <ComponentCard title="H-IDE BFF" subtitle="WORKSTATION" value={components.hIde}>
        <p className="diag-note">{components.hIde.detail}</p>
      </ComponentCard>

      <ComponentCard title="H-IDE Agent" subtitle="PLANNING & CONVERSATION" value={components.agent}>
        {components.agent.reasons?.length ? <ul className="diag-reasons">{components.agent.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : <p className="diag-note">规划与对话服务未报告阻塞原因。</p>}
      </ComponentCard>

      <ComponentCard title="Qwen Vision" subtitle="PURE VLM + SAM 2" value={vision}>
        <dl className="diag-facts"><Fact label="语义模型" value={`${vision.provider ?? "未知"} / ${vision.model ?? "未知"}`} bad={vision.provider !== "qwen"} /><Fact label="掩码细化" value={vision.segmentation ?? "未报告"} /><Fact label="健康探测" value={ms(vision.latencyMs)} /></dl>
        {lastAnalysis ? <div className="diag-evidence"><header><strong>最近视觉凭证</strong><ReadinessBadge value={lastAnalysis.evidenceCount > 0} /></header><dl><Fact label="分析耗时" value={ms(lastAnalysis.latencyMs)} /><Fact label="凭证数量" value={String(lastAnalysis.evidenceCount)} bad={lastAnalysis.evidenceCount < 1} /><Fact label="结果" value={lastAnalysis.status} /></dl><p>{lastAnalysis.evidenceKinds.length ? lastAnalysis.evidenceKinds.join(" · ") : "未记录凭证类型"}</p><small>这里只显示凭证元数据；图像内容不会经诊断接口返回。</small></div> : <p className="diag-note">尚无站点内视觉分析凭证，无法显示模型执行耗时。</p>}
      </ComponentCard>

      <ComponentCard title="Robot Agent Gateway" subtitle="H-IDE TRANSPORT BFF" value={components.gateway}>
        <dl className="diag-facts"><Fact label="连接" value={components.gateway.connected ? "connected" : "disconnected"} bad={!components.gateway.connected} /><Fact label="协议" value={components.gateway.protocol ?? "未报告"} /><Fact label="后端" value={components.gateway.executionBackend ?? "未报告"} /><Fact label="观测" value={components.gateway.observationStatus ?? "未报告"} bad={components.gateway.observationStatus !== "USABLE"} /></dl>
        {components.gateway.fault ? <p className="diag-fault"><strong>Gateway fault</strong>{components.gateway.fault}</p> : null}
      </ComponentCard>

      <ComponentCard title="Robot Agent" subtitle="TRANSPORT / SESSION / LEASE" value={components.robotAgent}>
        <dl className="diag-facts"><Fact label="Command gate" value={components.robotAgent.commandReadiness?.ready ? "ready" : components.robotAgent.commandReadiness?.mode ?? "未就绪"} bad={!components.robotAgent.commandReadiness?.ready} /><Fact label="Transport session" value={components.robotAgent.transport?.ready ? "ready" : components.robotAgent.transport?.reason ?? "未就绪"} bad={!components.robotAgent.transport?.ready} /><Fact label="控制模式" value={control?.mode ?? "未知"} bad={!control?.ready} /><Fact label="Lease" value={control?.leaseValid ? `有效 · ${ms(control.leaseRemainingMs)}` : control?.standby ? "待命，接单时获取" : "无效"} bad={!control?.leaseValid && !control?.standby} /><Fact label="活动命令" value={control?.activeCommandCount == null ? "未知" : String(control.activeCommandCount)} bad={control?.activeCommandCount == null || (components.robotAgent.standby && control.activeCommandCount !== 0)} /><Fact label="命令权限" value={control?.authorityRevoked == null ? "未知" : control.authorityRevoked ? "已撤销" : "已授予"} bad={control?.authorityRevoked == null || (components.robotAgent.standby ? control.authorityRevoked !== true : snapshot.summary.motionAuthorized && control.authorityRevoked !== false)} /><Fact label="物理安全停止" value={control?.safeStopState ?? "未知"} bad={control?.safeStopState !== "confirmed"} /><Fact label="Stop epoch" value={control?.safeStopEpoch == null ? "未知" : String(control.safeStopEpoch)} bad={control?.safeStopEpoch == null} /></dl>
        {components.robotAgent.safeStop?.consistent === false ? <p className="diag-fault"><strong>状态矛盾</strong>Robot Agent 顶层状态与 safeStop 检查不一致。</p> : null}
        {components.robotAgent.commandReadiness?.reason && components.robotAgent.commandReadiness.reason !== control?.reason ? <p className="diag-fault"><strong>Command gate</strong>{components.robotAgent.commandReadiness.reason}</p> : null}
        {control?.reason ? <p className={`diag-inline-state ${control.standby ? "standby" : control.ready ? "ok" : "bad"}`}>{control.reason}</p> : null}
      </ComponentCard>

      <ComponentCard title="Nav2 Action" subtitle="ONLY BASE MOTION OWNER" value={components.nav2}>
        {actions.length ? <div className="diag-action-list">{actions.map(([skill, action]) => <div className={action.ready ? "ready" : "failed"} key={skill}><span><strong>{skill}</strong><small>{action.reason ?? "action server ready"}</small></span><ReadinessBadge value={action.ready} /></div>)}</div> : <p className="diag-fault"><strong>故障</strong>Robot Agent 未报告 Nav2 action readiness。</p>}
      </ComponentCard>

      <ComponentCard title="SONIC Controller" subtitle="GAIT RUNTIME" value={components.sonic}>
        <dl className="diag-facts"><Fact label="motion_ready" value={components.sonic.motionReady ? "true" : "false"} bad={!components.sonic.motionReady} /><Fact label="模式" value={components.sonic.mode ?? "未知"} bad={components.sonic.standby && components.sonic.mode !== "IDLE"} /><Fact label="Nav2 状态" value={components.sonic.nav2State ?? "未知"} bad={components.sonic.standby && components.sonic.nav2State !== "safety_stop"} /><Fact label="控制时钟" value={components.sonic.controlClock ?? "未报告"} bad={components.sonic.controlClock !== "simulation_time_50hz_motion_ready"} /><Fact label="静态保持" value={components.sonic.staticHoldActive == null ? "未知" : components.sonic.staticHoldActive ? "active" : "inactive"} bad={components.sonic.standby && components.sonic.staticHoldActive !== true} /><Fact label="速度命令" value={components.sonic.commandZero == null ? "未知" : components.sonic.commandZero ? "全零" : "非零"} bad={components.sonic.commandZero == null || (components.sonic.standby && !components.sonic.commandZero)} /><Fact label="导航任务" value={components.sonic.navigationTaskActive == null ? "未知" : components.sonic.navigationTaskActive ? "active" : "none"} bad={components.sonic.navigationTaskActive == null || (components.sonic.standby && components.sonic.navigationTaskActive)} /><Fact label="Stop pending" value={components.sonic.stopPending == null ? "未知" : components.sonic.stopPending ? "pending" : "false"} bad={components.sonic.stopPending !== false} /><Fact label="Fault trace" value={components.sonic.faultTracePresent == null ? "未知" : components.sonic.faultTracePresent ? `${components.sonic.faultTraceSampleCount ?? "?"} samples` : "none"} bad={components.sonic.faultTracePresent !== false} /></dl>
        {components.sonic.lastErrorKnown === false ? <p className="diag-fault"><strong>未知状态</strong>SONIC 未明确报告 last_error。</p> : null}
        {components.sonic.fault ? <p className="diag-fault"><strong>SONIC fault</strong>{components.sonic.fault}</p> : null}
      </ComponentCard>

      <ComponentCard title="MuJoCo Plant" subtitle="SIMULATION / RGB-D" value={components.plant}>
        <dl className="diag-facts"><Fact label="Plant" value={components.plant.status ?? "未知"} bad={!components.plant.ready} /><Fact label="支撑模式" value={components.plant.supportMode ?? "未报告"} bad={components.plant.standby ? components.plant.supportMode !== "safety_stop" : !components.plant.supportMode} /><Fact label="安全停止锁存" value={components.plant.safetyStopLatched == null ? "未知" : components.plant.safetyStopLatched ? "已锁存" : "未锁存"} bad={components.plant.safetyStopLatched == null || (components.plant.standby ? !components.plant.safetyStopLatched : components.plant.safetyStopLatched)} /><Fact label="RGB" value={components.plant.rgbReady ? `ready · ${components.plant.rgbFrames ?? "?"} 帧` : "未就绪"} bad={!components.plant.rgbReady} /><Fact label="Depth" value={components.plant.depthReady ? `ready · ${components.plant.depthFrames ?? "?"} 帧` : "未就绪"} bad={!components.plant.depthReady} /></dl>
        {components.plant.faultKnown === false ? <p className="diag-fault"><strong>未知状态</strong>Plant 未明确报告 fault。</p> : null}
        {components.plant.fault || (components.plant.safetyStopLatched && !components.plant.standby) ? <p className="diag-fault"><strong>Plant latched fault</strong>{components.plant.fault ?? "未授权状态以外出现 safety stop latch"}</p> : null}
      </ComponentCard>
    </div>

    <div className="diag-detail-grid">
      <section><header><div><span>LOCALIZATION</span><h3>位姿与 TF</h3></div><ReadinessBadge value={components.localization.poseReady} /></header><dl><Fact label="位姿年龄" value={ms(components.localization.poseAgeMs)} bad={!components.localization.poseReady} /><Fact label="准入上限" value={ms(components.localization.poseMaximumAgeMs)} /><Fact label="TF readiness" value={components.localization.tfReady == null ? "Robot Agent v1 未暴露" : components.localization.tfReady ? "ready" : "未就绪"} bad={components.localization.tfReady === false} /></dl>{components.localization.reason ? <p className="diag-fault"><strong>定位故障</strong>{components.localization.reason}</p> : null}<p className="diag-note">不会用 Nav2 action ready 推断 TF ready；未观测值明确保持未知。</p></section>
      <section><header><div><span>RGB-D OBSERVATION</span><h3>观测新鲜度</h3></div><ReadinessBadge value={components.observation.ready} /></header><dl><Fact label="采集年龄" value={ms(components.observation.captureAgeMs)} bad={!components.observation.ready} /><Fact label="健康年龄" value={ms(components.observation.healthAgeMs)} /><Fact label="Gateway ACK" value={ms(components.observation.gatewayAckAgeMs)} /><Fact label="准入上限" value={ms(components.observation.maximumAgeMs)} /></dl>{components.observation.reason ? <p className="diag-fault"><strong>观测故障</strong>{components.observation.reason}</p> : null}</section>
    </div>

    <footer className="diag-footer">只读页面 · 无控制、故障注入或急停 API · 不返回密钥、图像内容、坐标、原始 action payload 或 ROS schema</footer>
  </section>;
}

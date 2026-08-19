import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  SimulationEvent,
  SimulationLive,
  SimulationBackend,
  SimulationMode,
  SimulationRecord,
  SimulationReceipt,
  SimulationStatus,
} from "./contracts";
import "./simulation-workspace.css";

type ApiFetch = <T>(path: string, init?: RequestInit) => Promise<T>;
type ApiFetchBlob = (path: string) => Promise<Blob>;

const statusText: Record<SimulationStatus | string, string> = {
  created: "待启动",
  starting: "启动中",
  running: "运行中",
  paused: "已暂停",
  stopped: "已停止",
  failed: "失败",
};

const eventText: Record<string, string> = {
  "simulation.reset": "仿真重置",
  "simulation.started": "仿真启动",
  "simulation.paused": "仿真暂停",
  "simulation.stopped": "仿真停止",
  "navigation.goal_received": "Nav2 收到目标",
  "navigation.feedback": "Nav2 反馈",
  "navigation.succeeded": "到达目标",
  "navigation.blocked": "路径被障碍阻塞",
  "navigation.failed": "导航失败",
  "vision.target_lost": "视觉目标丢失",
  "vision.target_found": "视觉目标恢复",
  "localization.pose_updated": "位姿更新",
  "mission.bound": "已绑定 Mission",
};

function point(x: number, y: number): string {
  // World bounds are fixed by the first office-floor fixture: x=-6..6,
  // y=-4..4. Keep the conversion local to the observatory so a future
  // SceneMap adapter can replace it without changing the API contract.
  return `${((x + 6) / 12) * 100},${((4 - y) / 8) * 66.6667}`;
}

function finiteMetric(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function apiJson(body: unknown): RequestInit {
  return { method: "POST", body: JSON.stringify(body) };
}

export function SimulationWorkspace({
  apiFetch,
  apiFetchBlob,
  onBack,
  robotId = "robot-1",
  robotBootId,
  localizationEpochId,
  missionId,
}: {
  apiFetch: ApiFetch;
  apiFetchBlob: ApiFetchBlob;
  onBack: () => void;
  robotId?: string;
  robotBootId?: string;
  localizationEpochId?: string;
  missionId?: string;
}) {
  const [records, setRecords] = useState<SimulationRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [mode, setMode] = useState<SimulationMode>("oracle");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [receipt, setReceipt] = useState<SimulationReceipt | null>(null);
  const [backendStatus, setBackendStatus] = useState<Array<{ backend: string; available: boolean; installed?: boolean; reason?: string }>>([]);
  const [backend, setBackend] = useState<SimulationBackend>("mujoco_sonic");
  const [live, setLive] = useState<SimulationLive | null>(null);
  const [liveRgbObjectUrl, setLiveRgbObjectUrl] = useState<string | null>(null);
  const [autoRun, setAutoRun] = useState(false);
  const autoStepBusy = useRef(false);
  const liveRgbObjectUrlRef = useRef<string | null>(null);

  const record = useMemo(
    () => records.find((item) => item.session.sessionId === selectedId) ?? null,
    [records, selectedId],
  );

  const refresh = useCallback(async (id = selectedId) => {
    const list = await apiFetch<SimulationRecord[]>("/api/v2/simulations");
    setRecords(list);
    const nextId = id ?? list[0]?.session.sessionId ?? null;
    if (nextId) {
      const detail = await apiFetch<SimulationRecord>(`/api/v2/simulations/${encodeURIComponent(nextId)}`);
      setRecords((current) => current.map((item) => item.session.sessionId === nextId ? detail : item));
      setSelectedId(nextId);
      const history = await apiFetch<SimulationEvent[]>(`/api/v2/simulations/${encodeURIComponent(nextId)}/events`);
      setEvents(history);
      if (detail.session.backend === "mujoco_sonic") {
        await apiFetch<SimulationRecord & { live?: SimulationLive }>(`/api/v2/simulations/${encodeURIComponent(nextId)}/live`)
          .then((value) => {
            setLive(value.live ?? null);
            setRecords((current) => current.map((item) => item.session.sessionId === nextId ? value : item));
          })
          .catch(() => setLive(null));
      } else {
        setLive(null);
      }
    }
  }, [apiFetch, selectedId]);

  useEffect(() => {
    void refresh().catch((cause: unknown) => setError(cause instanceof Error ? cause.message : "无法读取仿真状态"));
    void apiFetch<Array<{ backend: string; available: boolean; installed?: boolean; reason?: string }>>("/api/v2/simulations/backends")
      .then((items) => {
        setBackendStatus(items);
        const liveBackend = items.find((item) => item.backend === "mujoco_sonic" && item.available);
        setBackend(liveBackend ? "mujoco_sonic" : "deterministic");
      })
      .catch(() => setBackendStatus([]));
  }, [apiFetch, refresh]);

  useEffect(() => {
    const sessionId = record?.session.sessionId;
    const sequence = live?.state.sequence;
    if (!sessionId || record?.session.backend !== "mujoco_sonic" || !sequence) return undefined;
    let cancelled = false;
    void apiFetchBlob(`/api/v2/simulations/${encodeURIComponent(sessionId)}/rgb.jpg`)
      .then((blob) => {
        if (cancelled) return;
        const nextUrl = URL.createObjectURL(blob);
        const previous = liveRgbObjectUrlRef.current;
        liveRgbObjectUrlRef.current = nextUrl;
        setLiveRgbObjectUrl(nextUrl);
        if (previous) URL.revokeObjectURL(previous);
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(cause instanceof Error ? cause.message : "无法读取仿真第一视角");
      });
    return () => {
      cancelled = true;
    };
  }, [apiFetchBlob, live?.state.sequence, record?.session.backend, record?.session.sessionId]);

  useEffect(() => () => {
    if (liveRgbObjectUrlRef.current) URL.revokeObjectURL(liveRgbObjectUrlRef.current);
  }, []);

  useEffect(() => {
    if (!record || record.session.status !== "running") return undefined;
    const timer = window.setInterval(() => {
      void refresh(record.session.sessionId).catch(() => undefined);
    }, 600);
    return () => window.clearInterval(timer);
  }, [record, refresh]);

  useEffect(() => {
    if (!autoRun) return undefined;
    if (!record || record.session.status !== "running" || !record.snapshot.goalPose) {
      setAutoRun(false);
      return undefined;
    }
    const sessionId = record.session.sessionId;
    const timer = window.setInterval(() => {
      if (autoStepBusy.current) return;
      autoStepBusy.current = true;
      void apiFetch<{ receipt?: SimulationReceipt }>(
        `/api/v2/simulations/${encodeURIComponent(sessionId)}/step`,
        apiJson({ distanceM: 0.25 }),
      ).then((result) => {
        if (result.receipt) setReceipt(result.receipt);
        if (result.receipt?.status && ["succeeded", "blocked", "failed"].includes(result.receipt.status)) {
          setAutoRun(false);
        }
        return refresh(sessionId);
      }).catch(() => setAutoRun(false)).finally(() => {
        autoStepBusy.current = false;
      });
    }, 450);
    return () => window.clearInterval(timer);
  }, [apiFetch, autoRun, record, refresh]);

  const run = async (operation: () => Promise<{ snapshot?: unknown; receipt?: SimulationReceipt } | SimulationRecord>) => {
    setBusy(true);
    setError("");
    try {
      const result = await operation();
      if (result && "receipt" in result && result.receipt) setReceipt(result.receipt);
      await refresh(selectedId ?? undefined);
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "仿真操作失败");
    } finally {
      setBusy(false);
    }
  };

  const create = () => run(async () => {
    const created = await apiFetch<SimulationRecord>("/api/v2/simulations", apiJson({
      backend,
      mode,
      robotId: backend === "mujoco_sonic" ? "g1-sim" : robotId,
      robotBootId: backend === "mujoco_sonic" ? undefined : robotBootId,
      localizationEpochId: backend === "mujoco_sonic" ? undefined : localizationEpochId,
      missionId,
    }));
    setSelectedId(created.session.sessionId);
    return created;
  });

  const action = (name: "start" | "pause" | "reset" | "stop") => {
    if (!record) return;
    if (name !== "start") setAutoRun(false);
    return run(() => apiFetch<SimulationRecord>(`/api/v2/simulations/${encodeURIComponent(record.session.sessionId)}/${name}`, apiJson({})));
  };

  const step = () => {
    if (!record || record.session.backend === "mujoco_sonic") return;
    return run(() => apiFetch<{ receipt?: SimulationReceipt }>(`/api/v2/simulations/${encodeURIComponent(record.session.sessionId)}/step`, apiJson({ distanceM: 0.25 })));
  };

  const navigateTo = (landmarkId: string) => {
    if (!record) return;
    const landmark = record.snapshot.landmarks.find((item) => item.landmarkId === landmarkId);
    if (!landmark) return;
    return run(() => apiFetch<{ receipt?: SimulationReceipt }>(
      `/api/v2/simulations/${encodeURIComponent(record.session.sessionId)}/navigation-goals`,
      apiJson({ goalPose: landmark.pose }),
    ));
  };

  const effectivePose = live?.pose ?? record?.snapshot.pose;
  const liveTelemetry = live?.telemetry ?? null;
  const liveNavigationStatus = live?.navigation?.status ?? null;
  const liveRootVelocity = (live?.state.rootVelocity ?? null) as Record<string, unknown> | null;
  const liveSupport = (live?.state.support ?? null) as Record<string, unknown> | null;
  const legTrackingMax = finiteMetric(liveTelemetry?.lower_body_target_max_error_rad);
  const armTaskTrackingPeak = finiteMetric(liveTelemetry?.navigation_task_peak_arm_tracking_error_rad);
  const armTaskConstrainedPeak = finiteMetric(liveTelemetry?.navigation_task_peak_arm_constrained_joints);
  const terminalPoseRelation = String(live?.navigation?.result?.finalPoseRelation ?? "");
  const terminalPoseEvidence = ({
    latest_before_terminal: "等待成功后位姿",
    awaiting_post_terminal: "等待成功后位姿",
    settling_after_terminal: "等待机身静止",
    stable_after_terminal: "稳定姿态已锁存",
    first_at_or_after_terminal: "终止后姿态已锁存",
    settlement_timeout: "稳定等待超时降级",
  } as Record<string, string>)[terminalPoseRelation] ?? "—";

  const inject = (eventType: "vision.target_lost" | "navigation.blocked") => {
    if (!record) return;
    return run(() => apiFetch<SimulationRecord>(
      `/api/v2/simulations/${encodeURIComponent(record.session.sessionId)}/events`,
      apiJson({ eventType, payload: { source: "observatory" } }),
    ));
  };

  const syncReview = () => {
    if (!record || !record.snapshot.pendingReview?.missionId) return;
    return run(() => apiFetch<SimulationRecord>(
      `/api/v2/simulations/${encodeURIComponent(record.session.sessionId)}/mission-review/resolve`,
      apiJson({ revisionId: record.snapshot.pendingReview?.revisionId ?? undefined }),
    ));
  };

  return <section className="simulation-observatory" aria-label="仿真观测台">
    <header className="simulation-heading">
      <div>
        <span className="simulation-kicker">SIMULATION OBSERVATORY</span>
        <h2>连续任务仿真联调</h2>
        <p>MuJoCo + SONIC 联调观测台。导航仍只允许经 Robot Agent → Nav2；每次任务级重规划必须人工审核。</p>
      </div>
      <div className="simulation-heading-actions">
        <span className="simulation-safety-badge">SIMULATION · NO PHYSICAL ACTUATOR</span>
        <button className="mini-button" type="button" onClick={onBack}>返回聊天</button>
      </div>
    </header>

    {error ? <div className="simulation-error" role="alert">{error}<button type="button" onClick={() => setError("")}>关闭</button></div> : null}

      <div className="simulation-toolbar">
      <label>视觉模式
        <select value={mode} onChange={(event) => setMode(event.target.value as SimulationMode)} disabled={busy}>
          <option value="oracle">Oracle · 确定性</option>
          <option value="qwen">Qwen · VLM（后端配置后启用）</option>
        </select>
      </label>
      <label>执行后端
        <select value={backend} onChange={(event) => setBackend(event.target.value as SimulationBackend)} disabled={busy}>
          <option value="mujoco_sonic">MuJoCo + SONIC · ROS/Nav2</option>
          <option value="deterministic">确定性基线（迁移中）</option>
        </select>
      </label>
      <button className="primary-button" type="button" onClick={create} disabled={busy}>新建仿真会话</button>
      <span className={`simulation-backend-badge ${backendStatus.find((item) => item.backend === "mujoco_sonic")?.available ? "ready" : "pending"}`}>{backendStatus.length ? (backendStatus.find((item) => item.backend === "mujoco_sonic")?.available ? "MuJoCo + SONIC 就绪" : "MuJoCo + SONIC 未就绪") : "后端检查中"}</span>
      <select className="simulation-session-select" value={selectedId ?? ""} onChange={(event) => { setSelectedId(event.target.value || null); void refresh(event.target.value || undefined); }} aria-label="选择仿真会话">
        <option value="">选择会话</option>
        {records.map((item) => <option key={item.session.sessionId} value={item.session.sessionId}>{item.session.sessionId.slice(0, 16)} · {statusText[item.session.status]}</option>)}
      </select>
    </div>

    {!record ? <div className="simulation-empty"><strong>还没有仿真会话</strong><span>创建会话后可以在浏览器中观察机器人移动、障碍事件和连续规划状态。</span></div> : <>
      <div className="simulation-grid">
        <section className="simulation-map-card">
          <div className="simulation-card-header"><div><strong>办公楼层 / map</strong><small>{record.session.backend} · {record.session.mode.toUpperCase()} · {record.session.missionId ? `Mission ${record.session.missionId}` : "未绑定 Mission"}</small></div><span className={`simulation-status ${record.session.status}`}>{statusText[record.session.status]}</span></div>
          <svg className="simulation-map" viewBox="0 0 100 66.6667" role="img" aria-label="办公楼层仿真地图">
            <rect x="0" y="0" width="100" height="66.6667" className="sim-floor" />
            <line x1="0" y1={((4 - 1.15) / 8) * 66.6667} x2="100" y2={((4 - 1.15) / 8) * 66.6667} className="sim-wall" />
            {record.snapshot.obstacles.map((obstacle) => { const [x, y] = point(obstacle.center.x, obstacle.center.y).split(",").map(Number); const radius = (obstacle.radiusM / 12) * 100; return <circle key={obstacle.obstacleId} cx={x} cy={y} r={radius} className="sim-obstacle" />; })}
            {record.snapshot.landmarks.map((landmark) => { const [x, y] = point(landmark.pose.x, landmark.pose.y).split(",").map(Number); return <g key={landmark.landmarkId}><circle cx={x} cy={y} r="1.5" className="sim-landmark" /><text x={x + 1.8} y={y - 1.8} className="sim-label">{landmark.label}</text></g>; })}
            {record.snapshot.plannedWaypoints.length ? <polyline points={[record.snapshot.pose, ...record.snapshot.plannedWaypoints].map((pose) => point(pose.x, pose.y)).join(" ")} className="sim-planned-path" /> : null}
            {record.snapshot.trajectory.length > 1 ? <polyline points={record.snapshot.trajectory.map((pose) => point(pose.x, pose.y)).join(" ")} className="sim-trajectory" /> : null}
            {record.snapshot.goalPose ? <circle cx={Number(point(record.snapshot.goalPose.x, record.snapshot.goalPose.y).split(",")[0])} cy={Number(point(record.snapshot.goalPose.x, record.snapshot.goalPose.y).split(",")[1])} r="2.4" className="sim-goal" /> : null}
            <g transform={`translate(${point(effectivePose?.x ?? 0, effectivePose?.y ?? 0)}) rotate(${-(effectivePose?.yaw ?? 0) * 180 / Math.PI})`}>
              <circle cx="0" cy="0" r="2.2" className="sim-robot" /><path d="M0 -4 L2 1 L-2 1 Z" className="sim-heading-arrow" />
            </g>
          </svg>
          <div className="simulation-map-legend"><span><i className="legend-robot" />机器人</span><span><i className="legend-goal" />目标</span><span><i className="legend-obstacle" />固定障碍</span><span><i className="legend-landmark" />语义区域</span><span><i className="legend-planned" />Nav2 路径</span></div>
          {live ? <div className="simulation-live-evidence">
            <div className="simulation-camera-frame">{liveRgbObjectUrl
              ? <img src={liveRgbObjectUrl} alt="MuJoCo 仿真头部相机画面" />
              : <span>正在读取 MuJoCo 第一视角…</span>}
            </div>
            <div className="simulation-live-caption"><span className="live-dot" />{live.ready ? "实时 RGB-D / 根位姿" : "仿真观测未就绪"}<small>source_mode=synthetic · domain=232 · head_motion=fixed</small></div>
          </div> : null}
        </section>

        <section className="simulation-control-card">
          <div className="simulation-card-header"><div><strong>控制与事件注入</strong><small>{record.session.backend === "mujoco_sonic" ? "真实 Robot Agent/Nav2 回执 · 无物理执行器" : "只产生 simulated receipt"}</small></div><span className={`simulation-nav-state ${record.snapshot.navigationState}`}>{record.snapshot.navigationState}</span></div>
          <div className="simulation-action-row"><button type="button" onClick={() => action("start")} disabled={busy || record.session.status === "running"}>启动</button><button type="button" onClick={() => action("pause")} disabled={busy || record.session.status !== "running"}>暂停</button><button type="button" onClick={() => action("reset")} disabled={busy}>重置</button><button type="button" onClick={() => action("stop")} disabled={busy || record.session.status === "stopped"}>停止</button></div>
          <div className="simulation-action-group"><strong>{record.session.backend === "mujoco_sonic" ? "发送 Robot Agent → Nav2 目标" : "发送模拟 Nav2 目标"}</strong><div className="simulation-action-row">{record.snapshot.landmarks.slice(0, 2).map((landmark) => <button key={landmark.landmarkId} type="button" onClick={() => navigateTo(landmark.landmarkId)} disabled={busy || autoRun || record.session.status !== "running"}>去{landmark.label}</button>)}{record.session.backend !== "mujoco_sonic" ? <><button type="button" onClick={step} disabled={busy || autoRun || record.session.status !== "running" || !record.snapshot.goalPose}>推进 0.25m</button><button type="button" onClick={() => setAutoRun((current) => !current)} disabled={busy || record.session.status !== "running" || !record.snapshot.goalPose}>{autoRun ? "停止连续推进" : "连续推进"}</button></> : null}</div></div>
          <div className="simulation-action-group"><strong>故障注入</strong><div className="simulation-action-row"><button type="button" onClick={() => inject("navigation.blocked")} disabled={busy}>模拟阻塞</button><button type="button" onClick={() => inject("vision.target_lost")} disabled={busy}>模拟目标丢失</button></div></div>
          {record.snapshot.pendingReview ? <div className={`simulation-review ${record.snapshot.pendingReview.status}`} role="status">
            <strong>{record.snapshot.pendingReview.status === "error" ? "审核桥接异常" : "需要人工审核"}</strong>
            <span>{record.snapshot.pendingReview.detail}</span>
            {record.snapshot.pendingReview.revisionId ? <small>PlanRevision · {record.snapshot.pendingReview.revisionId}</small> : <small>尚未生成可执行修订；不会发送任何物理命令</small>}
            {record.snapshot.pendingReview.missionId ? <button type="button" onClick={syncReview} disabled={busy}>同步 Mission 审核状态</button> : null}
          </div> : null}
          <dl className="simulation-metrics"><div><dt>当前位置</dt><dd>{(effectivePose?.x ?? 0).toFixed(2)}, {(effectivePose?.y ?? 0).toFixed(2)}</dd></div><div><dt>轨迹点</dt><dd>{record.snapshot.trajectory.length}</dd></div><div><dt>事件序号</dt><dd>#{record.snapshot.lastEventSequence}</dd></div><div><dt>授权策略</dt><dd>{record.session.policy.manualRevisionApproval ? "每次修订人工确认" : "受控自动"}</dd></div></dl>
          {liveTelemetry ? <div className="simulation-live-metrics">
            <span>Robot Agent <b>{liveNavigationStatus ?? "—"}</b></span>
            <span>Nav2 / SONIC <b>{String(liveTelemetry.nav2_state ?? "—")} · {String(liveTelemetry.sonic_mode ?? "—")}</b></span>
            <span>Nav2 请求速度 <b>{finiteMetric(liveTelemetry.sonic_speed_mps).toFixed(2)} m/s</b></span>
            <span>SONIC 有效速度 <b>{finiteMetric(liveTelemetry.sonic_policy_speed_mps).toFixed(2)} m/s</b></span>
            <span>机身实测速度 <b>{finiteMetric(liveRootVelocity?.linearSpeedMps).toFixed(3)} m/s</b></span>
            <span className={terminalPoseRelation === "settlement_timeout" ? "metric-warning" : undefined} title="Nav2 结束后，H-IDE 等待机身连续稳定再把位姿写入审计事件">
              终点证据 <b>{terminalPoseEvidence}</b>
            </span>
            <span>仿真支撑 <b>{String(liveSupport?.mode ?? liveTelemetry.plant_support_mode ?? "—")}</b></span>
            <span className={legTrackingMax > 0.5 ? "metric-warning" : undefined} title="SONIC 目标角与 MuJoCo 实际关节角的瞬时误差">
              腿部跟踪误差 <b>RMS {finiteMetric(liveTelemetry.lower_body_target_rms_error_rad).toFixed(2)} · MAX {legTrackingMax.toFixed(2)} rad</b>
            </span>
            <span title="当前实际手臂关节相对标称站姿的最大偏离，以及当前最大关节速度">
              摆臂当前值 <b>{finiteMetric(liveTelemetry.navigation_arm_measured_max_deviation_rad).toFixed(2)} rad · {finiteMetric(liveTelemetry.navigation_arm_max_velocity_rps).toFixed(2)} rad/s</b>
            </span>
            <span className={armTaskTrackingPeak > 0.5 || armTaskConstrainedPeak > 0 ? "metric-warning" : undefined} title="从本次 Nav2 任务首个非零速度到停止期间记录，开始下一任务时重置">
              本任务摆臂峰值 <b>{liveTelemetry.navigation_task_active ? "记录中 · " : "已冻结 · "}跟踪 {armTaskTrackingPeak.toFixed(2)} rad · 约束 {armTaskConstrainedPeak.toFixed(0)} 关节</b>
            </span>
            <span title="本次任务内腿部目标与实际关节角的最大瞬时差值">
              本任务腿部峰值 <b>RMS {finiteMetric(liveTelemetry.navigation_task_peak_lower_body_target_rms_error_rad).toFixed(2)} · MAX {finiteMetric(liveTelemetry.navigation_task_peak_lower_body_target_max_error_rad).toFixed(2)} rad</b>
            </span>
            <span>停止交接 <b>{liveTelemetry.static_hold_active ? "实测姿态锁存" : "策略控制"}</b></span>
            <span>cmd age <b>{liveTelemetry.cmd_vel_age_s == null ? "—" : `${finiteMetric(liveTelemetry.cmd_vel_age_s).toFixed(2)} s`}</b></span>
          </div> : null}
          {receipt ? <div className="simulation-receipt"><strong>最近回执 · {receipt.status}</strong><span>{receipt.motion} · {receipt.executionBackend}</span><small>{receipt.detail ?? "—"}</small></div> : null}
        </section>
      </div>

      <section className="simulation-events-card"><div className="simulation-card-header"><div><strong>事件时间线</strong><small>所有事件可回放，仿真不会伪装成真机执行</small></div><span>{events.length} events</span></div><div className="simulation-events-list">{events.slice().reverse().slice(0, 12).map((event) => <article key={event.eventId}><span>#{event.sequence}</span><div><strong>{eventText[event.eventType] ?? event.eventType}</strong><small>{event.source} · {new Date(event.occurredAt).toLocaleTimeString()}</small></div><code>{event.eventType}</code></article>)}</div></section>
    </>}
  </section>;
}

import { useCallback, useEffect, useState } from "react";
import type { RobotState } from "./contracts";
import "./robot-monitor.css";

type Props = { apiFetch: <T>(path: string, init?: RequestInit) => Promise<T>; pollIntervalMs?: number };
type Row = { robot: RobotState; diagnostics?: { summary?: { status?: string; detail?: string; ready?: boolean; holdReason?: string | null; fault?: string | null }; components?: Record<string, { ready?: boolean; alive?: boolean; connected?: boolean; error?: string | null }> }; error?: string };

function liveConnected(row: Row) {
  const gateway = row.diagnostics?.components?.gateway;
  return gateway ? Boolean(gateway.alive && gateway.connected) : row.robot.connected;
}

function label(row: Row) {
  if (row.error) return ["不可用", "failed"];
  if (row.robot.status === "estop") return ["急停", "failed"];
  if (row.diagnostics?.summary?.ready || liveConnected(row)) return [row.diagnostics?.summary?.status === "ready" ? "就绪" : "在线 / 待命", "ready"];
  if (row.robot.status === "unknown") return ["未知", "unknown"];
  return ["离线", "failed"];
}

export function RobotMonitor({ apiFetch, pollIntervalMs = 5000 }: Props) {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>();
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    try {
      const robots = await apiFetch<RobotState[]>("/api/v1/robots");
      const next = await Promise.all(robots.map(async (robot): Promise<Row> => {
        try {
          const diagnostics = await apiFetch<Row["diagnostics"]>(`/api/v2/developer/diagnostics?robotId=${encodeURIComponent(robot.robotId)}`);
          return { robot, diagnostics };
        } catch (reason) {
          return { robot, error: reason instanceof Error ? reason.message : "诊断接口无响应" };
        }
      }));
      setRows(next); setLastUpdated(new Date().toLocaleTimeString()); setError("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "无法读取机器人列表"); }
    finally { setLoading(false); }
  }, [apiFetch]);
  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), Math.max(2000, pollIntervalMs)); return () => window.clearInterval(timer); }, [refresh, pollIntervalMs]);
  const available = rows.filter((row) => label(row)[1] === "ready").length;
  return <section className="robot-monitor" aria-label="机器人监控台">
    <header className="robot-monitor-heading"><div><span className="robot-monitor-kicker">ROBOT CONFIGURATION MONITOR</span><h2>机器人监控台</h2><p>只读聚合机器人配置、连接和执行链健康状态</p></div><div className="robot-monitor-summary"><strong>{available}/{rows.length}</strong><span>当前可用</span><small>{lastUpdated ? `更新于 ${lastUpdated}` : "检查中…"}</small></div></header>
    {error ? <div className="robot-monitor-error" role="alert">{error}<button type="button" onClick={() => { setLoading(true); void refresh(); }}>重试</button></div> : null}
    {loading && !rows.length ? <div className="robot-monitor-empty">正在检查机器人配置…</div> : null}
    {!loading && !rows.length ? <div className="robot-monitor-empty"><strong>没有发现机器人配置</strong><span>请确认 Robot Agent 或仿真网关已注册。</span></div> : null}
    <div className="robot-monitor-grid">{rows.map((row) => { const [status, tone] = label(row); const details = row.robot.details ?? {}; const diagnostic = row.diagnostics?.summary; const connected = liveConnected(row); return <article className={`robot-monitor-card ${tone}`} key={row.robot.robotId}>
      <div className="robot-monitor-card-head"><div><span>{details.simulation ? "仿真 · Unitree G1" : "Robot Agent"}</span><h3>{row.robot.name ?? row.robot.robotId}</h3><code>{row.robot.robotId}</code></div><b>{status}</b></div>
      <dl><div><dt>连接</dt><dd>{connected ? "已连接" : "未连接"}</dd></div><div><dt>安全模式</dt><dd>{row.robot.safetyMode}</dd></div><div><dt>执行后端</dt><dd>{String(details.executionBackend ?? details.adapter ?? "未报告")}</dd></div><div><dt>机器人启动 ID</dt><dd>{String(details.robotBootId ?? "未报告")}</dd></div></dl>
      <p className="robot-monitor-reason">{row.error ?? diagnostic?.fault ?? diagnostic?.holdReason ?? diagnostic?.detail ?? (connected ? "连接正常，等待完整就绪诊断" : "服务未连接或未通过实时健康检查")}</p>
      <div className="robot-monitor-link"><span>可用性判断：连接 + 诊断就绪 + 无故障</span>{row.diagnostics ? <span>{Object.values(row.diagnostics.components ?? {}).filter((component) => component.ready).length} 项组件已就绪</span> : null}</div>
    </article>; })}</div>
  </section>;
}

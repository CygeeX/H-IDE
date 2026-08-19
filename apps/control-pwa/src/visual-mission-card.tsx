import { useEffect, useState } from "react";
import type { AssistantDraft, RobotState, VisualMissionRecord } from "./contracts";
import { isVisualMissionPlan, statusLabel } from "./contracts";
import { VisualEvidenceDrawer } from "./visual-evidence-drawer";

export function VisualMissionCard({
  draft,
  robot,
  mission,
  busy,
  expired,
  reviewable,
  expiry,
  onConfirm,
  onRun,
  onSelectCandidate,
  onCancel,
  feedback,
  setFeedback,
  onReject,
  onRevise,
  onLoadEvidence,
  onReplan,
}: {
  draft: AssistantDraft;
  robot: RobotState | null;
  mission: VisualMissionRecord | null;
  busy: boolean;
  expired: boolean;
  reviewable: boolean;
  expiry: string;
  onConfirm: () => void;
  onRun: () => void;
  onSelectCandidate?: (candidateId: string) => void;
  onCancel: () => void;
  feedback?: string;
  setFeedback?: (value: string) => void;
  onReject?: () => void;
  onRevise?: () => void;
  onLoadEvidence?: (missionId: string, sha256: string) => Promise<string>;
  onReplan: () => void;
}) {
  const [holding, setHolding] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [timer, setTimer] = useState<number | null>(null);
  useEffect(() => () => { if (timer != null) window.clearTimeout(timer); }, [timer]);
  if (!isVisualMissionPlan(draft.plan)) return null;
  const spec = draft.plan.mission;
  const isObserve = mission?.mode === "observe";
  const canRun = mission?.status === "approved" && mission.mode === "lab_execute" && !busy;
  const active = mission && ["resolving", "dispatching", "verifying"].includes(mission.status);
  const ambiguityStage = mission?.status === "needs_input"
    ? [...mission.stages].reverse().find(
      (stage) => stage.status === "needs_input" && stage.state?.ambiguityCandidates?.length,
    )
    : undefined;
  const ambiguityCandidates = ambiguityStage?.state?.ambiguityCandidates ?? [];
  const startHold = () => {
    if (!reviewable || timer != null) return;
    setHolding(true);
    setTimer(window.setTimeout(() => {
      setHolding(false);
      setTimer(null);
      onConfirm();
    }, 900));
  };
  const stopHold = () => {
    if (timer != null) window.clearTimeout(timer);
    setTimer(null);
    setHolding(false);
  };
  return <article className={`visual-mission-card visual-${mission?.status ?? draft.status}`}>
    <header>
      <div><span className="visual-kicker">VISION MISSION</span><h2>{spec.targetDescription}</h2><p>{spec.instruction}</p></div>
      <span className={`status-pill ${mission?.status ?? draft.status}`}>{statusLabel(mission?.status ?? (expired ? "expired" : draft.status))}</span>
    </header>
    <div className="visual-simulation-banner">{mission?.mode === "lab_execute" || spec.requestedSourceMode !== "live" ? "运动为模拟 · 不代表正式或物理验收" : "视觉观察模式"}</div>
    <div className="visual-mission-facts">
      <span><small>机器人</small><strong>{spec.robotId}</strong></span>
      <span><small>输入源</small><strong>{spec.requestedSourceMode} · {spec.sensorIds.join("、")}</strong></span>
      <span><small>融合模式</small><strong>{spec.fusionMode}</strong></span>
      <span><small>审核有效期</small><strong>{expiry}</strong></span>
    </div>
    {spec.robotId !== robot?.robotId ? <p className="visual-warning">当前机器人与草案不一致，不能执行。</p> : null}
    <ol className="visual-stage-list">
      {spec.stages.map((stage, index) => {
        const state = mission?.stages.find((item) => item.stageId === stage.stageId);
        return <li key={stage.stageId}><span>{index + 1}</span><div><strong>{stage.targetDescription}</strong><small>{stage.kind} · {stage.goalPurpose} · {state?.visualResolutionId ?? "确认后生成坐标"}</small></div><em>{statusLabel(state?.status ?? "pending")}</em></li>;
      })}
    </ol>
    {ambiguityCandidates.length ? <section className="visual-candidate-selection" aria-label="选择视觉候选目标">
      <strong>画面中有多个匹配目标</strong>
      <p>请选择一个不可变候选。选择只会批准新的视觉解析，不会立即发送机器人命令。</p>
      <div>{ambiguityCandidates.map((candidate, index) => <button
        key={candidate.candidateId}
        type="button"
        disabled={busy || !onSelectCandidate}
        onClick={() => onSelectCandidate?.(candidate.candidateId)}
        aria-label={`选择候选 ${index + 1}：${candidate.label}`}
      ><span>{index + 1}</span><strong>{candidate.label}</strong><small>{candidate.score == null ? "模型未提供置信度" : `${Math.round(candidate.score * 100)}%`}</small></button>)}</div>
    </section> : null}
    <details className="visual-review-details">
      <summary>查看边界与版本</summary>
      <p>单目标最大 {spec.maximumGoalDistanceM} m · 累计 {spec.maximumCumulativeDistanceM} m · 速度上限 {spec.maximumSpeedMps} m/s · {spec.maximumDurationSeconds} 秒</p>
      <p>{spec.vlmProvider} / {spec.vlmModel} · {spec.visualAlgorithmVersion} · ApproachPolicy {spec.approachPolicyVersion}</p>
      <p>不支持：{spec.unsupportedCapabilities.join("、")}</p>
    </details>
    <div className="visual-mission-actions">
      {draft.status === "draft" && !expired ? <button
        className={`visual-hold-confirm ${holding ? "holding" : ""}`}
        type="button"
        disabled={!reviewable}
        onPointerDown={startHold}
        onPointerUp={stopHold}
        onPointerCancel={stopHold}
        onPointerLeave={stopHold}
        onClick={(event) => { if (event.detail === 0 && reviewable) onConfirm(); }}
      >{holding ? "继续按住…" : "按住审核视觉任务"}</button> : null}
      {expired ? <button type="button" onClick={onReplan} disabled={busy}>重新规划</button> : null}
      {canRun ? <button className="visual-run" type="button" onClick={onRun}>开始模拟目标解析与发送</button> : null}
      {isObserve && mission?.status === "approved" ? <span className="observe-note">observe 模式只允许审核，不能发送目标</span> : null}
      {active ? <button type="button" onClick={onCancel}>取消视觉任务</button> : null}
      {mission ? <button type="button" onClick={() => setDrawerOpen(true)}>查看视觉证据</button> : null}
    </div>
    {draft.status === "draft" && !expired && setFeedback && onReject && onRevise ? <div className="visual-review-controls">
      <label htmlFor={`visual-review-feedback-${draft.draftId}`}>修改意见</label>
      <textarea
        id={`visual-review-feedback-${draft.draftId}`}
        aria-label="视觉任务审核意见"
        value={feedback ?? ""}
        onChange={(event) => setFeedback(event.target.value)}
        rows={2}
        maxLength={2000}
        placeholder="例如：改为停在目标左侧，或只观察不要接近"
      />
      <div>
        <button type="button" onClick={onRevise} disabled={!reviewable || !(feedback ?? "").trim()}>要求修改</button>
        <button className="reject-plan" type="button" onClick={onReject} disabled={!reviewable}>拒绝</button>
      </div>
    </div> : null}
    {mission?.error ? <p className="visual-warning">{mission.error}</p> : null}
    {drawerOpen && mission ? <VisualEvidenceDrawer
      mission={mission}
      onClose={() => setDrawerOpen(false)}
      onLoadEvidence={onLoadEvidence}
    /> : null}
  </article>;
}

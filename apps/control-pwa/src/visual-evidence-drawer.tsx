import { useEffect, useState } from "react";
import type { VisualMissionRecord } from "./contracts";
import { statusLabel } from "./contracts";

export function VisualEvidenceDrawer({
  mission,
  onClose,
  onLoadEvidence,
}: {
  mission: VisualMissionRecord;
  onClose: () => void;
  onLoadEvidence?: (missionId: string, sha256: string) => Promise<string>;
}) {
  const stage = mission.stages.at(-1);
  const receipt = stage?.state?.receipt;
  const payload = mission.plan.mission;
  const imageRef = stage?.state?.evidence?.find((item) => item.mediaType.startsWith("image/"));
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageError, setImageError] = useState("");
  useEffect(() => {
    let disposed = false;
    let loadedUrl: string | null = null;
    if (!imageRef || !onLoadEvidence) return undefined;
    setImageError("");
    void onLoadEvidence(mission.missionId, imageRef.sha256).then((url) => {
      loadedUrl = url;
      if (disposed) URL.revokeObjectURL(url);
      else setImageUrl(url);
    }).catch((error: unknown) => {
      if (!disposed) setImageError(error instanceof Error ? error.message : "无法读取视觉证据");
    });
    return () => {
      disposed = true;
      if (loadedUrl) URL.revokeObjectURL(loadedUrl);
    };
  }, [imageRef?.sha256, mission.missionId, onLoadEvidence]);
  return <div className="visual-drawer-backdrop" role="presentation" onMouseDown={(event) => {
    if (event.target === event.currentTarget) onClose();
  }}>
    <aside className="visual-evidence-drawer" role="dialog" aria-modal="true" aria-labelledby="visual-evidence-title">
      <header>
        <div><small>VISUAL EVIDENCE</small><h2 id="visual-evidence-title">视觉任务依据</h2></div>
        <button type="button" onClick={onClose} aria-label="关闭视觉证据">×</button>
      </header>
      <section className="evidence-health-ladder" aria-label="视觉健康状态">
        <span className="healthy"><i />任务已审核</span>
        <span className={stage?.visualResolutionId ? "healthy" : "pending"}><i />RGB-D 目标解析</span>
        <span className={receipt ? "healthy" : "pending"}><i />目标发送回执</span>
        <span className="pending"><i />Robot Agent / Nav2 正式验收待完成</span>
      </section>
      {imageRef ? <figure className="visual-keyframe">
        {imageUrl ? <img src={imageUrl} alt={`目标关键帧：${payload.targetDescription}`} /> : <div>{imageError || "正在读取受控关键帧…"}</div>}
        <figcaption>{imageRef.kind} · {imageRef.sha256.slice(0, 12)}… · 不缓存</figcaption>
      </figure> : null}
      <dl className="evidence-facts">
        <div><dt>目标</dt><dd>{payload.targetDescription}</dd></div>
        <div><dt>数据源</dt><dd>{payload.requestedSourceMode} · {payload.sensorIds.join("、")}</dd></div>
        <div><dt>视觉模型</dt><dd>{payload.vlmProvider} / {payload.vlmModel}</dd></div>
        <div><dt>算法版本</dt><dd>{payload.visualAlgorithmVersion}</dd></div>
        <div><dt>接近策略</dt><dd>{payload.approachPolicyVersion} · {payload.approachPolicySha256.slice(0, 12)}…</dd></div>
        <div><dt>Robot boot / epoch</dt><dd>{payload.robotBootId} / {payload.localizationEpochId}</dd></div>
        <div><dt>视觉解析 ID</dt><dd>{stage?.visualResolutionId ?? "尚未解析"}</dd></div>
        <div><dt>阶段状态</dt><dd>{statusLabel(stage?.status ?? mission.status)}</dd></div>
      </dl>
      {stage?.state?.goalPose ? <section className="evidence-explanation"><strong>建议目标点（map）</strong><p>{stage.state.goalPose.x.toFixed(2)}, {stage.state.goalPose.y.toFixed(2)}, {stage.state.goalPose.z.toFixed(2)} · yaw {stage.state.goalPose.yaw.toFixed(2)}</p></section> : null}
      {stage?.state?.safetyGates?.length ? <section className="evidence-gates"><strong>确定性安全门</strong><ul>{stage.state.safetyGates.map((gate) => <li key={gate.name} className={gate.passed ? "passed" : "failed"}><span>{gate.passed ? "通过" : "拒绝"}</span>{gate.name}{gate.measured != null ? ` · ${gate.measured}` : ""}</li>)}</ul></section> : null}
      {stage?.state?.explanation ? <section className="evidence-explanation"><strong>视觉 Worker 说明</strong><p>{stage.state.explanation}</p></section> : null}
      {receipt ? <section className="evidence-receipt"><strong>ActionRuntime 回执</strong><code>{receipt.commandId}</code><p>{statusLabel(receipt.status)}{receipt.error ? ` · ${receipt.error}` : ""}</p></section> : null}
      <p className="evidence-boundary">当前证据不包含路径或速度。工作站只确定一个 map 目标点；路径规划、避障与运动属于 Robot Agent/Nav2。是否完成本地预联调以版本化集成报告为准；当前状态不代表正式或物理验收。</p>
    </aside>
  </div>;
}

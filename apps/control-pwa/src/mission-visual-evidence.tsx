import { useEffect, useMemo, useState } from "react";
import type { MissionRecordV2 } from "./contracts";

interface EvidenceRef {
  kind: string;
  sha256: string;
  mediaType: string;
}

interface VisualHypothesis {
  hypothesisId: string;
  source: "qwen";
  label: string;
  pixelBox: [number, number, number, number];
  score: number | null;
}

interface SafetyGate {
  name: string;
  passed: boolean;
  measured?: number | null;
  threshold?: number | null;
  detail?: string | null;
}

interface MapPose {
  frameId: string;
  x: number;
  y: number;
  z: number;
  yaw: number;
}

interface GeometryCandidate {
  geometryCandidateId: string;
  role: string;
  verified: boolean;
  landmarkPose?: MapPose | null;
  targetPose?: MapPose | null;
  safetyGates: SafetyGate[];
}

export interface MissionVisualResult {
  analysisId: string;
  mode: string;
  status: string;
  source: {
    sourceMode: string;
    sourceId: string;
    sensorId: string;
    capturedAt: string;
  };
  versions: {
    vlmProvider: "qwen";
    vlmModel: "qwen3-vl-plus";
    visualAlgorithmVersion: string;
    segmentationProvider: "sam2";
    segmentationModel?: string | null;
  };
  hypotheses: VisualHypothesis[];
  geometryCandidates: GeometryCandidate[];
  observationSummary: string | null;
  directionalClear: boolean | null;
  rejectionReason: string | null;
  evidence: EvidenceRef[];
  warnings: string[];
}

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function mapPose(value: unknown): MapPose | null {
  const item = object(value);
  if (!item || typeof item.frameId !== "string"
    || !finite(item.x) || !finite(item.y) || !finite(item.z) || !finite(item.yaw)) return null;
  return { frameId: item.frameId, x: item.x, y: item.y, z: item.z, yaw: item.yaw };
}

function parseVisualResult(value: unknown): MissionVisualResult | null {
  const visual = object(value);
  const source = object(visual?.source);
  const versions = object(visual?.versions);
  if (!visual || visual.schemaVersion !== "h-ide.visual-result/2"
    || typeof visual.analysisId !== "string" || typeof visual.mode !== "string"
    || typeof visual.status !== "string" || !source || !versions) return null;
  const hypotheses = Array.isArray(visual.hypotheses) ? visual.hypotheses.flatMap((raw) => {
    const item = object(raw);
    const box = item?.pixelBox;
    if (!item || typeof item.hypothesisId !== "string" || typeof item.label !== "string"
      || item.source !== "qwen"
      || !Array.isArray(box) || box.length !== 4 || !box.every(finite)) return [];
    return [{
      hypothesisId: item.hypothesisId,
      source: item.source as VisualHypothesis["source"],
      label: item.label,
      pixelBox: box as [number, number, number, number],
      score: finite(item.score) ? item.score : null,
    }];
  }) : [];
  const geometryCandidates = Array.isArray(visual.geometryCandidates)
    ? visual.geometryCandidates.flatMap((raw) => {
      const item = object(raw);
      if (!item || typeof item.geometryCandidateId !== "string" || typeof item.role !== "string") return [];
      const safetyGates = Array.isArray(item.safetyGates) ? item.safetyGates.flatMap((rawGate) => {
        const gate = object(rawGate);
        if (!gate || typeof gate.name !== "string" || typeof gate.passed !== "boolean") return [];
        return [{
          name: gate.name,
          passed: gate.passed,
          measured: finite(gate.measured) ? gate.measured : null,
          threshold: finite(gate.threshold) ? gate.threshold : null,
          detail: typeof gate.detail === "string" ? gate.detail : null,
        }];
      }) : [];
      return [{
        geometryCandidateId: item.geometryCandidateId,
        role: item.role,
        verified: item.verified === true,
        landmarkPose: mapPose(item.landmarkPose),
        targetPose: mapPose(item.targetPose),
        safetyGates,
      }];
    }) : [];
  const evidence = Array.isArray(visual.evidence) ? visual.evidence.flatMap((raw) => {
    const item = object(raw);
    if (!item || typeof item.kind !== "string" || typeof item.sha256 !== "string"
      || typeof item.mediaType !== "string") return [];
    return [{ kind: item.kind, sha256: item.sha256, mediaType: item.mediaType }];
  }) : [];
  if (typeof source.sourceMode !== "string" || typeof source.sourceId !== "string"
    || typeof source.sensorId !== "string" || typeof source.capturedAt !== "string"
    || versions.vlmProvider !== "qwen" || versions.vlmModel !== "qwen3-vl-plus"
    || versions.segmentationProvider !== "sam2"
    || (versions.poseModel != null && versions.poseModel !== "")
    || typeof versions.visualAlgorithmVersion !== "string") return null;
  return {
    analysisId: visual.analysisId,
    mode: visual.mode,
    status: visual.status,
    source: {
      sourceMode: source.sourceMode,
      sourceId: source.sourceId,
      sensorId: source.sensorId,
      capturedAt: source.capturedAt,
    },
    versions: {
      vlmProvider: "qwen",
      vlmModel: "qwen3-vl-plus",
      visualAlgorithmVersion: versions.visualAlgorithmVersion,
      segmentationProvider: "sam2",
      segmentationModel: typeof versions.segmentationModel === "string" ? versions.segmentationModel : null,
    },
    hypotheses,
    geometryCandidates,
    observationSummary: typeof visual.observationSummary === "string"
      ? visual.observationSummary
      : null,
    directionalClear: typeof visual.directionalClear === "boolean"
      ? visual.directionalClear
      : null,
    rejectionReason: typeof visual.rejectionReason === "string"
      ? visual.rejectionReason
      : null,
    evidence,
    warnings: Array.isArray(visual.warnings)
      ? visual.warnings.filter((item): item is string => typeof item === "string")
      : [],
  };
}

export function latestMissionVisualResult(mission: MissionRecordV2): MissionVisualResult | null {
  for (const rawExecution of [...mission.skillExecutions].reverse()) {
    const execution = object(rawExecution);
    const skillResult = object(execution?.result);
    const result = object(skillResult?.result);
    const parsed = parseVisualResult(result?.visualResult);
    if (parsed) return parsed;
  }
  return null;
}

export function MissionVisualSummary({
  result,
  observationOnly,
  onOpen,
}: {
  result: MissionVisualResult;
  observationOnly: boolean;
  onOpen: () => void;
}) {
  const sceneDescription = result.mode === "scene_description";
  const directional = result.mode === "directional_free_space";
  const hypothesis = result.hypotheses[0];
  const geometry = result.geometryCandidates.find((item) => item.verified)
    ?? result.geometryCandidates[0];
  const pose = geometry?.targetPose ?? geometry?.landmarkPose;
  const clearance = geometry?.safetyGates.find((gate) => gate.name === "directional_clearance_m");
  const motionSuppressed = directional && result.directionalClear === false;
  return <section className="mission-visual-summary" aria-label="Mission 视觉结果">
    <div>
      <small>{sceneDescription ? "场景观察" : directional ? "方向净空" : "视觉结果"}</small>
      <strong>{sceneDescription
        ? result.observationSummary ?? "场景描述不可用"
        : directional
          ? result.directionalClear ? "前方净空满足本次移动" : "前方净空不足"
          : hypothesis?.label ?? "未识别到目标"}</strong>
      <span>{hypothesis
        ? `${hypothesis.source}${hypothesis.score == null ? "" : ` · ${(hypothesis.score * 100).toFixed(0)}%`}`
        : directional ? "RGB-D 深度扇区检测" : result.status}</span>
    </div>
    <div>
      <small>{sceneDescription
        ? "可见对象"
        : directional ? "可用净空 / 所需净空" : geometry?.targetPose ? "Nav2 候选目标（map）" : "目标位置（map）"}</small>
      <strong>{sceneDescription
        ? `${result.hypotheses.length} 个已识别对象（场景描述不生成坐标）`
        : directional
          ? clearance?.measured != null && clearance.threshold != null
            ? `${clearance.measured.toFixed(2)} m / ${clearance.threshold.toFixed(2)} m`
            : "净空测量不可用"
          : pose ? `${pose.x.toFixed(2)}, ${pose.y.toFixed(2)}, ${pose.z.toFixed(2)}` : "未形成可信坐标"}</strong>
      <span>{sceneDescription
        ? "黄色框仅表示可见物体"
        : directional ? result.directionalClear ? "确定性安全门已通过" : "安全门拒绝本次移动"
          : geometry?.verified ? "RGB-D 几何门已通过" : "几何仍待确认"}</span>
    </div>
    <div className={observationOnly || motionSuppressed ? "no-motion" : "nav2-boundary"}>
      <small>运动边界</small>
      <strong>{motionSuppressed
        ? "净空不足，已跳过本次导航"
        : observationOnly ? "仅观察，未向 Nav2 发动作" : "底盘动作仅由 Nav2 执行"}</strong>
      <button type="button" onClick={onOpen}>查看视觉证据</button>
    </div>
  </section>;
}

export function MissionVisualEvidenceDrawer({
  missionId,
  result,
  observationOnly,
  onClose,
  onLoadEvidence,
}: {
  missionId: string;
  result: MissionVisualResult;
  observationOnly: boolean;
  onClose: () => void;
  onLoadEvidence: (missionId: string, analysisId: string, sha256: string) => Promise<string>;
}) {
  const sceneDescription = result.mode === "scene_description";
  const directional = result.mode === "directional_free_space";
  const imageRef = result.evidence.find((item) => item.mediaType.startsWith("image/"));
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageError, setImageError] = useState("");
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  useEffect(() => {
    let disposed = false;
    let loadedUrl: string | null = null;
    setImageUrl(null);
    setImageError("");
    if (!imageRef) return undefined;
    void onLoadEvidence(missionId, result.analysisId, imageRef.sha256).then((url) => {
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
  }, [imageRef?.sha256, missionId, onLoadEvidence, result.analysisId]);
  const overlays = useMemo(() => dimensions.width && dimensions.height
    ? result.hypotheses.map((item) => ({
      ...item,
      style: {
        left: `${100 * item.pixelBox[0] / dimensions.width}%`,
        top: `${100 * item.pixelBox[1] / dimensions.height}%`,
        width: `${100 * (item.pixelBox[2] - item.pixelBox[0]) / dimensions.width}%`,
        height: `${100 * (item.pixelBox[3] - item.pixelBox[1]) / dimensions.height}%`,
      },
    })) : [], [dimensions, result.hypotheses]);
  const geometry = result.geometryCandidates.find((item) => item.verified)
    ?? result.geometryCandidates[0];
  const pose = geometry?.targetPose ?? geometry?.landmarkPose;
  const geometryComputed = Boolean(geometry);
  const safetyPassed = Boolean(geometry?.safetyGates.length)
    && geometry!.safetyGates.every((gate) => gate.passed);
  return <div className="visual-drawer-backdrop" role="presentation" onMouseDown={(event) => {
    if (event.target === event.currentTarget) onClose();
  }}>
    <aside className="visual-evidence-drawer mission-v2-evidence" role="dialog" aria-modal="true" aria-labelledby="mission-visual-title">
      <header>
        <div><small>MISSION V2 · VISUAL EVIDENCE</small><h2 id="mission-visual-title">视觉理解与空间依据</h2></div>
        <button type="button" onClick={onClose} aria-label="关闭视觉证据">×</button>
      </header>
      <section className="evidence-health-ladder" aria-label="视觉处理链状态">
        <span className={sceneDescription
          ? result.observationSummary ? "healthy" : "failed"
          : directional ? "healthy" : result.hypotheses.length ? "healthy" : "failed"}><i />
          {sceneDescription ? "场景语义描述" : directional ? "无需语义目标（深度净空模式）" : "语义目标识别"}
        </span>
        <span className={sceneDescription ? "healthy" : geometryComputed ? "healthy" : "failed"}><i />
          {sceneDescription ? "场景观察无需 map 坐标" : directional ? "RGB-D 深度扇区测量" : "RGB-D / TF 三维定位"}
        </span>
        <span className={sceneDescription ? "healthy" : safetyPassed ? "healthy" : "failed"}><i />
          {sceneDescription ? "无运动几何门" : safetyPassed ? "确定性安全门通过" : "确定性安全门拒绝"}
        </span>
        <span className={observationOnly || result.directionalClear === false ? "healthy" : "pending"}><i />
          {result.directionalClear === false
            ? "净空不足，导航分支已跳过"
            : observationOnly ? "仅观察，无底盘动作" : "后续运动必须交给 Nav2"}
        </span>
      </section>
      {imageRef ? <figure className="visual-keyframe mission-keyframe">
        {imageUrl ? <div className="mission-evidence-image-wrap">
          <img src={imageUrl} alt="机器人第一视角关键帧" onLoad={(event) => setDimensions({
            width: event.currentTarget.naturalWidth,
            height: event.currentTarget.naturalHeight,
          })} />
          {overlays.map((item) => <span className="mission-evidence-box" style={item.style} key={item.hypothesisId}>
            <b>{item.label}{item.score == null ? "" : ` · ${(item.score * 100).toFixed(0)}%`}</b>
          </span>)}
        </div> : <div>{imageError || "正在读取受控关键帧…"}</div>}
        <figcaption>{imageRef.kind} · {imageRef.sha256.slice(0, 12)}… · {directional
          ? "本模式只分析 RGB-D 净空，不生成语义框"
          : `黄色框来自 ${result.hypotheses[0]?.source ?? "视觉模型"}`}</figcaption>
      </figure> : <p className="evidence-boundary">该视觉结果没有可显示的 RGB 关键帧。</p>}
      <dl className="evidence-facts">
        <div><dt>分析 ID</dt><dd>{result.analysisId}</dd></div>
        <div><dt>输入源</dt><dd>{result.source.sourceMode} · {result.source.sourceId}</dd></div>
        <div><dt>传感器</dt><dd>{result.source.sensorId}</dd></div>
        <div><dt>语义模型</dt><dd>{result.versions.vlmProvider} / {result.versions.vlmModel}</dd></div>
        <div><dt>几何算法</dt><dd>{result.versions.visualAlgorithmVersion}</dd></div>
        <div><dt>分割算法</dt><dd>{result.versions.segmentationModel ?? "模型原生定位"}</dd></div>
        <div><dt>采集时间</dt><dd>{new Date(result.source.capturedAt).toLocaleString("zh-CN")}</dd></div>
      </dl>
      {result.observationSummary ? <section className="evidence-explanation"><strong>场景观察结论</strong><p>{result.observationSummary}</p></section> : null}
      {pose ? <section className="evidence-explanation"><strong>{geometry?.targetPose ? "建议导航目标点（map）" : "语义目标位置（map）"}</strong><p>{pose.x.toFixed(3)}, {pose.y.toFixed(3)}, {pose.z.toFixed(3)} · yaw {pose.yaw.toFixed(3)}</p></section> : null}
      {geometry?.safetyGates.length ? <section className="evidence-gates"><strong>确定性安全门</strong><ul>{geometry.safetyGates.map((gate) => <li key={gate.name} className={gate.passed ? "passed" : "failed"}><span>{gate.passed ? "通过" : "拒绝"}</span>{gate.name}{gate.measured != null ? ` · 实测 ${gate.measured.toFixed(3)}` : ""}{gate.threshold != null ? ` / 阈值 ${gate.threshold.toFixed(3)}` : ""}</li>)}</ul></section> : null}
      <p className="evidence-boundary">{sceneDescription
        ? "黄色框是当前画面的语义观察，不代表可导航坐标。该模式不会发出底盘动作。"
        : directional
          ? "方向净空由深度图确定性计算，不要求识别语义目标；净空不足会跳过导航。任何位移和原地转向仍必须经过 Robot Agent 与 Nav2。"
          : "黄色框是语义假设；三维坐标由深度、相机标定和 TF 确定性计算。视觉结果不包含路径或速度，任何位移和原地转向都必须经过 Robot Agent 与 Nav2。"}</p>
    </aside>
  </div>;
}

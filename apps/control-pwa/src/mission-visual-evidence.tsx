import { useEffect, useMemo, useState, type CSSProperties } from "react";
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

export interface VisualAmbiguityCandidate {
  candidateId: string;
  label: string;
  box: [number, number, number, number];
  score: number | null;
}

interface SafetyGate {
  name: string;
  passed: boolean;
  measured?: number | string | boolean | null;
  threshold?: number | string | boolean | null;
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

const VISUAL_TASK_MODES = [
  "object_approach",
  "landmark_observation",
  "directional_free_space",
  "arrival_verification",
  "scene_description",
] as const;
type VisualTaskMode = typeof VISUAL_TASK_MODES[number];

const VISUAL_TERMINAL_STATUSES = ["succeeded", "needs_input", "rejected", "failed"] as const;
type VisualTerminalStatus = typeof VISUAL_TERMINAL_STATUSES[number];

function isVisualTaskMode(value: string): value is VisualTaskMode {
  return (VISUAL_TASK_MODES as readonly string[]).includes(value);
}

function isVisualTerminalStatus(value: string): value is VisualTerminalStatus {
  return (VISUAL_TERMINAL_STATUSES as readonly string[]).includes(value);
}

export interface MissionVisualResult {
  analysisId: string;
  mode: VisualTaskMode;
  status: VisualTerminalStatus;
  source: {
    robotId: string;
    robotBootId: string;
    gatewayInstanceId: string;
    observationSessionId: string;
    localizationEpochId: string;
    sourceMode: "live" | "replay" | "synthetic";
    sourceId: string;
    bundleId: string;
    sensorId: string;
    capturedAt: string;
    calibrationSha256: string;
    sceneId: string | null;
    sceneRevision: number | null;
  };
  versions: {
    vlmProvider: "qwen";
    vlmModel: "qwen3-vl-plus";
    promptVersion: string;
    visualAlgorithmVersion: string;
    approachPolicyVersion: string;
    approachPolicySha256: string;
    segmentationProvider: "sam2";
    segmentationModel?: string | null;
  };
  hypotheses: VisualHypothesis[];
  ambiguityCandidates: VisualAmbiguityCandidate[];
  geometryCandidates: GeometryCandidate[];
  observationSummary: string | null;
  directionalClear: boolean | null;
  rejectionReason: string | null;
  evidence: EvidenceRef[];
  warnings: string[];
}

export function MissionVisualCandidateSelection({
  result,
  busy,
  onSelect,
  missionId,
  onLoadEvidence,
}: {
  result: MissionVisualResult;
  busy: boolean;
  onSelect: (candidateId: string, analysisId: string, label: string) => void;
  missionId?: string;
  onLoadEvidence?: (
    missionId: string,
    analysisId: string,
    sha256: string,
  ) => Promise<string>;
}) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageError, setImageError] = useState(false);
  const imageEvidence = result.evidence.find((item) => item.mediaType.startsWith("image/"));
  useEffect(() => {
    let disposed = false;
    let loadedUrl: string | null = null;
    setImageUrl(null);
    setImageError(false);
    if (!missionId || !onLoadEvidence || !imageEvidence) return () => undefined;
    void onLoadEvidence(missionId, result.analysisId, imageEvidence.sha256)
      .then((url) => {
        if (disposed) {
          if (typeof URL.revokeObjectURL === "function") {
            URL.revokeObjectURL(url);
          }
          return;
        }
        loadedUrl = url;
        setImageUrl(url);
      })
      .catch(() => {
        if (!disposed) setImageError(true);
      });
    return () => {
      disposed = true;
      if (loadedUrl && typeof URL.revokeObjectURL === "function") {
        URL.revokeObjectURL(loadedUrl);
      }
    };
  }, [imageEvidence?.sha256, missionId, onLoadEvidence, result.analysisId]);
  if (!result.ambiguityCandidates.length) return null;
  const colors = ["#8b5cf6", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444"];
  return <section className="visual-candidate-selection" aria-label="选择 Mission 视觉候选">
    <strong>请选择你指的是哪一个目标</strong>
    <p>候选编号与视觉凭证中的同色框一一对应；选择后会对同一目标再观察一次，不会直接触发移动。</p>
    {imageUrl ? <div className="candidate-evidence-stage" aria-label="带编号候选框的视觉凭证">
      <svg viewBox="0 0 640 480" role="img" aria-label="视觉候选编号图">
        <image href={imageUrl} x="0" y="0" width="640" height="480" preserveAspectRatio="xMidYMid slice" />
        {result.ambiguityCandidates.map((candidate, index) => {
          const [x1, y1, x2, y2] = candidate.box;
          const color = colors[index % colors.length];
          return <g key={candidate.candidateId}>
            <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={color} strokeWidth="5" />
            <circle cx={x1 + 15} cy={Math.max(18, y1 - 15)} r="16" fill={color} />
            <text x={x1 + 15} y={Math.max(18, y1 - 15) + 6} textAnchor="middle" fill="white" fontSize="18" fontWeight="700">{index + 1}</text>
          </g>;
        })}
      </svg>
    </div> : imageError ? <p className="candidate-evidence-error">视觉凭证暂时未能加载；不会允许盲选，请重试加载。</p> : onLoadEvidence ? <p className="candidate-evidence-loading">正在加载带编号的视觉凭证…</p> : null}
    <div>{result.ambiguityCandidates.map((candidate, index) => <button
      key={candidate.candidateId}
      type="button"
      disabled={busy || Boolean(onLoadEvidence && !imageUrl)}
      style={{ "--candidate-color": colors[index % colors.length] } as CSSProperties}
      onClick={() => onSelect(candidate.candidateId, result.analysisId, candidate.label)}
    >
      {imageUrl ? <svg className="candidate-crop" viewBox={`${candidate.box[0]} ${candidate.box[1]} ${candidate.box[2] - candidate.box[0]} ${candidate.box[3] - candidate.box[1]}`} aria-hidden="true">
        <image href={imageUrl} x="0" y="0" width="640" height="480" preserveAspectRatio="xMidYMid slice" />
      </svg> : null}
      <span>{index + 1}</span>
      <strong>{candidate.label}</strong>
      <small>{candidate.score == null ? "置信度未知" : `置信度 ${(candidate.score * 100).toFixed(0)}%`}</small>
    </button>)}</div>
  </section>;
}

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && Boolean(value.trim());
}

function sha256(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function timestamp(value: unknown): value is string {
  return typeof value === "string"
    && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value)
    && Number.isFinite(Date.parse(value));
}

const SHA256_ROUND_CONSTANTS = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function rotateRight(value: number, bits: number): number {
  return (value >>> bits) | (value << (32 - bits));
}

function sha256Text(value: string): string {
  const bytes = new TextEncoder().encode(value);
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  const bitLength = bytes.length * 8;
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x1_0000_0000), false);
  view.setUint32(paddedLength - 4, bitLength >>> 0, false);
  const hash = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const words = new Uint32Array(64);
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) {
      words[index] = view.getUint32(offset + index * 4, false);
    }
    for (let index = 16; index < 64; index += 1) {
      const left = words[index - 15];
      const right = words[index - 2];
      const sigma0 = rotateRight(left, 7) ^ rotateRight(left, 18) ^ (left >>> 3);
      const sigma1 = rotateRight(right, 17) ^ rotateRight(right, 19) ^ (right >>> 10);
      words[index] = (words[index - 16] + sigma0 + words[index - 7] + sigma1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = hash;
    for (let index = 0; index < 64; index += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choice = (e & f) ^ (~e & g);
      const temporary1 = (h + sum1 + choice + SHA256_ROUND_CONSTANTS[index] + words[index]) >>> 0;
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temporary2 = (sum0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temporary1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temporary1 + temporary2) >>> 0;
    }
    hash[0] = (hash[0] + a) >>> 0;
    hash[1] = (hash[1] + b) >>> 0;
    hash[2] = (hash[2] + c) >>> 0;
    hash[3] = (hash[3] + d) >>> 0;
    hash[4] = (hash[4] + e) >>> 0;
    hash[5] = (hash[5] + f) >>> 0;
    hash[6] = (hash[6] + g) >>> 0;
    hash[7] = (hash[7] + h) >>> 0;
  }
  return [...hash].map((word) => word.toString(16).padStart(8, "0")).join("");
}

const VISUAL_TASK_FLOAT_PATHS = new Set([
  "requestedDistanceM",
  "verificationMaximumDistanceM",
  "verificationMaximumYawRad",
  "expectedTargetPose.x",
  "expectedTargetPose.y",
  "expectedTargetPose.z",
  "expectedTargetPose.yaw",
]);

function pythonFloat(value: number): string {
  if (Object.is(value, -0)) return "-0.0";
  const sign = value < 0 ? "-" : "";
  const raw = Math.abs(value).toString();
  const exponentMarker = raw.indexOf("e");
  if (exponentMarker >= 0) {
    const coefficient = raw.slice(0, exponentMarker);
    const exponent = Number(raw.slice(exponentMarker + 1));
    return `${sign}${coefficient}e${exponent >= 0 ? "+" : "-"}${Math.abs(exponent).toString().padStart(2, "0")}`;
  }
  const [integerPart, fractionPart = ""] = raw.split(".");
  let digits = `${integerPart}${fractionPart}`;
  let exponent = integerPart === "0"
    ? -(fractionPart.search(/[1-9]/) + 1)
    : integerPart.length - 1;
  if (integerPart === "0" && !/[1-9]/.test(fractionPart)) exponent = 0;
  digits = digits.replace(/^0+/, "").replace(/0+$/, "") || "0";
  if (exponent < -4 || exponent >= 16) {
    const coefficient = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    return `${sign}${coefficient}e${exponent >= 0 ? "+" : "-"}${Math.abs(exponent).toString().padStart(2, "0")}`;
  }
  return `${sign}${raw}${Number.isInteger(value) ? ".0" : ""}`;
}

function pythonCanonicalJson(
  value: unknown,
  path: string[] = [],
): string | undefined {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return undefined;
    if (VISUAL_TASK_FLOAT_PATHS.has(path.join("."))) return pythonFloat(value);
    return Number.isSafeInteger(value) ? String(value) : undefined;
  }
  if (Array.isArray(value)) {
    const items = value.map((item, index) => pythonCanonicalJson(item, [...path, String(index)]));
    return items.some((item) => item === undefined) ? undefined : `[${items.join(",")}]`;
  }
  const item = object(value);
  if (!item) return undefined;
  const keys = Object.keys(item).sort();
  const fields: string[] = [];
  for (const key of keys) {
    const encoded = pythonCanonicalJson(item[key], [...path, key]);
    if (encoded === undefined) return undefined;
    fields.push(`${JSON.stringify(key)}:${encoded}`);
  }
  return `{${fields.join(",")}}`;
}

/** SHA-256 of the canonical JSON used by H-IDE Vision's VisualTaskRequest/2. */
export function visualTaskRequestSha256(value: Record<string, unknown>): string | null {
  const canonical = pythonCanonicalJson(value);
  return canonical === undefined ? null : sha256Text(canonical);
}

function safetyValue(value: unknown): value is number | string | boolean | null | undefined {
  return value == null || typeof value === "string" || typeof value === "boolean" || finite(value);
}

function formatSafetyValue(value: number | string | boolean): string {
  return typeof value === "number" ? value.toFixed(3) : String(value);
}

function mapPose(value: unknown): MapPose | null {
  const item = object(value);
  if (!item || typeof item.frameId !== "string" || !item.frameId.trim()
    || !finite(item.x) || !finite(item.y) || !finite(item.z) || !finite(item.yaw)) return null;
  return { frameId: item.frameId.trim(), x: item.x, y: item.y, z: item.z, yaw: item.yaw };
}

export function parseMissionVisualResult(value: unknown): MissionVisualResult | null {
  const visual = object(value);
  const source = object(visual?.source);
  const versions = object(visual?.versions);
  const analysisId = typeof visual?.analysisId === "string" ? visual.analysisId.trim() : "";
  const mode = typeof visual?.mode === "string" ? visual.mode.trim() : "";
  const status = typeof visual?.status === "string" ? visual.status.trim() : "";
  if (!visual || visual.schemaVersion !== "h-ide.visual-result/2"
    || !analysisId || !sha256(visual.requestSha256)
    || !nonEmptyString(visual.requestId) || !nonEmptyString(visual.missionId)
    || !nonEmptyString(visual.revisionId) || !nonEmptyString(visual.nodeId)
    || !Number.isInteger(visual.attempt) || (visual.attempt as number) < 1
    || (visual.attempt as number) > 20 || !timestamp(visual.createdAt)
    || !isVisualTaskMode(mode) || !isVisualTerminalStatus(status)
    || !source || !versions) return null;

  const hypotheses: VisualHypothesis[] = [];
  if (visual.hypotheses != null && !Array.isArray(visual.hypotheses)) return null;
  for (const raw of visual.hypotheses ?? []) {
    const item = object(raw);
    const box = item?.pixelBox;
    const hypothesisId = typeof item?.hypothesisId === "string" ? item.hypothesisId.trim() : "";
    const label = typeof item?.label === "string" ? item.label.trim() : "";
    if (!item || !hypothesisId || !label
      || item.source !== "qwen"
      || !Array.isArray(box) || box.length !== 4
      || !box.every((coordinate) => Number.isInteger(coordinate) && coordinate >= 0)
      || box[0] >= box[2] || box[1] >= box[3]
      || (item.score != null && (!finite(item.score) || item.score < 0 || item.score > 1))) return null;
    hypotheses.push({
      hypothesisId,
      source: "qwen",
      label,
      pixelBox: box as [number, number, number, number],
      score: finite(item.score) ? item.score : null,
    });
  }

  const geometryCandidates: GeometryCandidate[] = [];
  if (visual.geometryCandidates != null && !Array.isArray(visual.geometryCandidates)) return null;
  for (const raw of visual.geometryCandidates ?? []) {
    const item = object(raw);
    const geometryCandidateId = typeof item?.geometryCandidateId === "string"
      ? item.geometryCandidateId.trim()
      : "";
    const role = typeof item?.role === "string" ? item.role.trim() : "";
    if (!item || !geometryCandidateId
      || item.source !== "deterministic_rgbd"
      || !["landmark", "approach", "free_space", "verification"].includes(role)
      || typeof item.verified !== "boolean"
      || (item.landmarkPose != null && !mapPose(item.landmarkPose))
      || (item.targetPose != null && !mapPose(item.targetPose))
      || !Array.isArray(item.safetyGates)
      || (item.evidenceIds != null && (!Array.isArray(item.evidenceIds)
        || item.evidenceIds.some((evidenceId) => !nonEmptyString(evidenceId))))) return null;
    const safetyGates: SafetyGate[] = [];
    for (const rawGate of item.safetyGates ?? []) {
      const gate = object(rawGate);
      const name = typeof gate?.name === "string" ? gate.name.trim() : "";
      if (!gate || !name || typeof gate.passed !== "boolean"
        || !safetyValue(gate.measured)
        || !safetyValue(gate.threshold)
        || (gate.detail != null && typeof gate.detail !== "string")) return null;
      safetyGates.push({
        name,
        passed: gate.passed,
        measured: gate.measured == null ? null : gate.measured,
        threshold: gate.threshold == null ? null : gate.threshold,
        detail: typeof gate.detail === "string" ? gate.detail : null,
      });
    }
    const landmarkPose = item.landmarkPose == null ? null : mapPose(item.landmarkPose);
    const targetPose = item.targetPose == null ? null : mapPose(item.targetPose);
    const gatesPassed = safetyGates.length > 0 && safetyGates.every((gate) => gate.passed);
    if (item.verified !== gatesPassed || (!landmarkPose && !targetPose)
      || (["approach", "free_space"].includes(role) && !targetPose)) return null;
    geometryCandidates.push({
      geometryCandidateId,
      role,
      verified: item.verified,
      landmarkPose,
      targetPose,
      safetyGates,
    });
  }

  const ambiguityCandidates: VisualAmbiguityCandidate[] = [];
  const candidateIds = new Set<string>();
  if (visual.ambiguityCandidates != null) {
    if (!Array.isArray(visual.ambiguityCandidates)) return null;
    for (const raw of visual.ambiguityCandidates) {
      const item = object(raw);
      const box = item?.box;
      if (!item || typeof item.candidateId !== "string"
        || typeof item.label !== "string"
        || !Array.isArray(box) || box.length !== 4 || !box.every(Number.isInteger)
        || (item.score != null && !finite(item.score))) return null;
      const candidateId = item.candidateId.trim();
      const label = item.label.trim();
      if (!candidateId || !label || candidateIds.has(candidateId)) return null;
      candidateIds.add(candidateId);
      ambiguityCandidates.push({
        candidateId,
        label,
        box: box as [number, number, number, number],
        score: finite(item.score) ? item.score : null,
      });
    }
  }

  const evidence: EvidenceRef[] = [];
  if (visual.evidence != null && !Array.isArray(visual.evidence)) return null;
  for (const raw of visual.evidence ?? []) {
    const item = object(raw);
    const kind = typeof item?.kind === "string" ? item.kind.trim() : "";
    const sha256 = typeof item?.sha256 === "string" ? item.sha256.trim() : "";
    const mediaType = typeof item?.mediaType === "string" ? item.mediaType.trim() : "";
    if (!item || !kind || !/^[0-9a-f]{64}$/.test(sha256) || !mediaType) return null;
    evidence.push({ kind, sha256, mediaType });
  }

  if (!nonEmptyString(source.robotId)
    || !nonEmptyString(source.robotBootId)
    || !nonEmptyString(source.gatewayInstanceId)
    || !nonEmptyString(source.observationSessionId)
    || !nonEmptyString(source.localizationEpochId)
    || typeof source.sourceMode !== "string"
    || !["live", "replay", "synthetic"].includes(source.sourceMode)
    || !nonEmptyString(source.sourceId)
    || !nonEmptyString(source.bundleId)
    || !nonEmptyString(source.sensorId)
    || !timestamp(source.capturedAt)
    || !sha256(source.calibrationSha256)
    || (source.sceneId != null && !nonEmptyString(source.sceneId))
    || (source.sceneRevision != null
      && (typeof source.sceneRevision !== "number"
        || !Number.isInteger(source.sceneRevision) || source.sceneRevision < 1))
    || versions.vlmProvider !== "qwen" || versions.vlmModel !== "qwen3-vl-plus"
    || !nonEmptyString(versions.promptVersion)
    || versions.segmentationProvider !== "sam2"
    || (versions.poseModel != null && versions.poseModel !== "")
    || !nonEmptyString(versions.visualAlgorithmVersion)
    || !nonEmptyString(versions.approachPolicyVersion)
    || !sha256(versions.approachPolicySha256)
    || (versions.segmentationModel != null && typeof versions.segmentationModel !== "string")
    || (visual.observationSummary != null && typeof visual.observationSummary !== "string")
    || (visual.directionalClear != null && typeof visual.directionalClear !== "boolean")
    || (visual.rejectionReason != null && typeof visual.rejectionReason !== "string")
    || (visual.warnings != null && !Array.isArray(visual.warnings))) return null;
  const warnings: string[] = [];
  for (const rawWarning of visual.warnings ?? []) {
    if (typeof rawWarning !== "string") return null;
    warnings.push(rawWarning);
  }
  const observationSummary = typeof visual.observationSummary === "string"
    ? visual.observationSummary.trim()
    : null;
  const rejectionReason = typeof visual.rejectionReason === "string"
    ? visual.rejectionReason.trim()
    : null;
  const directionalClear = typeof visual.directionalClear === "boolean"
    ? visual.directionalClear
    : null;
  const verification = visual.verification;
  if (verification != null && !["confirmed", "not_confirmed", "unknown"].includes(String(verification))) {
    return null;
  }
  const selectedCandidateId = typeof visual.selectedCandidateId === "string"
    ? visual.selectedCandidateId.trim()
    : null;
  const selectionSourceAnalysisId = typeof visual.selectionSourceAnalysisId === "string"
    ? visual.selectionSourceAnalysisId.trim()
    : null;
  if ((visual.selectedCandidateId != null && (!selectedCandidateId || typeof visual.selectedCandidateId !== "string"))
    || (visual.selectionSourceAnalysisId != null
      && (!selectionSourceAnalysisId || typeof visual.selectionSourceAnalysisId !== "string"))
    || (selectedCandidateId === null) !== (selectionSourceAnalysisId === null)
    || (status === "needs_input" && ambiguityCandidates.length === 0)
    || (status === "succeeded" && rejectionReason !== null)
    || (["rejected", "failed"].includes(status) && !rejectionReason)
    || (mode === "arrival_verification" && status === "succeeded" && verification == null)
    || (mode !== "arrival_verification" && verification != null)
    || (mode !== "scene_description" && visual.observationSummary != null)
    || (mode !== "directional_free_space" && visual.directionalClear != null)) return null;
  if (status === "succeeded") {
    if (mode === "scene_description") {
      if (!observationSummary || geometryCandidates.length) return null;
    } else if (mode === "directional_free_space") {
      if (geometryCandidates.length !== 1
        || geometryCandidates[0].role !== "free_space"
        || directionalClear === null
        || directionalClear !== geometryCandidates[0].verified) return null;
    } else if (!geometryCandidates.length
      || geometryCandidates.some((candidate) => !candidate.verified)) return null;
  }
  return {
    analysisId,
    mode,
    status,
    source: {
      robotId: source.robotId.trim(),
      robotBootId: source.robotBootId.trim(),
      gatewayInstanceId: source.gatewayInstanceId.trim(),
      observationSessionId: source.observationSessionId.trim(),
      localizationEpochId: source.localizationEpochId.trim(),
      sourceMode: source.sourceMode as MissionVisualResult["source"]["sourceMode"],
      sourceId: source.sourceId.trim(),
      bundleId: source.bundleId.trim(),
      sensorId: source.sensorId.trim(),
      capturedAt: source.capturedAt.trim(),
      calibrationSha256: source.calibrationSha256,
      sceneId: typeof source.sceneId === "string" ? source.sceneId.trim() : null,
      sceneRevision: typeof source.sceneRevision === "number" ? source.sceneRevision : null,
    },
    versions: {
      vlmProvider: "qwen",
      vlmModel: "qwen3-vl-plus",
      promptVersion: versions.promptVersion.trim(),
      visualAlgorithmVersion: versions.visualAlgorithmVersion.trim(),
      approachPolicyVersion: versions.approachPolicyVersion.trim(),
      approachPolicySha256: versions.approachPolicySha256,
      segmentationProvider: "sam2",
      segmentationModel: typeof versions.segmentationModel === "string"
        ? versions.segmentationModel.trim() || null
        : null,
    },
    hypotheses,
    ambiguityCandidates,
    geometryCandidates,
    observationSummary,
    directionalClear,
    rejectionReason,
    evidence,
    warnings,
  };
}

type MissionVisualNode = MissionRecordV2["activeRevision"]["nodes"][number];
type MissionVisualNodeState = MissionRecordV2["nodeStates"][number];

function digest(value: unknown): value is string {
  return typeof value === "string" && /^sha256:[0-9a-f]{64}$/.test(value);
}

function expectedTerminalStatus(status: VisualTerminalStatus): string {
  return status === "needs_input" || status === "rejected" ? "rejected" : status;
}

function jsonEquivalent(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left) && Array.isArray(right)
      && left.length === right.length
      && left.every((item, index) => jsonEquivalent(item, right[index]));
  }
  const leftObject = object(left);
  const rightObject = object(right);
  if (!leftObject || !rightObject) return false;
  const leftKeys = Object.keys(leftObject).sort();
  const rightKeys = Object.keys(rightObject).sort();
  return leftKeys.length === rightKeys.length
    && leftKeys.every((key, index) => key === rightKeys[index]
      && jsonEquivalent(leftObject[key], rightObject[key]));
}

function visualTaskRequestForResult(
  mission: MissionRecordV2,
  request: Record<string, unknown>,
  input: Record<string, unknown>,
  nodeState: MissionVisualNodeState,
  parsed: MissionVisualResult,
  rawSource: Record<string, unknown>,
): Record<string, unknown> | null {
  const expectedTargetPose = input.expectedTargetPose == null
    ? null
    : mapPose(input.expectedTargetPose);
  const verificationMaximumDistanceM = input.verificationMaximumDistanceM == null
    ? 1.25
    : input.verificationMaximumDistanceM;
  const verificationMaximumYawRad = input.verificationMaximumYawRad == null
    ? 0.35
    : input.verificationMaximumYawRad;
  const targetDescription = input.targetDescription == null ? null : input.targetDescription;
  const direction = input.direction == null ? null : input.direction;
  const requestedDistanceM = input.requestedDistanceM == null ? null : input.requestedDistanceM;
  const candidateId = input.candidateId == null ? null : input.candidateId;
  const selectionSourceAnalysisId = input.selectionSourceAnalysisId == null
    ? null
    : input.selectionSourceAnalysisId;
  if ((input.expectedTargetPose != null && !expectedTargetPose)
    || (targetDescription !== null && typeof targetDescription !== "string")
    || (direction !== null && typeof direction !== "string")
    || (requestedDistanceM !== null && !finite(requestedDistanceM))
    || !finite(verificationMaximumDistanceM) || verificationMaximumDistanceM <= 0
    || verificationMaximumDistanceM > 3
    || !finite(verificationMaximumYawRad) || verificationMaximumYawRad <= 0
    || verificationMaximumYawRad > Math.PI
    || (candidateId !== null && typeof candidateId !== "string")
    || (selectionSourceAnalysisId !== null && typeof selectionSourceAnalysisId !== "string")
    || typeof rawSource.sensorId !== "string") return null;
  return {
    schemaVersion: "h-ide.visual-task/2",
    requestId: request.requestId,
    missionId: request.missionId,
    revisionId: request.revisionId,
    nodeId: request.nodeId,
    authorizationId: request.authorizationId,
    attempt: nodeState.attempt,
    traceId: request.requestId,
    mode: parsed.mode,
    instruction: input.instruction,
    targetDescription,
    direction,
    requestedDistanceM,
    expectedTargetPose,
    verificationMaximumDistanceM,
    verificationMaximumYawRad,
    robotId: request.robotId,
    robotBootId: mission.plan.envelope.robotBootId,
    localizationEpochId: mission.plan.envelope.localizationEpochId,
    robotProfileDigest: mission.plan.envelope.robotProfileDigest,
    capabilityManifestDigest: request.capabilityManifestDigest,
    sensorId: rawSource.sensorId,
    requestedSourceMode: parsed.source.sourceMode,
    sceneId: parsed.source.sceneId,
    sceneRevision: parsed.source.sceneRevision,
    candidateId,
    selectionSourceAnalysisId,
  };
}

function boundMissionVisualResult(
  mission: MissionRecordV2,
  node: MissionVisualNode,
  nodeState: MissionVisualNodeState,
  requestId: string,
): MissionVisualResult | null {
  const revisionId = mission.activeRevision.revisionId;
  const state = object(nodeState.state);
  const executions = mission.skillExecutions.filter((raw) => object(raw)?.requestId === requestId);
  if (executions.length !== 1 || !state) return null;
  const execution = object(executions[0]);
  const request = object(execution?.request);
  const input = object(request?.input);
  const terminal = object(execution?.result);
  const terminalResult = object(terminal?.result);
  const visual = object(terminalResult?.visualResult);
  const envelope = mission.plan.envelope;
  const authorization = object(mission.authorization);
  const parsed = parseMissionVisualResult(visual);
  const rawSource = object(visual?.source);
  const expectedRequestId = `skill-${mission.missionId}-${mission.activeRevision.sequence}-${node.nodeId}-${nodeState.attempt}`;
  const terminalEvent = parsed
    ? terminalEventForRequest(
      mission,
      revisionId,
      node.nodeId,
      requestId,
      expectedTerminalStatus(parsed.status),
    )
    : null;
  const visualTaskRequest = parsed && request && input && rawSource
    ? visualTaskRequestForResult(mission, request, input, nodeState, parsed, rawSource)
    : null;
  const expectedRequestSha256 = visualTaskRequest
    ? visualTaskRequestSha256(visualTaskRequest)
    : null;
  if (!execution || !request || !input || !terminal || !visual || !parsed || !rawSource
    || mission.snapshot.missionId !== mission.missionId
    || mission.snapshot.robotId !== mission.robotId
    || mission.snapshot.status !== mission.status
    || mission.snapshot.activeRevisionId !== revisionId
    || mission.plan.robotId !== mission.robotId || envelope.robotId !== mission.robotId
    || mission.plan.envelope.schemaVersion !== "h-ide.mission-envelope/2"
    || !nonEmptyString(envelope.robotBootId) || !nonEmptyString(envelope.localizationEpochId)
    || !digest(envelope.capabilityManifestDigest)
    || !digest(envelope.robotProfileDigest)
    || !digest(envelope.workstationCapabilityDigest)
    || !Number.isInteger(envelope.maximumStageAttempts)
    || envelope.maximumStageAttempts < 1 || envelope.maximumStageAttempts > 20
    || !Array.isArray(envelope.allowedSkillIds)
    || !envelope.allowedSkillIds.includes("vision.observe")
    || !Array.isArray(envelope.sceneRefs)
    || node.skillId !== "vision.observe"
    || nodeState.missionId !== mission.missionId
    || nodeState.revisionId !== revisionId || nodeState.nodeId !== node.nodeId
    || state.skillRequestId !== requestId
    || !Number.isInteger(nodeState.attempt) || nodeState.attempt < 1
    || nodeState.attempt > envelope.maximumStageAttempts || nodeState.attempt > 20
    || !Number.isInteger(mission.activeRevision.sequence) || mission.activeRevision.sequence < 1
    || requestId !== expectedRequestId
    || execution.requestId !== requestId || execution.missionId !== mission.missionId
    || execution.revisionId !== revisionId || execution.nodeId !== node.nodeId
    || request.schemaVersion !== "h-ide.skill-request/2"
    || request.requestId !== requestId || request.missionId !== mission.missionId
    || request.revisionId !== revisionId || request.nodeId !== node.nodeId
    || request.robotId !== mission.robotId || request.skillId !== "vision.observe"
    || request.skillVersion !== node.skillVersion
    || request.capabilityManifestDigest !== envelope.capabilityManifestDigest
    || !digest(request.capabilityManifestDigest)
    || !authorization || !nonEmptyString(authorization.authorizationId)
    || request.authorizationId !== authorization.authorizationId
    || request.idempotencyKey !== requestId
    || !timestamp(request.issuedAt) || !timestamp(request.expiresAt)
    || Date.parse(request.expiresAt) <= Date.parse(request.issuedAt)
    || terminal.requestId !== requestId || !timestamp(terminal.completedAt)
    || Date.parse(terminal.completedAt) < Date.parse(request.issuedAt)
    || (terminal.errorCode !== null && typeof terminal.errorCode !== "string")
    || (terminal.detail !== null && typeof terminal.detail !== "string")
    || !Array.isArray(terminal.evidenceIds)
    || terminal.evidenceIds.length > 50
    || terminal.evidenceIds.some((item) => !nonEmptyString(item))
    || !jsonEquivalent(terminal.evidenceIds, parsed.evidence.map((item) => item.sha256).slice(0, 50))
    || !terminalEvent || !timestamp(terminalEvent.occurredAt)
    || !jsonEquivalent(terminalEvent.payload, terminal)
    || visual.requestId !== requestId || visual.missionId !== mission.missionId
    || visual.revisionId !== revisionId || visual.nodeId !== node.nodeId
    || expectedRequestSha256 === null || visual.requestSha256 !== expectedRequestSha256
    || input.instruction !== node.purpose
    || visual.analysisId !== parsed.analysisId
    || visual.mode !== parsed.mode || visual.status !== parsed.status
    || visual.attempt !== nodeState.attempt || parsed.mode !== input.mode
    || rawSource.robotId !== mission.robotId
    || rawSource.robotBootId !== envelope.robotBootId
    || rawSource.localizationEpochId !== envelope.localizationEpochId) return null;

  const targetModes = ["object_approach", "landmark_observation", "arrival_verification"];
  if ((targetModes.includes(parsed.mode) && !nonEmptyString(input.targetDescription))
    || (parsed.mode === "directional_free_space" && (
      typeof input.direction !== "string"
      || !["forward", "left", "right"].includes(input.direction)
      || !finite(input.requestedDistanceM)
      || input.requestedDistanceM <= 0 || input.requestedDistanceM > 5
    ))
    || (parsed.mode !== "directional_free_space"
      && (input.direction != null || input.requestedDistanceM != null))) return null;

  const attemptText = requestId.slice(requestId.lastIndexOf("-") + 1);
  const requestAttempt = Number(attemptText);
  const terminalStatus = expectedTerminalStatus(parsed.status);
  const expectedErrorCode = parsed.status === "succeeded"
    ? null
    : `VISUAL_${parsed.status.toUpperCase()}`;
  if (!/^[1-9][0-9]*$/.test(attemptText)
    || !Number.isSafeInteger(requestAttempt) || requestAttempt !== nodeState.attempt
    || execution.status !== terminalStatus || terminal.status !== terminalStatus
    || state.resultStatus !== terminalStatus
    || nodeState.status !== (terminalStatus === "succeeded" ? "succeeded" : terminalStatus)
    || (expectedErrorCode === null
      ? terminal.errorCode != null
      : terminal.errorCode !== expectedErrorCode)) return null;

  const expectedCandidateId = input.candidateId;
  const expectedAnalysisId = input.selectionSourceAnalysisId;
  if ((expectedCandidateId == null) !== (expectedAnalysisId == null)
    || (expectedCandidateId == null
      ? visual.selectedCandidateId != null || visual.selectionSourceAnalysisId != null
      : !nonEmptyString(expectedCandidateId) || !nonEmptyString(expectedAnalysisId)
        || visual.selectedCandidateId !== expectedCandidateId
        || visual.selectionSourceAnalysisId !== expectedAnalysisId)
    || (expectedCandidateId != null && nodeState.attempt < 2)) return null;

  const sensorId = input.sensorId;
  if (sensorId != null && sensorId !== parsed.source.sensorId) return null;
  const sceneRef = envelope.sceneRefs[0];
  if (sceneRef) {
    if (parsed.source.sceneId !== sceneRef.sceneId
      || parsed.source.sceneRevision === null
      || String(parsed.source.sceneRevision) !== String(sceneRef.revision)) return null;
  } else if (parsed.source.sceneId !== null || parsed.source.sceneRevision !== null) return null;
  return parsed;
}

type MissionVisualEvent = MissionRecordV2["events"][number];

function terminalEventForRequest(
  mission: MissionRecordV2,
  revisionId: string,
  nodeId: string,
  requestId: string,
  terminalStatus?: string,
): MissionVisualEvent | null {
  const events = mission.events.filter((event) => (
    event.correlationId === requestId
    && ["skill.succeeded", "skill.failed"].includes(event.eventType)
  ));
  if (events.length !== 1) return null;
  const [event] = events;
  const expectedEventType = terminalStatus === undefined
    ? null
    : terminalStatus === "succeeded" ? "skill.succeeded" : "skill.failed";
  return event.missionId === mission.missionId
    && event.revisionId === revisionId
    && event.nodeId === nodeId
    && (expectedEventType === null || event.eventType === expectedEventType)
    && Number.isInteger(event.sequence) && event.sequence > 0
    && event.sequence <= mission.snapshot.lastSequence
    ? event
    : null;
}

function terminalEventSequence(
  mission: MissionRecordV2,
  revisionId: string,
  nodeId: string,
  requestId: string,
): number | null {
  return terminalEventForRequest(mission, revisionId, nodeId, requestId)?.sequence ?? null;
}

/** Resolve the most recent terminal visual pointer without trusting projection order. */
export function latestMissionVisualResult(mission: MissionRecordV2): MissionVisualResult | null {
  const revisionId = mission.snapshot.activeRevisionId;
  if (typeof revisionId !== "string" || !revisionId
    || revisionId !== mission.activeRevision.revisionId) return null;
  const visualNodes = mission.activeRevision.nodes.filter((node) => node.skillId === "vision.observe");
  const activeRequestId = mission.snapshot.activeSkillRequestId;
  if (activeRequestId != null) {
    if (typeof activeRequestId !== "string" || !activeRequestId
      || activeRequestId !== activeRequestId.trim()) return null;
    const activeNodes = mission.activeRevision.nodes.filter(
      (node) => node.nodeId === mission.snapshot.activeNodeId,
    );
    const activeExecutions = mission.skillExecutions.filter(
      (raw) => object(raw)?.requestId === activeRequestId,
    );
    const activeExecution = activeExecutions.length === 1 ? object(activeExecutions[0]) : null;
    const activeRequest = object(activeExecution?.request);
    if (activeNodes.length !== 1 || !activeExecution || !activeRequest
      || activeExecution.missionId !== mission.missionId
      || activeExecution.revisionId !== revisionId
      || activeExecution.nodeId !== activeNodes[0].nodeId
      || activeRequest.requestId !== activeRequestId
      || activeRequest.missionId !== mission.missionId
      || activeRequest.revisionId !== revisionId
      || activeRequest.nodeId !== activeNodes[0].nodeId
      || activeRequest.robotId !== mission.robotId
      || activeRequest.skillId !== activeNodes[0].skillId
      || activeRequest.skillVersion !== activeNodes[0].skillVersion) return null;
    if (activeNodes[0].skillId === "vision.observe") return null;
  }

  const pointers: Array<{
    node: MissionVisualNode;
    state: MissionVisualNodeState;
    requestId: string;
    sequence: number;
  }> = [];
  for (const node of visualNodes) {
    const states = mission.nodeStates.filter((state) => (
      state.revisionId === revisionId
      && state.nodeId === node.nodeId
    ));
    if (states.length !== 1 || states[0].missionId !== mission.missionId) return null;
    const state = states[0];
    const payload = object(state.state);
    if (!payload) return null;
    const requestId = payload?.skillRequestId;
    if (requestId == null) {
      if (state.attempt !== 0) return null;
      continue;
    }
    if (typeof requestId !== "string" || !requestId || requestId !== requestId.trim()) return null;
    const sequence = terminalEventSequence(mission, revisionId, node.nodeId, requestId);
    if (sequence === null) return null;
    pointers.push({ node, state, requestId, sequence });
  }
  if (!pointers.length) return null;
  const maximumSequence = Math.max(...pointers.map((item) => item.sequence));
  const latest = pointers.filter((item) => item.sequence === maximumSequence);
  if (latest.length !== 1) return null;
  return boundMissionVisualResult(
    mission,
    latest[0].node,
    latest[0].state,
    latest[0].requestId,
  );
}

/**
 * Resolve only the ambiguity result named by the active rejected node state.
 *
 * `skillExecutions` is a read projection whose ordering is not an execution
 * identity. Candidate buttons therefore follow the durable `skillRequestId`
 * pointer and independently verify every Mission/revision/node/result binding.
 */
export function currentMissionVisualAmbiguity(
  mission: MissionRecordV2,
): MissionVisualResult | null {
  const { snapshot } = mission;
  const revisionId = snapshot.activeRevisionId;
  const nodeId = snapshot.activeNodeId;
  if (mission.status !== "needs_input"
    || snapshot.missionId !== mission.missionId
    || snapshot.status !== "needs_input" || snapshot.errorCode !== "VISUAL_NEEDS_INPUT"
    || snapshot.activeSkillRequestId !== null
    || typeof revisionId !== "string" || !revisionId || revisionId !== revisionId.trim()
    || typeof nodeId !== "string" || !nodeId || nodeId !== nodeId.trim()
    || mission.activeRevision.revisionId !== revisionId) return null;
  const activeNodes = mission.activeRevision.nodes.filter((node) => node.nodeId === nodeId);
  if (activeNodes.length !== 1 || activeNodes[0].skillId !== "vision.observe") return null;

  const nodeStates = mission.nodeStates.filter((item) => (
    item.revisionId === revisionId && item.nodeId === nodeId
  ));
  if (nodeStates.length !== 1 || nodeStates[0].missionId !== mission.missionId
    || nodeStates[0].status !== "rejected") return null;
  const nodeState = nodeStates[0];
  const nodeStatePayload = object(nodeState.state);
  const requestId = nodeStatePayload?.skillRequestId;
  if (typeof requestId !== "string" || !requestId.trim()
    || requestId !== requestId.trim()
    || !Number.isInteger(nodeState.attempt) || nodeState.attempt < 1) return null;

  const parsed = boundMissionVisualResult(mission, activeNodes[0], nodeState, requestId);
  return parsed?.status === "needs_input"
    && parsed.rejectionReason === "ambiguous_target"
    && parsed.ambiguityCandidates.length
    ? parsed
    : null;
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
          ? typeof clearance?.measured === "number" && typeof clearance.threshold === "number"
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
      {geometry?.safetyGates.length ? <section className="evidence-gates"><strong>确定性安全门</strong><ul>{geometry.safetyGates.map((gate) => <li key={gate.name} className={gate.passed ? "passed" : "failed"}><span>{gate.passed ? "通过" : "拒绝"}</span>{gate.name}{gate.measured != null ? ` · 实测 ${formatSafetyValue(gate.measured)}` : ""}{gate.threshold != null ? ` / 阈值 ${formatSafetyValue(gate.threshold)}` : ""}</li>)}</ul></section> : null}
      <p className="evidence-boundary">{sceneDescription
        ? "黄色框是当前画面的语义观察，不代表可导航坐标。该模式不会发出底盘动作。"
        : directional
          ? "方向净空由深度图确定性计算，不要求识别语义目标；净空不足会跳过导航。任何位移和原地转向仍必须经过 Robot Agent 与 Nav2。"
          : "黄色框是语义假设；三维坐标由深度、相机标定和 TF 确定性计算。视觉结果不包含路径或速度，任何位移和原地转向都必须经过 Robot Agent 与 Nav2。"}</p>
    </aside>
  </div>;
}

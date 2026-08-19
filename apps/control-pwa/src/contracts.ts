export type RobotStatus = "online" | "offline" | "degraded" | "unknown" | "estop";
export type SafetyMode = "normal" | "degraded" | "stopped" | "estop" | "unknown";
export type CommandStatus =
  | "pending"
  | "accepted"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "expired"
  | "rejected"
  | "unknown";

export interface RobotState {
  robotId: string;
  name?: string;
  status: RobotStatus;
  connected: boolean;
  safetyMode: SafetyMode;
  battery?: number;
  taskState?: string;
  pose?: PoseMap;
  details?: Record<string, unknown>;
}

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D extends Point2D {
  z: number;
}

export interface PoseMap extends Point3D {
  frameId: "map";
  yaw: number;
  observedAt?: string | null;
}

export interface Bounds3D {
  min: Point3D;
  max: Point3D;
}

export interface Calibration {
  mode: "ros_map_identity" | "interactive_similarity";
  TMapFromScene: number[][];
  TSceneFromMap: number[][];
  scaleMetersPerUnit: number;
  residualMeters: number;
}

export interface SemanticRegion {
  locationId: string;
  name: string;
  aliases: string[];
  polygonXY: Point2D[];
  zMin: number;
  zMax: number;
  entryPose: PoseMap;
}

export interface SceneRevision {
  sceneId: string;
  revision: number;
  status: "draft" | "published";
  baseRevision?: number | null;
  importJobId?: string | null;
  artifactsAvailable?: boolean;
  previewAvailable?: boolean;
  processingSummary?: SceneProcessingSummary | null;
  calibration?: Calibration | null;
  annotations: SemanticRegion[];
  boundsMap?: Bounds3D | null;
  createdAt: string;
  publishedAt?: string | null;
  mapJob?: SceneJob | null;
  mapSummary?: SceneMapSummary | null;
}

export interface SceneJob {
  jobId: string;
  jobKind?: "import" | "map_derivation";
  revision?: number | null;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled" | "superseded";
  attempts: number;
  errorCode?: string | null;
  message?: string | null;
  previewAvailable?: boolean;
  processingSummary?: SceneProcessingSummary | null;
}

export interface SceneProcessingSummary {
  sourcePointCount?: number;
  analysisInputPointCount?: number;
  finitePointCount?: number;
  visualPointCount?: number;
  medianSpacing?: number;
  voxelSize?: number;
  decimationStep?: number;
  warnings?: string[];
}

export interface SceneMapSummary {
  schemaVersion: "h-ide.scene-map/1";
  mapVersion: string;
  frameId: "scene" | "map";
  verified: boolean;
  status: "ready" | "degraded";
  resolution: number;
  groundZ: number;
  observedCoverageRatio: number;
  confidence: "low" | "medium" | "high";
  warnings: string[];
}

export interface SceneMapMetadata {
  schemaVersion: "h-ide.scene-map/1";
  mapVersion: string;
  frameId: "scene" | "map";
  verified: boolean;
  status: "ready" | "degraded";
  resolution: number;
  origin: Point2D;
  width: number;
  height: number;
  imageRowOrder: "negative_y";
  groundZ: number;
  summary: SceneMapSummary;
}

export interface SceneSummary {
  sceneId: string;
  sourceFileName?: string;
  rawAssetId: string;
  sourceAvailable: boolean;
  artifactsAvailable?: boolean;
  previewAvailable?: boolean;
  processingSummary?: SceneProcessingSummary | null;
  job?: SceneJob | null;
  revisions: SceneRevision[];
  createdAt: string;
}

export interface SceneUploadSession {
  uploadId: string;
  status: string;
  offset: number;
  sizeBytes: number;
  chunkSize: number;
  expiresAt: string;
}

export interface SceneUploadCompleted {
  sceneId: string;
  assetId: string;
  jobId: string;
  sha256: string;
  status: string;
  dataEncoding: "ascii" | "binary" | "binary_compressed";
}

export interface CapabilitySnapshot {
  robotId: string;
  agent: string;
  protocol: string;
  capabilities: string[];
  safetyMode: SafetyMode;
  schema?: Record<string, unknown>;
}

export interface CommandReceipt {
  commandId: string;
  robotId?: string;
  status: CommandStatus;
  acceptedAt?: string;
  startedAt?: string | null;
  completedAt?: string | null;
  result?: unknown;
  error?: string | null;
  safetyStop?: boolean;
  safetyWarning?: string | null;
}

export interface MappingSnapshot {
  mapping: Record<string, unknown>;
  mappingId: string;
  version: string;
  commandKey: string;
  status: "draft" | "validated" | "probed" | "active" | "superseded";
  targetId?: string | null;
  probe?: { status?: string; result?: unknown; error?: string | null } | null;
  updatedAt?: string;
}

export interface PlanStep {
  stepId: string;
  commandKey: string;
  payload: Record<string, unknown>;
  dependsOn?: string[];
}

export interface ActionPlanDraft {
  planType?: "action";
  planId?: string;
  draftId?: string;
  robotId: string;
  steps: PlanStep[];
  expiresAt?: string;
}

export interface VisualMissionStage {
  stageId: string;
  kind: "locate" | "approach" | "verify" | "complete";
  targetDescription: string;
  goalPurpose: "observe" | "approach" | "final";
}

export interface VisualMissionSpec {
  missionId: string;
  robotId: string;
  instruction: string;
  locale: string;
  targetDescription: string;
  requestedSourceMode: "live" | "replay" | "synthetic";
  fusionMode: "vision_only" | "scene_only" | "fused";
  stages: VisualMissionStage[];
  robotBootId: string;
  localizationEpochId: string;
  sensorIds: string[];
  allowedTargetCategories: string[];
  unsupportedCapabilities: string[];
  sceneRef?: { sceneId: string; revision: string | number } | null;
  maximumGoalDistanceM: number;
  maximumCumulativeDistanceM: number;
  maximumDurationSeconds: number;
  maximumSpeedMps: number;
  vlmProvider: string;
  vlmModel: string;
  promptVersion: string;
  visualAlgorithmVersion: string;
  approachPolicyVersion: string;
  approachPolicySha256: string;
  reviewExpiresAt: string;
}

export interface VisualMissionPlanDraft {
  planType: "visual_mission";
  planId: string;
  robotId: string;
  mission: VisualMissionSpec;
  createdAt?: string;
  expiresAt?: string;
}

export type PlanDraft = ActionPlanDraft | VisualMissionPlanDraft;

export type MissionStatus =
  | "draft"
  | "approved"
  | "ready"
  | "planning"
  | "observing"
  | "dispatching"
  | "executing"
  | "verifying"
  | "needs_input"
  | "awaiting_review"
  | "suspended"
  | "completed"
  | "failed"
  | "cancelled";

export interface MissionEnvelopeV2 {
  schemaVersion: "h-ide.mission-envelope/2";
  robotId: string;
  robotBootId: string;
  localizationEpochId: string;
  capabilityManifestDigest: string;
  robotProfileDigest: string;
  workstationCapabilityDigest: string;
  allowedSkillIds: string[];
  maximumRiskLevel: "low" | "medium" | "high" | "critical";
  maximumGoalDistanceM: number;
  maximumCumulativeDistanceM: number;
  maximumSpeedMps: number;
  maximumDurationSeconds: number;
  maximumAutoRevisions: number;
  maximumStageAttempts: number;
  maximumScanYawRad: number;
  sceneRefs: Array<{ sceneId: string; revision: string | number }>;
  expiresAt: string;
}

export interface MissionNodeV2 {
  nodeId: string;
  skillId: string;
  skillVersion: string;
  purpose: string;
  intent: Record<string, unknown>;
  dependsOn: string[];
  preconditionIds: string[];
  postconditionIds: string[];
  riskLevel: "low" | "medium" | "high" | "critical";
}

export interface MissionPlanDraftV2 {
  planType: "mission";
  schemaVersion: "h-ide.mission/2";
  planId: string;
  robotId: string;
  objective: string;
  nodes: MissionNodeV2[];
  envelope: MissionEnvelopeV2;
  createdAt: string;
  expiresAt: string;
}

export interface MissionDraftRecordV2 {
  draftId: string;
  robotId: string;
  status: string;
  plan: MissionPlanDraftV2;
  error?: string | null;
  createdAt?: string;
  updatedAt?: string;
}

export interface MissionSnapshotV2 {
  missionId: string;
  robotId: string;
  status: MissionStatus;
  lastSequence: number;
  activeRevisionId?: string | null;
  activeNodeId?: string | null;
  activeSkillRequestId?: string | null;
  autoRevisionCount: number;
  updatedAt: string;
  errorCode?: string | null;
  detail?: string | null;
}

export interface MissionEventV2 {
  eventId: string;
  missionId: string;
  sequence: number;
  eventType: string;
  occurredAt: string;
  correlationId: string;
  revisionId?: string | null;
  nodeId?: string | null;
  payload: Record<string, unknown>;
}

export interface MissionWorldRecordV2 {
  recordId: string;
  missionId: string;
  robotId: string;
  kind: "hypothesis" | "fact";
  subject: string;
  predicate: string;
  value: unknown;
  source:
    | "qwen"
    | "mujoco_oracle"
    | "geometry"
    | "robot_agent"
    | "scene"
    | "operator"
    | "verifier";
  sourceRef: string;
  confidence: number;
  observedAt: string;
  evidenceIds: string[];
}

export interface MissionNodeStateV2 {
  missionId: string;
  revisionId: string;
  nodeId: string;
  status: string;
  attempt: number;
  state: Record<string, unknown>;
  updatedAt: string;
}

export interface MissionRecordV2 {
  missionId: string;
  robotId: string;
  status: MissionStatus;
  plan: MissionPlanDraftV2;
  authorization: Record<string, unknown>;
  activeRevision: {
    revisionId: string;
    sequence: number;
    objective: string;
    nodes: MissionNodeV2[];
    disposition: "human_approved" | "policy_inherited" | "human_required";
    changeSummary: string[];
    createdAt: string;
  };
  revisions: Array<Record<string, unknown>>;
  snapshot: MissionSnapshotV2;
  events: MissionEventV2[];
  worldRecords: MissionWorldRecordV2[];
  nodeStates: MissionNodeStateV2[];
  skillExecutions: Array<Record<string, unknown>>;
  error?: string | null;
}

export interface MissionAdvanceResultV2 {
  mission: MissionRecordV2;
  actions: Array<{
    kind: string;
    nodeId?: string;
    skillId?: string;
    requestId?: string;
    dispatch?: string;
    errorCode?: string;
    detail?: string;
  }>;
  stoppedBecause: MissionStatus;
}

export type SimulationBackend = "deterministic" | "mujoco_sonic" | "gazebo" | "isaac";
export type SimulationMode = "oracle" | "qwen";
export type SimulationStatus = "created" | "starting" | "running" | "paused" | "stopped" | "failed";
export type SimulationNavigationState = "idle" | "moving" | "blocked" | "arrived" | "failed";

export interface SimulationPolicy {
  schemaVersion: "h-ide.simulation-policy/1";
  manualRevisionApproval: boolean;
  allowAutoRevision: boolean;
  allowPhysicalExecution: false;
  maximumAutoRevisions: number;
  maximumGoalDistanceM: number;
  maximumCumulativeDistanceM: number;
  maximumDurationSeconds: number;
}

export interface SimulationSession {
  schemaVersion: "h-ide.simulation-session/1";
  sessionId: string;
  backend: SimulationBackend;
  executionProfile: "simulation";
  sourceMode: "synthetic" | "replay";
  robotId: string;
  robotBootId: string;
  localizationEpochId: string;
  missionId?: string | null;
  sceneRef?: { sceneId: string; revision: string | number } | null;
  mode: SimulationMode;
  status: SimulationStatus;
  releaseState: "simulation_only" | "validated" | "supervised_auto";
  policy: SimulationPolicy;
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface SimulationObstacle {
  obstacleId: string;
  label: string;
  center: Point2D;
  radiusM: number;
  static: boolean;
}

export interface SimulationLandmark {
  landmarkId: string;
  label: string;
  pose: PoseMap;
  aliases: string[];
}

export interface SimulationSnapshot {
  schemaVersion: "h-ide.simulation-snapshot/1";
  sessionId: string;
  status: SimulationStatus;
  navigationState: SimulationNavigationState;
  pose: PoseMap;
  goalPose?: PoseMap | null;
  trajectory: PoseMap[];
  obstacles: SimulationObstacle[];
  landmarks: SimulationLandmark[];
  plannedWaypoints: PoseMap[];
  activeMissionId?: string | null;
  activeRevisionId?: string | null;
  pendingReview?: SimulationReview | null;
  lastEventSequence: number;
  updatedAt: string;
}

export interface SimulationReview {
  schemaVersion: "h-ide.simulation-review/1";
  status: "pending" | "resolved" | "error";
  triggerEventId: string;
  triggerEventType: string;
  missionId?: string | null;
  decisionKind: "replan" | "needs_review" | "pause" | "continue" | "error";
  revisionId?: string | null;
  detail: string;
  createdAt: string;
  updatedAt: string;
}

export interface SimulationEvent {
  schemaVersion: "h-ide.simulation-event/1";
  eventId: string;
  sessionId: string;
  sequence: number;
  eventType: string;
  source: "operator" | "simulator" | "robot_agent" | "nav2" | "oracle" | "qwen";
  payload: Record<string, unknown>;
  occurredAt: string;
}

export interface SimulationRecord {
  session: SimulationSession;
  snapshot: SimulationSnapshot;
}

export interface SimulationReceipt {
  schemaVersion: "h-ide.simulation-receipt/1";
  receiptId: string;
  sessionId: string;
  commandId: string;
  status: "accepted" | "running" | "succeeded" | "blocked" | "failed" | "cancelled";
  executionBackend: string;
  motion: "simulated" | "mujoco";
  goalPose?: PoseMap | null;
  detail?: string | null;
  createdAt: string;
}

export interface SimulationLive {
  backend: "mujoco_sonic";
  ready: boolean;
  health: { telemetry?: Record<string, unknown>; plant?: Record<string, unknown> };
  state: Record<string, unknown>;
  telemetry: Record<string, unknown>;
  navigation?: { status?: string; commandId?: string; detail?: string | null; result?: Record<string, unknown> } | null;
  pose?: PoseMap | null;
  rgbUrl?: string;
  depthUrl?: string;
}

export interface VisualMissionStageState {
  stageId: string;
  attempt: number;
  status: string;
  visualResolutionId?: string | null;
  commandId?: string | null;
  state?: {
    explanation?: string;
    targetPhrase?: string;
    landmarkPose?: { frameId: string; x: number; y: number; z: number; yaw: number } | null;
    goalPose?: { frameId: string; x: number; y: number; z: number; yaw: number } | null;
    safetyGates?: Array<{ name: string; passed: boolean; measured?: number | string | null; threshold?: number | string | null; detail?: string | null }>;
    ambiguityCandidates?: Array<{ candidateId: string; label: string; box: [number, number, number, number]; score?: number | null }>;
    evidence?: Array<{ kind: string; sha256: string; mediaType: string; expiresAt?: string }>;
    source?: Record<string, unknown> | null;
    versions?: Record<string, unknown> | null;
    receipt?: CommandReceipt;
  };
  updatedAt?: string;
}

export interface VisualMissionRecord {
  missionId: string;
  draftId: string;
  robotId: string;
  status: string;
  plan: VisualMissionPlanDraft;
  state: {
    currentStageIndex?: number;
    cancelRequested?: boolean;
    receipts?: CommandReceipt[];
    candidateSelection?: {
      stageId: string;
      candidateId: string;
      selectionSourceVisualResolutionId: string;
    };
  };
  stages: VisualMissionStageState[];
  mode: "off" | "observe" | "lab_execute" | "robot_execute";
  releaseState: string;
  motion: "simulated" | "not_executed";
  error?: string | null;
}

export function isVisualMissionPlan(plan: PlanDraft): plan is VisualMissionPlanDraft {
  return plan.planType === "visual_mission";
}

export interface PlanReviewStep extends PlanStep {
  rationale: string;
  contributedBy: string;
}

export interface WorkerContribution {
  agentId: string;
  contribution: string;
  summary: string;
  steps: PlanStep[];
  assumptions: string[];
  warnings: string[];
}

export interface PlanReview {
  robotId: string;
  summary: string;
  steps: PlanReviewStep[];
  assumptions: string[];
  warnings: string[];
  contributions: WorkerContribution[];
  contextVersion: string;
  spatialChecks?: PlanSpatialCheck[];
}

export interface PlanSpatialCheck {
  stepId: string;
  goalPose: PoseMap;
  check: {
    mapVersion: string;
    classification: "observed_free" | "occupied" | "unknown" | "outside";
    clearanceMeters: number;
    distanceToObservedObstacleMeters?: number | null;
    warnings: string[];
  };
}

export interface AssistantDraft {
  draftId: string;
  status: string;
  plan: PlanDraft;
  review?: PlanReview;
  revisionOfDraftId?: string | null;
  error?: string | null;
}

export interface AssistantMessage {
  messageId: string;
  taskId?: string | null;
  role: "user" | "assistant" | "system";
  content: string;
  modality: "text" | "voice";
  transcriptionId?: string | null;
  createdAt: string;
}

export interface AssistantTurn {
  taskId: string;
  conversationId: string;
  status: string;
  result?: { status?: string; assistantMessage?: string; draftId?: string } | null;
  error?: string | null;
  cancellationRequested?: boolean;
  draft?: AssistantDraft;
}

export interface AssistantConversation {
  conversationId: string;
  robotId: string;
  requestedBy: string;
  locale: string;
  status: string;
  createdAt?: string;
  updatedAt?: string;
  messages?: AssistantMessage[];
  turns?: AssistantTurn[];
  drafts?: AssistantDraft[];
}

export interface SpeechTranscription {
  transcriptionId: string;
  text: string;
  language?: string | null;
  locale: string;
  createdAt: string;
}

export interface ControlEvent {
  eventId?: number;
  commandId?: string;
  robotId?: string;
  status?: string;
  mappingVersion?: string;
  traceId?: string;
  at?: string;
  createdAt?: string;
  error?: string | null;
  [key: string]: unknown;
}

export interface ApiError {
  detail?: unknown;
}

export function statusLabel(status: string | undefined): string {
  const labels: Record<string, string> = {
    idle: "空闲",
    low: "低风险",
    medium: "中风险",
    high: "高风险",
    critical: "严重风险",
    online: "在线",
    offline: "离线",
    degraded: "降级",
    unknown: "未知",
    estop: "急停",
    normal: "正常",
    stopped: "已停止",
    pending: "排队",
    accepted: "已接收",
    running: "执行中",
    succeeded: "成功",
    skipped: "已跳过（条件不满足）",
    failed: "失败",
    cancelled: "已取消",
    expired: "已过期",
    rejected: "已拒绝",
    draft: "草稿",
    validated: "已校验",
    probed: "已探测",
    active: "启用",
    superseded: "已替代",
    planning: "规划中",
    review: "待审核",
    awaiting_user: "等待补充",
    draft_ready: "草案就绪",
    needs_input: "需要补充",
    cannot_plan: "无法规划",
    revision_requested: "修改中",
    confirmed: "已确认",
    confirming: "确认中",
    completed: "已完成",
    approved: "已审核",
    ready: "准备执行",
    observing: "视觉观察中",
    resolving: "视觉解析中",
    dispatching: "发送目标中",
    executing: "执行中",
    verifying: "验证中",
    awaiting_review: "等待审核",
    suspended: "已暂停",
  };
  return labels[status ?? ""] ?? status ?? "未知";
}

export function isAssistantTurnTerminal(status: string | undefined): boolean {
  return ["needs_input", "draft_ready", "cannot_plan", "failed", "cancelled", "unknown"].includes(status ?? "");
}

export function isTerminal(status: string | undefined): boolean {
  return ["succeeded", "failed", "cancelled", "expired", "rejected"].includes(status ?? "");
}

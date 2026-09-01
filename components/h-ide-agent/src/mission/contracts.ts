import { isDeepStrictEqual } from "node:util";
import { z } from "zod";

const id = z.string().trim().min(1).max(200);
const shortText = z.string().trim().min(1).max(4_000);
const isoDateTime = z.string().datetime({ offset: true });
// Capability/profile digests use an algorithm-qualified wire value so they
// cannot be confused with evidence hashes that are plain hexadecimal.
const sha256 = z.string().regex(/^sha256:[0-9a-f]{64}$/);

// Keep planner-owned JSON opaque without emitting propertyNames or
// patternProperties. DashScope rejects those keywords in response schemas.
export const jsonObjectSchema = z.looseObject({});

export const missionStatusSchema = z.enum([
  "draft",
  "approved",
  "ready",
  "planning",
  "observing",
  "dispatching",
  "executing",
  "verifying",
  "needs_input",
  "awaiting_review",
  "suspended",
  "completed",
  "failed",
  "cancelled",
]);

export const approvalDispositionSchema = z.enum([
  "human_approved",
  "policy_inherited",
  "human_required",
]);

export const interventionKindSchema = z.enum([
  "pause",
  "resume",
  "cancel",
  "modify_goal",
  "append_requirement",
  "answer",
]);

export const skillResultStatusSchema = z.enum([
  "succeeded",
  "failed",
  "cancelled",
  "rejected",
  "timed_out",
  "unknown",
]);

export const worldRecordKindSchema = z.enum(["hypothesis", "fact"]);
export const worldRecordSourceSchema = z.enum([
  "qwen",
  "geometry",
  "robot_agent",
  "scene",
  "operator",
  "verifier",
]);

export const riskLevelSchema = z.enum(["low", "medium", "high", "critical"]);
const riskOrder: Record<z.infer<typeof riskLevelSchema>, number> = {
  low: 0,
  medium: 1,
  high: 2,
  critical: 3,
};

export const sceneRefV2Schema = z.object({
  sceneId: id,
  revision: z.union([id, z.number().int().nonnegative()]),
}).strict();

export const point2dV2Schema = z.object({
  x: z.number().finite(),
  y: z.number().finite(),
}).strict();

export const geofenceSchema = z.object({
  geofenceId: id,
  frameId: z.literal("map").default("map"),
  polygonXY: z.array(point2dV2Schema).min(3).max(1_000),
}).strict().superRefine((value, context) => {
  if (isDeepStrictEqual(value.polygonXY[0], value.polygonXY.at(-1))) {
    context.addIssue({ code: "custom", path: ["polygonXY"], message: "geofence must not repeat its first vertex" });
  }
  const unique = new Set(value.polygonXY.map((point) => `${point.x}:${point.y}`));
  if (unique.size < 3) {
    context.addIssue({ code: "custom", path: ["polygonXY"], message: "geofence requires three distinct vertices" });
  }
});

export const modelPinSchema = z.object({
  role: z.enum(["text_planner", "visual_grounder", "segmentation", "geometry"]),
  provider: id,
  model: id,
  promptVersion: id.nullable().optional(),
}).strict();

export const missionEnvelopeSchema = z.object({
  schemaVersion: z.literal("h-ide.mission-envelope/2"),
  robotId: id,
  robotBootId: id,
  localizationEpochId: id,
  capabilityManifestDigest: sha256,
  robotProfileDigest: sha256,
  workstationCapabilityDigest: sha256,
  allowedSkillIds: z.array(id).min(1).max(100),
  maximumRiskLevel: riskLevelSchema.default("medium"),
  allowedTargetCategories: z.array(id).max(100).default([]),
  allowedRegionIds: z.array(id).max(500).default([]),
  sceneRefs: z.array(sceneRefV2Schema).max(8).default([]),
  geofences: z.array(geofenceSchema).max(16).default([]),
  maximumGoalDistanceM: z.number().finite().positive().max(25).default(5),
  maximumCumulativeDistanceM: z.number().finite().positive().max(200).default(20),
  maximumSpeedMps: z.number().finite().positive().max(1).default(0.2),
  maximumDurationSeconds: z.number().int().positive().max(7_200).default(600),
  maximumAutoRevisions: z.number().int().min(0).max(20).default(3),
  maximumStageAttempts: z.number().int().min(1).max(10).default(3),
  // Cumulative Mission budget. Individual rotate/scan actions remain bounded
  // by their capability schema and the downstream policy gate, while a
  // multi-station inspection may legitimately consume many full scans.
  maximumScanYawRad: z.number().finite().positive().max(40 * Math.PI).default(2 * Math.PI),
  modelPins: z.array(modelPinSchema).min(2).max(16),
  expiresAt: isoDateTime,
}).strict().superRefine((value, context) => {
  const unique = <T>(items: T[]) => new Set(items).size === items.length;
  for (const field of ["allowedSkillIds", "allowedTargetCategories", "allowedRegionIds"] as const) {
    if (!unique(value[field])) {
      context.addIssue({ code: "custom", path: [field], message: `${field} must be unique` });
    }
  }
  const sceneIds = value.sceneRefs.map((scene) => `${scene.sceneId}:${scene.revision}`);
  if (!unique(sceneIds)) {
    context.addIssue({ code: "custom", path: ["sceneRefs"], message: "sceneRefs must be unique" });
  }
  const pinRoles = value.modelPins.map((pin) => pin.role);
  if (!unique(pinRoles)) {
    context.addIssue({ code: "custom", path: ["modelPins"], message: "modelPins must be unique" });
  }
  for (const required of ["text_planner", "visual_grounder"] as const) {
    if (!pinRoles.includes(required)) {
      context.addIssue({ code: "custom", path: ["modelPins"], message: `modelPins missing required role: ${required}` });
    }
  }
  if (new Date(value.expiresAt).getTime() <= Date.now()) {
    context.addIssue({ code: "custom", path: ["expiresAt"], message: "mission envelope has expired" });
  }
});

const executableIntentKeys = new Set([
  "angularvelocity", "cmdvel", "executor", "frameid", "goalpose", "linearvelocity",
  "path", "pose", "position", "quaternion", "rosaction", "rosservice", "waypoints",
  "x", "y", "yaw", "z",
]);

function rejectExecutableIntent(value: unknown, path = "intent") {
  if (Array.isArray(value)) {
    value.forEach((item, index) => rejectExecutableIntent(item, `${path}[${index}]`));
    return;
  }
  if (!value || typeof value !== "object") return;
  for (const [key, item] of Object.entries(value)) {
    const normalized = [...key.toLowerCase()].filter((character) => /[a-z0-9]/u.test(character)).join("");
    if (executableIntentKeys.has(normalized)) {
      throw new Error(`${path}.${key} is executable geometry, not semantic intent`);
    }
    rejectExecutableIntent(item, `${path}.${key}`);
  }
}

export const placeTargetSchema = z.object({
  schemaVersion: z.literal("h-ide.place-target/1"),
  locationId: id,
  locationName: id,
  anchorRole: z.enum(["entry", "interior", "observation_vantage"]),
  stationId: id,
  targetRecordId: id,
  candidateTargetRecordIds: z.array(id).min(1).max(20),
}).strict().superRefine((value, context) => {
  if (new Set(value.candidateTargetRecordIds).size !== value.candidateTargetRecordIds.length) {
    context.addIssue({ code: "custom", path: ["candidateTargetRecordIds"], message: "place candidates must be unique" });
  }
  if (value.candidateTargetRecordIds[0] !== value.targetRecordId) {
    context.addIssue({ code: "custom", path: ["targetRecordId"], message: "selected place target must be the first candidate" });
  }
});

export const placeCompletionSchema = z.object({
  schemaVersion: z.literal("h-ide.place-completion/1"),
  predicate: z.enum(["at_anchor", "inside_region"]),
  regionId: id,
  minimumClearanceM: z.number().finite().min(0).max(2),
  maximumAnchorDistanceM: z.number().finite().positive().max(2),
}).strict();

export const placeNavigationIntentSchema = z.object({
  schemaVersion: z.literal("h-ide.place-intent/1"),
  task: z.enum(["navigate_to_place", "verify_inside_region"]),
  place: placeTargetSchema,
  completion: placeCompletionSchema,
}).strict().superRefine((value, context) => {
  if (value.place.locationId !== value.completion.regionId) {
    context.addIssue({ code: "custom", path: ["completion", "regionId"], message: "place completion region mismatch" });
  }
  if (value.completion.predicate === "inside_region" && value.place.anchorRole === "entry") {
    context.addIssue({ code: "custom", path: ["place", "anchorRole"], message: "inside completion requires an interior anchor" });
  }
});

export const missionNodeSchema = z.object({
  nodeId: id,
  skillId: id,
  skillVersion: id,
  purpose: shortText,
  intent: jsonObjectSchema.default({}),
  dependsOn: z.array(id).max(32).default([]),
  preconditionIds: z.array(id).max(32).default([]),
  postconditionIds: z.array(id).max(32).default([]),
  riskLevel: riskLevelSchema,
}).strict().superRefine((value, context) => {
  try {
    rejectExecutableIntent(value.intent);
  } catch (error) {
    context.addIssue({ code: "custom", path: ["intent"], message: error instanceof Error ? error.message : String(error) });
  }
  if (value.skillId === "navigation.navigate_to_pose") {
    for (const retired of ["target_region", "target_description", "entry_pose"] as const) {
      if (retired in value.intent) {
        context.addIssue({ code: "custom", path: ["intent", retired], message: `${retired} is retired; use a typed place target or targetRecordId` });
      }
    }
  }
  if ("place" in value.intent || value.intent.schemaVersion === "h-ide.place-intent/1") {
    const typed = placeNavigationIntentSchema.safeParse(value.intent);
    if (!typed.success) {
      context.addIssue({ code: "custom", path: ["intent"], message: `invalid typed place intent: ${typed.error.issues[0]?.message ?? "unknown error"}` });
    }
  }
  for (const field of ["dependsOn", "preconditionIds", "postconditionIds"] as const) {
    if (new Set(value[field]).size !== value[field].length) {
      context.addIssue({ code: "custom", path: [field], message: `${field} must be unique` });
    }
  }
});

function validateDag(nodes: Array<{ nodeId: string; dependsOn: string[] }>, context: z.RefinementCtx) {
  const ids = nodes.map((node) => node.nodeId);
  if (new Set(ids).size !== ids.length) {
    context.addIssue({ code: "custom", path: ["nodes"], message: "mission node IDs must be unique" });
    return;
  }
  const known = new Set(ids);
  const graph = new Map(nodes.map((node) => [node.nodeId, node.dependsOn]));
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const visit = (nodeId: string): boolean => {
    if (visiting.has(nodeId)) return false;
    if (visited.has(nodeId)) return true;
    visiting.add(nodeId);
    for (const dependency of graph.get(nodeId) ?? []) {
      if (!known.has(dependency) || !visit(dependency)) return false;
    }
    visiting.delete(nodeId);
    visited.add(nodeId);
    return true;
  };
  for (const nodeId of ids) {
    if (!visit(nodeId)) {
      context.addIssue({ code: "custom", path: ["nodes"], message: "mission dependency cycle or unknown dependency detected" });
      return;
    }
  }
}

export const missionPlanDraftSchema = z.object({
  planType: z.literal("mission").default("mission"),
  schemaVersion: z.literal("h-ide.mission/2"),
  planId: id,
  robotId: id,
  objective: shortText,
  nodes: z.array(missionNodeSchema).min(1).max(100),
  envelope: missionEnvelopeSchema,
  createdAt: isoDateTime,
  expiresAt: isoDateTime,
}).strict().superRefine((value, context) => {
  validateDag(value.nodes, context);
  if (value.robotId !== value.envelope.robotId) {
    context.addIssue({ code: "custom", path: ["robotId"], message: "mission envelope robotId mismatch" });
  }
  const created = new Date(value.createdAt).getTime();
  const expires = new Date(value.expiresAt).getTime();
  const envelopeExpires = new Date(value.envelope.expiresAt).getTime();
  if (!(expires > created && expires > Date.now())) {
    context.addIssue({ code: "custom", path: ["expiresAt"], message: "mission draft has expired or invalid lifetime" });
  }
  if (expires > envelopeExpires) {
    context.addIssue({ code: "custom", path: ["expiresAt"], message: "mission draft cannot outlive its envelope" });
  }
  const allowed = new Set(value.envelope.allowedSkillIds);
  for (const node of value.nodes) {
    if (!allowed.has(node.skillId)) {
      context.addIssue({ code: "custom", path: ["nodes"], message: `mission uses skill outside envelope: ${node.skillId}` });
    }
    if (riskOrder[node.riskLevel] > riskOrder[value.envelope.maximumRiskLevel]) {
      context.addIssue({ code: "custom", path: ["nodes"], message: `mission node exceeds envelope risk: ${node.nodeId}` });
    }
  }
});

export const missionAuthorizationSchema = z.object({
  authorizationId: id,
  missionId: id,
  planId: id,
  envelopeSha256: sha256,
  disposition: z.literal("human_approved").default("human_approved"),
  approvedBy: id,
  approvedAt: isoDateTime,
  expiresAt: isoDateTime,
}).strict().superRefine((value, context) => {
  if (new Date(value.expiresAt).getTime() <= new Date(value.approvedAt).getTime()
    || new Date(value.expiresAt).getTime() <= Date.now()) {
    context.addIssue({ code: "custom", path: ["expiresAt"], message: "mission authorization has expired" });
  }
});

export const planRevisionSchema = z.object({
  schemaVersion: z.literal("h-ide.plan-revision/2"),
  revisionId: id,
  missionId: id,
  sequence: z.number().int().min(1),
  parentRevisionId: id.nullable().optional(),
  objective: shortText,
  nodes: z.array(missionNodeSchema).min(1).max(100),
  reasonCode: id,
  reason: shortText,
  changeSummary: z.array(id).min(1).max(50),
  disposition: approvalDispositionSchema,
  authorizationId: id.nullable().optional(),
  createdBy: id,
  createdAt: isoDateTime,
}).strict().superRefine((value, context) => {
  validateDag(value.nodes, context);
  if (value.disposition === "human_required" && value.authorizationId) {
    context.addIssue({ code: "custom", path: ["authorizationId"], message: "human-required revision cannot have authorization" });
  }
  if (value.disposition !== "human_required" && !value.authorizationId) {
    context.addIssue({ code: "custom", path: ["authorizationId"], message: "approved revision requires authorization" });
  }
});

export const worldRecordSchema = z.object({
  schemaVersion: z.literal("h-ide.world-record/2"),
  recordId: id,
  missionId: id,
  robotId: id,
  kind: worldRecordKindSchema,
  subject: id,
  predicate: id,
  value: z.unknown(),
  source: worldRecordSourceSchema,
  sourceRef: id,
  confidence: z.number().finite().min(0).max(1),
  observedAt: isoDateTime,
  validUntil: isoDateTime.nullable().optional(),
  frameId: id.nullable().optional(),
  sceneRef: sceneRefV2Schema.nullable().optional(),
  evidenceIds: z.array(id).max(50).default([]),
  supersedesRecordId: id.nullable().optional(),
}).strict().superRefine((value, context) => {
  if (value.kind === "fact" && value.source === "qwen") {
    context.addIssue({ code: "custom", path: ["kind"], message: "Qwen output cannot be promoted directly to a fact" });
  }
  if (value.validUntil && new Date(value.validUntil).getTime() <= new Date(value.observedAt).getTime()) {
    context.addIssue({ code: "custom", path: ["validUntil"], message: "validUntil must be later than observedAt" });
  }
  if (new Set(value.evidenceIds).size !== value.evidenceIds.length) {
    context.addIssue({ code: "custom", path: ["evidenceIds"], message: "evidenceIds must be unique" });
  }
});

export const skillRequestSchema = z.object({
  schemaVersion: z.literal("h-ide.skill-request/2"),
  requestId: id,
  missionId: id,
  revisionId: id,
  nodeId: id,
  robotId: id,
  skillId: id,
  skillVersion: id,
  input: jsonObjectSchema.default({}),
  capabilityManifestDigest: sha256,
  authorizationId: id,
  idempotencyKey: id,
  issuedAt: isoDateTime,
  expiresAt: isoDateTime,
}).strict().superRefine((value, context) => {
  if (new Date(value.expiresAt).getTime() <= new Date(value.issuedAt).getTime()
    || new Date(value.expiresAt).getTime() <= Date.now()) {
    context.addIssue({ code: "custom", path: ["expiresAt"], message: "skill request has expired" });
  }
});

export const skillProgressSchema = z.object({
  requestId: id,
  sequence: z.number().int().min(0),
  progress: z.number().finite().min(0).max(1).nullable().optional(),
  phase: id,
  observedAt: isoDateTime,
  detail: z.string().nullable().optional(),
}).strict();

export const skillResultSchema = z.object({
  requestId: id,
  status: skillResultStatusSchema,
  completedAt: isoDateTime,
  result: jsonObjectSchema.default({}),
  evidenceIds: z.array(id).max(50).default([]),
  errorCode: id.nullable().optional(),
  detail: z.string().nullable().optional(),
}).strict();

export const missionSnapshotSchema = z.object({
  missionId: id,
  robotId: id,
  status: missionStatusSchema,
  lastSequence: z.number().int().min(0),
  activeRevisionId: id.nullable().optional(),
  activeNodeId: id.nullable().optional(),
  activeSkillRequestId: id.nullable().optional(),
  autoRevisionCount: z.number().int().min(0).default(0),
  cumulativeDistanceM: z.number().finite().min(0).default(0),
  cumulativeScanYawRad: z.number().finite().min(0).default(0),
  updatedAt: isoDateTime,
  errorCode: id.nullable().optional(),
  detail: z.string().nullable().optional(),
}).strict();

export const missionCapabilitySchema = z.object({
  skillId: id,
  skillVersion: id,
  description: shortText,
  kind: z.enum(["motion", "observation", "manipulation", "safety", "non_motion"]),
  riskLevel: riskLevelSchema,
  executor: z.enum(["nav2", "behavior_tree", "ros_action", "ros_service", "local"]),
  executorProfile: id,
  // The executor is the orchestration boundary, while baseMotionOwner is the
  // authority boundary. A workstation composite (for example the reviewed
  // Office V2 station scan) may be dispatched by a local behavior tree while
  // its physical navigation/rotation children remain owned by Nav2.
  baseMotionOwner: z.literal("nav2").optional(),
  requiredSensorIds: z.array(id).max(16).default([]),
}).strict();

export const missionBriefSchema = z.object({
  requestId: id,
  missionId: id,
  robotId: id,
  requestedBy: id,
  objective: shortText,
  locale: z.string().trim().min(2).max(35),
  modality: z.enum(["text", "voice"]),
  contextVersion: id,
  envelope: missionEnvelopeSchema,
  capabilities: z.array(missionCapabilitySchema).min(1).max(256),
  history: z.array(z.object({ role: z.enum(["user", "assistant", "system"]), content: shortText, createdAt: isoDateTime }).strict()).max(20).default([]),
  worldRecords: z.array(worldRecordSchema).max(500).default([]),
}).strict().superRefine((value, context) => {
  if (value.robotId !== value.envelope.robotId) {
    context.addIssue({ code: "custom", path: ["robotId"], message: "mission envelope robotId mismatch" });
  }
  const capabilityIds = value.capabilities.map((item) => item.skillId);
  if (new Set(capabilityIds).size !== capabilityIds.length) {
    context.addIssue({ code: "custom", path: ["capabilities"], message: "mission capability IDs must be unique" });
  }
  if (
    capabilityIds.length !== value.envelope.allowedSkillIds.length
    || capabilityIds.some((skillId) => !value.envelope.allowedSkillIds.includes(skillId))
  ) {
    context.addIssue({ code: "custom", path: ["capabilities"], message: "mission capabilities do not match envelope allowedSkillIds" });
  }
  for (const record of value.worldRecords) {
    if (record.missionId !== value.missionId || record.robotId !== value.robotId) {
      context.addIssue({ code: "custom", path: ["worldRecords"], message: "world record mission identity mismatch" });
      break;
    }
  }
});

export const missionDraftModelOutputSchema = z.object({
  objective: shortText,
  nodes: z.array(missionNodeSchema).min(1).max(100),
  assumptions: z.array(shortText).max(20).default([]),
  warnings: z.array(shortText).max(20).default([]),
}).strict();

export const missionReplanRequestSchema = z.object({
  requestId: id,
  missionId: id,
  robotId: id,
  requestedBy: id,
  objective: shortText,
  contextVersion: id,
  envelope: missionEnvelopeSchema,
  capabilities: z.array(missionCapabilitySchema).min(1).max(256),
  activeRevision: planRevisionSchema,
  worldRecords: z.array(worldRecordSchema).max(500).default([]),
  trigger: z.enum([
    "observation_updated",
    "skill_failed",
    "verification_failed",
    "safety_degraded",
    "goal_changed",
    "operator_intervention",
    "timeout",
  ]),
  triggerDetail: shortText,
  candidateId: id.nullable().optional(),
  selectionSourceAnalysisId: id.nullable().optional(),
  autoRevisionCount: z.number().int().min(0).default(0),
  authorizationId: id.nullable().optional(),
}).strict().superRefine((value, context) => {
  if (value.robotId !== value.envelope.robotId || value.activeRevision.missionId !== value.missionId) {
    context.addIssue({ code: "custom", path: ["missionId"], message: "replan mission identity mismatch" });
  }
  const versions = new Map(value.capabilities.map((item) => [item.skillId, item.skillVersion]));
  if (
    value.capabilities.length !== value.envelope.allowedSkillIds.length
    || value.envelope.allowedSkillIds.some((skillId) => !versions.has(skillId))
  ) {
    context.addIssue({ code: "custom", path: ["capabilities"], message: "replan capabilities do not match envelope" });
  }
  if (value.activeRevision.nodes.some((node) => versions.get(node.skillId) !== node.skillVersion)) {
    context.addIssue({ code: "custom", path: ["activeRevision"], message: "active revision skill version is stale" });
  }
  if (value.activeRevision.nodes.some((node) => !value.envelope.allowedSkillIds.includes(node.skillId))) {
    context.addIssue({ code: "custom", path: ["activeRevision"], message: "active revision uses a skill outside envelope" });
  }
  if (value.activeRevision.nodes.some((node) => riskOrder[node.riskLevel] > riskOrder[value.envelope.maximumRiskLevel])) {
    context.addIssue({ code: "custom", path: ["activeRevision"], message: "active revision exceeds envelope risk" });
  }
  for (const record of value.worldRecords) {
    if (record.missionId !== value.missionId || record.robotId !== value.robotId) {
      context.addIssue({ code: "custom", path: ["worldRecords"], message: "world record mission identity mismatch" });
      break;
    }
  }
  if (value.autoRevisionCount > value.envelope.maximumAutoRevisions) {
    context.addIssue({ code: "custom", path: ["autoRevisionCount"], message: "maximum automatic revisions exceeded" });
  }
  const hasCandidateId = value.candidateId != null;
  const hasSelectionSource = value.selectionSourceAnalysisId != null;
  if (hasCandidateId !== hasSelectionSource) {
    context.addIssue({
      code: "custom",
      path: [hasCandidateId ? "selectionSourceAnalysisId" : "candidateId"],
      message: "operator candidate selection binding is incomplete",
    });
  } else if (hasCandidateId && value.trigger !== "operator_intervention") {
    context.addIssue({
      code: "custom",
      path: ["candidateId"],
      message: "operator candidate selection requires an operator intervention trigger",
    });
  }
});

export const missionRevisionModelOutputSchema = z.object({
  objective: shortText,
  nodes: z.array(missionNodeSchema).min(1).max(100),
  reasonCode: id,
  reason: shortText,
  changeSummary: z.array(id).min(1).max(50),
}).strict();

export const missionEventTriggerSchema = z.object({
  missionId: id,
  robotId: id,
  eventType: z.enum([
    "observation.updated",
    "skill.progress",
    "skill.result",
    "verification.failed",
    "safety.changed",
    "operator.intervention",
    "mission.timeout",
  ]),
  payload: jsonObjectSchema.default({}),
  correlationId: id,
  occurredAt: isoDateTime,
  activeRevision: planRevisionSchema.nullable().optional(),
  autoRevisionCount: z.number().int().min(0).default(0),
}).strict().superRefine((value, context) => {
  if (value.activeRevision && value.activeRevision.missionId !== value.missionId) {
    context.addIssue({ code: "custom", path: ["activeRevision"], message: "active revision mission identity mismatch" });
  }
});

export const missionEventContextSchema = z.object({
  envelope: missionEnvelopeSchema,
  capabilities: z.array(missionCapabilitySchema).min(1).max(256),
  objective: shortText,
  requestedBy: id,
  contextVersion: id,
  worldRecords: z.array(worldRecordSchema).max(500).default([]),
}).strict().superRefine((value, context) => {
  const capabilityIds = value.capabilities.map((item) => item.skillId);
  if (
    new Set(capabilityIds).size !== capabilityIds.length
    || capabilityIds.length !== value.envelope.allowedSkillIds.length
    || capabilityIds.some((skillId) => !value.envelope.allowedSkillIds.includes(skillId))
  ) {
    context.addIssue({ code: "custom", path: ["capabilities"], message: "event capabilities do not match envelope" });
  }
});

export const missionEventEvaluationRequestSchema = z.object({
  event: missionEventTriggerSchema,
  context: missionEventContextSchema,
}).strict();

export const missionDecisionSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("continue"), reason: shortText }).strict(),
  z.object({ kind: z.literal("pause"), reason: shortText }).strict(),
  z.object({ kind: z.literal("needs_review"), reason: shortText }).strict(),
  z.object({
    kind: z.literal("replan"),
    request: missionReplanRequestSchema,
    revision: planRevisionSchema,
  }).strict(),
]);

export const missionCandidateSchema = z.object({
  candidateId: id,
  skillId: id,
  skillVersion: id,
  purpose: shortText,
  intent: jsonObjectSchema.default({}),
  riskLevel: riskLevelSchema,
  modelScore: z.number().finite().min(0).max(1).nullable().optional(),
  evidenceRecordIds: z.array(id).max(50).default([]),
}).strict().superRefine((value, context) => {
  try {
    rejectExecutableIntent(value.intent, "candidate.intent");
  } catch (error) {
    context.addIssue({ code: "custom", path: ["intent"], message: error instanceof Error ? error.message : String(error) });
  }
});

export const candidateScoreRequestSchema = z.object({
  missionId: id,
  robotId: id,
  envelope: missionEnvelopeSchema,
  candidates: z.array(missionCandidateSchema).min(1).max(50),
  worldRecords: z.array(worldRecordSchema).max(500).default([]),
}).strict().superRefine((value, context) => {
  if (value.robotId !== value.envelope.robotId) {
    context.addIssue({ code: "custom", path: ["robotId"], message: "candidate score robot identity mismatch" });
  }
  for (const record of value.worldRecords) {
    if (record.missionId !== value.missionId || record.robotId !== value.robotId) {
      context.addIssue({ code: "custom", path: ["worldRecords"], message: "world record mission identity mismatch" });
      break;
    }
  }
});

export const candidateScoreSchema = z.object({
  candidateId: id,
  score: z.number().finite().min(0).max(1),
  eligible: z.boolean(),
  reasons: z.array(shortText).max(20),
}).strict();

export const candidateScoreResultSchema = z.object({
  missionId: id,
  scores: z.array(candidateScoreSchema).min(1).max(50),
  selectedCandidateId: id.nullable(),
  generatedAt: isoDateTime,
}).strict().superRefine((value, context) => {
  const ids = value.scores.map((item) => item.candidateId);
  if (new Set(ids).size !== ids.length) {
    context.addIssue({ code: "custom", path: ["scores"], message: "candidate IDs must be unique" });
  }
  if (value.selectedCandidateId && !ids.includes(value.selectedCandidateId)) {
    context.addIssue({ code: "custom", path: ["selectedCandidateId"], message: "selected candidate is missing" });
  }
  if (value.selectedCandidateId && !value.scores.find((item) => item.candidateId === value.selectedCandidateId)?.eligible) {
    context.addIssue({ code: "custom", path: ["selectedCandidateId"], message: "selected candidate is not eligible" });
  }
});

export type MissionEnvelope = z.infer<typeof missionEnvelopeSchema>;
export type MissionNode = z.infer<typeof missionNodeSchema>;
export type MissionPlanDraft = z.infer<typeof missionPlanDraftSchema>;
export type MissionAuthorization = z.infer<typeof missionAuthorizationSchema>;
export type PlanRevision = z.infer<typeof planRevisionSchema>;
export type WorldRecord = z.infer<typeof worldRecordSchema>;
export type SkillRequest = z.infer<typeof skillRequestSchema>;
export type SkillProgress = z.infer<typeof skillProgressSchema>;
export type SkillResult = z.infer<typeof skillResultSchema>;
export type MissionSnapshot = z.infer<typeof missionSnapshotSchema>;
export type MissionBrief = z.infer<typeof missionBriefSchema>;
export type MissionDraftModelOutput = z.infer<typeof missionDraftModelOutputSchema>;
export type MissionReplanRequest = z.infer<typeof missionReplanRequestSchema>;
export type MissionRevisionModelOutput = z.infer<typeof missionRevisionModelOutputSchema>;
export type MissionEventTrigger = z.infer<typeof missionEventTriggerSchema>;
export type MissionEventContext = z.infer<typeof missionEventContextSchema>;
export type MissionDecision = z.infer<typeof missionDecisionSchema>;
export type MissionCandidate = z.infer<typeof missionCandidateSchema>;
export type CandidateScoreRequest = z.infer<typeof candidateScoreRequestSchema>;
export type CandidateScoreResult = z.infer<typeof candidateScoreResultSchema>;

export function riskRank(level: z.infer<typeof riskLevelSchema>): number {
  return riskOrder[level];
}

export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value as Record<string, unknown>).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key])}`).join(",")}}`;
}

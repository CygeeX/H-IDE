import { isDeepStrictEqual } from "node:util";
import { z } from "zod";

const id = z.string().trim().min(1);
const isoDateTime = z.string().datetime({ offset: true });
// Keep opaque JSON envelopes open at runtime without emitting `propertyNames`.
// DashScope rejects that keyword in response_format JSON schemas, while a
// loose object preserves the same JSON-object boundary and arbitrary values.
const jsonObject = z.looseObject({});

export const point2dSchema = z.object({ x: z.number().finite(), y: z.number().finite() }).strict();
export const point3dSchema = point2dSchema.extend({ z: z.number().finite() }).strict();
export const poseMapSchema = point3dSchema.extend({
  frameId: z.literal("map"),
  yaw: z.number().finite(),
  observedAt: isoDateTime.nullable().optional(),
}).strict();
export const bounds3dSchema = z.object({ min: point3dSchema, max: point3dSchema }).strict()
  .superRefine((value, context) => {
    if (value.min.x > value.max.x || value.min.y > value.max.y || value.min.z > value.max.z) {
      context.addIssue({ code: "custom", message: "scene bounds min/max are not ordered" });
    }
  });
export const sceneRefSchema = z.object({
  sceneId: id,
  revision: z.union([id, z.number().int().nonnegative()]),
}).strict();
export const semanticRegionSchema = z.object({
  locationId: id,
  name: id,
  aliases: z.array(id).max(50),
  parentLocationId: id.nullable().optional(),
  anchorRoles: z.array(z.enum([
    "region",
    "entry",
    "interior",
    "observation_vantage",
  ])).min(1).max(4).default(["region"]),
  purpose: id.nullable().optional(),
  scanHeadingsRad: z.array(z.number().finite()).max(16).default([]),
  polygonXY: z.array(point2dSchema).min(3).max(1_000),
  zMin: z.number().finite(),
  zMax: z.number().finite(),
  entryPose: poseMapSchema,
}).strict();
export const sceneMapSummarySchema = z.object({
  schemaVersion: z.literal("h-ide.scene-map/1"),
  mapVersion: id,
  frameId: z.enum(["scene", "map"]),
  verified: z.boolean(),
  status: z.enum(["ready", "degraded"]),
  resolution: z.number().finite().positive(),
  groundZ: z.number().finite(),
  observedCoverageRatio: z.number().finite().min(0).max(1),
  confidence: z.enum(["low", "medium", "high"]),
  warnings: z.array(id).max(10),
}).strict();
export const sceneContextSchema = z.object({
  sceneRef: sceneRefSchema,
  bounds: bounds3dSchema,
  annotations: z.array(semanticRegionSchema).max(500),
  mapSummary: sceneMapSummarySchema.nullable().optional(),
  currentPose: poseMapSchema.nullable().optional(),
  selectedPoint: point3dSchema.nullable().optional(),
}).strict();
export const spatialIntentSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("relative"),
    forwardMeters: z.number().finite(),
    lateralMeters: z.number().finite(),
    yawDeltaRad: z.number().finite(),
  }).strict(),
  z.object({ kind: z.literal("semantic"), locationId: id }).strict(),
  z.object({
    kind: z.literal("selected_point"),
    pointMap: point3dSchema,
    yaw: z.number().finite().nullable().optional(),
  }).strict(),
]);
export const spatialResolutionSchema = z.object({
  resolutionId: id,
  sceneRef: sceneRefSchema,
  intent: spatialIntentSchema,
  sourcePose: poseMapSchema.nullable().optional(),
  goalPose: poseMapSchema,
  resolverVersion: z.literal("scene-core/1"),
  warnings: z.array(id).max(20),
  computedAt: isoDateTime,
}).strict();
export const navigationPayloadSchema = z.object({
  goalPose: poseMapSchema,
  spatialResolution: spatialResolutionSchema,
}).strict().superRefine((value, context) => {
  if (!isDeepStrictEqual(value.goalPose, value.spatialResolution.goalPose)) {
    context.addIssue({ code: "custom", path: ["goalPose"], message: "resolved goal mismatch" });
  }
});

export const visualMissionStageSchema = z
  .object({
    stageId: id,
    kind: z.enum(["locate", "approach", "verify", "complete"]),
    targetDescription: id,
    goalPurpose: z.enum(["observe", "approach", "final"]).default("approach"),
  })
  .strict();

export const visualMissionSpecSchema = z
  .object({
    missionId: id,
    robotId: id,
    instruction: z.string().trim().min(1).max(4_000),
    locale: z.string().trim().min(2).max(35).default("zh-CN"),
    targetDescription: id,
    requestedSourceMode: z.enum(["live", "replay", "synthetic"]),
    fusionMode: z.enum(["vision_only", "scene_only", "fused"]).default("vision_only"),
    stages: z.array(visualMissionStageSchema).min(1).max(6),
    robotBootId: id,
    localizationEpochId: id,
    sensorIds: z.array(id).min(1).max(4),
    allowedTargetCategories: z.array(id).min(1).max(20),
    unsupportedCapabilities: z.array(id).max(20).default([]),
    sceneRef: sceneRefSchema.nullable().optional(),
    maximumGoalDistanceM: z.number().finite().positive().max(5).default(5),
    maximumCumulativeDistanceM: z.number().finite().positive().max(20).default(20),
    maximumDurationSeconds: z.number().int().positive().max(600).default(600),
    vlmProvider: id,
    vlmModel: id,
    promptVersion: id,
    visualAlgorithmVersion: id,
    approachPolicyVersion: id,
    approachPolicySha256: z.string().regex(/^[0-9a-f]{64}$/),
    maximumSpeedMps: z.number().finite().positive().max(0.5).default(0.2),
    reviewExpiresAt: isoDateTime,
  })
  .strict()
  .superRefine((value, context) => {
    const stageIds = value.stages.map((stage) => stage.stageId);
    if (new Set(stageIds).size !== stageIds.length) {
      context.addIssue({ code: "custom", path: ["stages"], message: "visual mission stage IDs must be unique" });
    }
    if (value.fusionMode !== "vision_only" && value.sceneRef == null) {
      context.addIssue({ code: "custom", path: ["sceneRef"], message: "scene fusion requires sceneRef" });
    }
  });

export const visualMissionIntentSchema = z
  .object({
    targetDescription: id,
    goalPurpose: z.enum(["approach", "final"]),
    fusionMode: z.enum(["vision_only", "fused"]),
    targetCategory: id,
    summary: z.string().trim().min(1).max(1_000),
    warnings: z.array(z.string().trim().min(1).max(500)).max(10).default([]),
  })
  .strict();

export const ambiguityCandidateSchema = z
  .object({
    candidateId: id,
    label: id,
    box: z.tuple([z.number().int(), z.number().int(), z.number().int(), z.number().int()]),
    score: z.number().finite().nullable().optional(),
  })
  .strict();

const resolutionPoseSchema = z
  .object({
    frameId: id,
    x: z.number().finite(),
    y: z.number().finite(),
    z: z.number().finite(),
    yaw: z.number().finite(),
  })
  .strict();

export const visualResolutionSchema = z
  .object({
    visualResolutionId: id,
    missionId: id,
    stageId: id,
    attempt: z.number().int().positive(),
    traceId: id,
    status: z.enum(["queued", "resolving", "succeeded", "needs_input", "rejected", "cancelled", "failed"]),
    executable: z.boolean(),
    targetPhrase: id,
    source: z.object({
      robotId: id,
      robotBootId: id,
      gatewayInstanceId: id,
      sensorId: id,
      observationSessionId: id,
      localizationEpochId: id,
      sourceMode: z.enum(["live", "replay", "synthetic"]),
      sourceId: id,
      bundleId: id,
      capturedAt: isoDateTime,
      calibrationSha256: z.string().regex(/^[0-9a-f]{64}$/),
      sceneId: id.nullable().optional(),
      sceneRevision: z.number().int().positive().nullable().optional(),
    }).strict(),
    versions: z.object({
      vlmProvider: id,
      vlmModel: id,
      promptVersion: id,
      visualAlgorithmVersion: id,
      approachPolicyVersion: id,
      approachPolicySha256: z.string().regex(/^[0-9a-f]{64}$/),
      segmentationModel: id.nullable().optional(),
      poseModel: id.nullable().optional(),
    }).strict(),
    landmarkPose: resolutionPoseSchema.nullable().optional(),
    goalPose: resolutionPoseSchema.nullable().optional(),
    goalPurpose: z.enum(["observe", "approach", "final"]),
    safetyGates: z.array(z.object({
      name: id,
      passed: z.boolean(),
      measured: z.union([z.number(), z.string(), z.boolean()]).nullable().optional(),
      threshold: z.union([z.number(), z.string(), z.boolean()]).nullable().optional(),
      detail: z.string().nullable().optional(),
    }).strict()),
    ambiguityCandidates: z.array(ambiguityCandidateSchema),
    selectedCandidateId: id.nullable().optional(),
    selectionSourceVisualResolutionId: id.nullable().optional(),
    warnings: z.array(z.string()),
    rejectionReason: z.string().nullable().optional(),
    evidence: z.array(z.object({
      kind: id,
      sha256: z.string().regex(/^[0-9a-f]{64}$/),
      mediaType: id,
      expiresAt: isoDateTime.nullable().optional(),
    }).strict()),
    createdAt: isoDateTime,
  })
  .strict();

export const visualStageRequestSchema = z
  .object({
    mission: visualMissionSpecSchema,
    stage: visualMissionStageSchema,
    attempt: z.number().int().positive(),
    candidateId: id.nullable().optional(),
    selectionSourceVisualResolutionId: id.nullable().optional(),
  })
  .strict()
  .superRefine((value, context) => {
    if (!value.mission.stages.some((stage) => isDeepStrictEqual(stage, value.stage))) {
      context.addIssue({ code: "custom", path: ["stage"], message: "stage is not pinned by mission" });
    }
    const selectionComplete = value.candidateId != null
      && value.selectionSourceVisualResolutionId != null;
    if ((value.candidateId != null || value.selectionSourceVisualResolutionId != null) && !selectionComplete) {
      context.addIssue({ code: "custom", path: ["candidateId"], message: "candidate selection binding is incomplete" });
    }
    if (selectionComplete && value.attempt < 2) {
      context.addIssue({ code: "custom", path: ["attempt"], message: "candidate selection requires a new attempt" });
    }
  });

export const visualStageResultSchema = z
  .object({
    status: z.enum(["succeeded", "needs_input", "rejected", "cancelled", "failed"]),
    visualResolutionId: id,
    explanation: z.string().trim().min(1).max(2_000),
    ambiguityCandidates: z.array(ambiguityCandidateSchema).max(20).default([]),
  })
  .strict();

export const taskRequestSchema = z
  .object({
    schemaVersion: z.literal("1.0"),
    taskId: id,
    conversationId: id,
    robotId: id,
    requestedBy: id,
    modality: z.enum(["text", "voice"]),
    text: z.string().trim().min(1).max(4_000),
    locale: z.string().trim().min(2).max(35),
    transcriptionId: id.nullable().optional(),
    revisionOfDraftId: id.nullable().optional(),
    sceneContext: sceneContextSchema.nullable().optional(),
    createdAt: isoDateTime,
  })
  .strict();

export const historyMessageSchema = z
  .object({
    role: z.enum(["user", "assistant", "system"]),
    content: z.string().trim().min(1).max(4_000),
    createdAt: isoDateTime,
  })
  .strict();

export const robotStateSchema = z
  .object({
    robotId: id,
    status: z.enum(["online", "offline", "degraded", "unknown", "estop"]),
    connected: z.boolean(),
    lastSeenAt: isoDateTime.nullable().optional(),
    safetyMode: z.enum(["normal", "degraded", "stopped", "estop", "unknown"]),
    pose: poseMapSchema.optional(),
    details: jsonObject,
  })
  .strict();

export const capabilitySnapshotSchema = z
  .object({
    agent: id,
    protocol: id,
    schema: jsonObject,
    capabilities: z.array(id).max(100),
    safetyMode: z.enum(["normal", "degraded", "stopped", "estop", "unknown"]),
    robotId: id.nullable().optional(),
  })
  .strict();

export const logicalCommandSchema = z
  .object({
    commandKey: id,
    version: id,
    kind: z.enum(["motion", "safety", "non_motion"]),
    description: z.string().trim().min(1).nullable().optional(),
    payloadSchema: jsonObject,
  })
  .strict();

export const planningEnvelopeSchema = z
  .object({
    taskRequest: taskRequestSchema,
    contextVersion: id,
    history: z.array(historyMessageSchema).max(20),
    robotState: robotStateSchema,
    capabilitySnapshot: capabilitySnapshotSchema,
    commandCatalog: z.array(logicalCommandSchema).min(1).max(100),
    sceneContext: sceneContextSchema.nullable().optional(),
  })
  .strict()
  .superRefine((value, context) => {
    const robotId = value.taskRequest.robotId;
    if (value.robotState.robotId !== robotId) {
      context.addIssue({ code: "custom", path: ["robotState", "robotId"], message: "robotId mismatch" });
    }
    if (value.capabilitySnapshot.robotId != null && value.capabilitySnapshot.robotId !== robotId) {
      context.addIssue({
        code: "custom",
        path: ["capabilitySnapshot", "robotId"],
        message: "robotId mismatch",
      });
    }
    if (!isDeepStrictEqual(value.sceneContext ?? null, value.taskRequest.sceneContext ?? null)) {
      context.addIssue({ code: "custom", path: ["sceneContext"], message: "scene context mismatch" });
    }
    if (
      value.sceneContext != null &&
      !isDeepStrictEqual(value.sceneContext.currentPose ?? null, value.robotState.pose ?? null)
    ) {
      context.addIssue({ code: "custom", path: ["sceneContext", "currentPose"], message: "robot pose mismatch" });
    }
  });

export const fragmentStepSchema = z
  .object({
    stepId: id,
    commandKey: id,
    payload: jsonObject,
    dependsOn: z.array(id).max(10),
    rationale: z.string().trim().min(1).max(1_000),
  })
  .strict();

export const planFragmentSchema = z
  .object({
    agentId: id,
    contribution: id,
    summary: z.string().trim().min(1).max(2_000),
    steps: z.array(fragmentStepSchema).min(1).max(10),
    assumptions: z.array(z.string().trim().min(1)).max(20),
    warnings: z.array(z.string().trim().min(1)).max(20),
  })
  .strict()
  .superRefine((value, context) => {
    const ids = value.steps.map((step) => step.stepId);
    if (new Set(ids).size !== ids.length) {
      context.addIssue({ code: "custom", path: ["steps"], message: "fragment step IDs must be unique" });
    }
  });

export const proposalStepSchema = fragmentStepSchema.extend({ contributedBy: id }).strict();

function validateDag(
  steps: Array<{ stepId: string; dependsOn: string[] }>,
  context: z.RefinementCtx,
) {
  const known = new Set(steps.map((step) => step.stepId));
  if (known.size !== steps.length) {
    context.addIssue({ code: "custom", path: ["steps"], message: "plan step IDs must be unique" });
    return;
  }
  const dependencies = new Map(steps.map((step) => [step.stepId, step.dependsOn]));
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const visit = (stepId: string): boolean => {
    if (visiting.has(stepId)) return false;
    if (visited.has(stepId)) return true;
    visiting.add(stepId);
    for (const dependency of dependencies.get(stepId) ?? []) {
      if (!known.has(dependency)) {
        context.addIssue({
          code: "custom",
          path: ["steps"],
          message: `unknown plan dependency: ${dependency}`,
        });
        return false;
      }
      if (!visit(dependency)) return false;
    }
    visiting.delete(stepId);
    visited.add(stepId);
    return true;
  };
  for (const step of steps) {
    if (!visit(step.stepId)) {
      context.addIssue({ code: "custom", path: ["steps"], message: "plan dependency cycle detected" });
      return;
    }
  }
}

export const planProposalSchema = z
  .object({
    robotId: id,
    summary: z.string().trim().min(1).max(2_000),
    steps: z.array(proposalStepSchema).max(10).default([]),
    assumptions: z.array(z.string().trim().min(1)).max(20),
    warnings: z.array(z.string().trim().min(1)).max(20),
    contributions: z.array(planFragmentSchema).max(8).default([]),
    contextVersion: id,
  })
  .strict()
  .superRefine((value, context) => {
    if (value.steps.length === 0 && value.contributions.length === 0) return;
    if (value.steps.length === 0 || value.contributions.length === 0) {
      context.addIssue({
        code: "custom",
        path: ["steps"],
        message: "action plan steps and contributions must be supplied together",
      });
      return;
    }
    validateDag(value.steps, context);
    const agentIds = value.contributions.map((fragment) => fragment.agentId);
    if (new Set(agentIds).size !== agentIds.length) {
      context.addIssue({ code: "custom", path: ["contributions"], message: "agent IDs must be unique" });
    }
    const knownAgents = new Set(agentIds);
    for (const [index, step] of value.steps.entries()) {
      if (!knownAgents.has(step.contributedBy)) {
        context.addIssue({
          code: "custom",
          path: ["steps", index, "contributedBy"],
          message: "plan step contributor is missing",
        });
      }
    }
    const fragmentSteps = new Map<
      string,
      { agentId: string; step: z.infer<typeof fragmentStepSchema> }
    >();
    for (const fragment of value.contributions) {
      for (const step of fragment.steps) {
        if (fragmentSteps.has(step.stepId)) {
          context.addIssue({
            code: "custom",
            path: ["contributions"],
            message: `worker fragments reuse step ID: ${step.stepId}`,
          });
          return;
        }
        fragmentSteps.set(step.stepId, { agentId: fragment.agentId, step });
      }
    }
    if (fragmentSteps.size !== value.steps.length) {
      context.addIssue({
        code: "custom",
        path: ["steps"],
        message: "merged plan omitted or invented a worker fragment step",
      });
      return;
    }
    for (const [index, step] of value.steps.entries()) {
      const source = fragmentSteps.get(step.stepId);
      if (
        !source ||
        source.agentId !== step.contributedBy ||
        source.step.commandKey !== step.commandKey ||
        source.step.rationale !== step.rationale ||
        !isDeepStrictEqual(source.step.payload, step.payload) ||
        source.step.dependsOn.some((dependency) => !step.dependsOn.includes(dependency))
      ) {
        context.addIssue({
          code: "custom",
          path: ["steps", index],
          message: `merged plan rewrote worker fragment step: ${step.stepId}`,
        });
      }
    }
  });

export const needsInputResultSchema = z
  .object({ status: z.literal("needs_input"), assistantMessage: z.string().trim().min(1).max(4_000) })
  .strict();

export const cannotPlanResultSchema = z
  .object({
    status: z.literal("cannot_plan"),
    assistantMessage: z.string().trim().min(1).max(4_000),
    reason: z.string().trim().min(1).max(1_000),
    missingCapabilities: z.array(id).max(50).default([]),
  })
  .strict();

export const draftReadyResultSchema = z
  .object({
    status: z.literal("draft_ready"),
    assistantMessage: z.string().trim().min(1).max(4_000),
    proposal: planProposalSchema,
    visualMission: visualMissionSpecSchema.nullable().optional(),
  })
  .strict()
  .superRefine((value, context) => {
    const hasAction = value.proposal.steps.length > 0;
    const hasVisualMission = value.visualMission != null;
    if (hasAction === hasVisualMission) {
      context.addIssue({
        code: "custom",
        message: "draft_ready must contain exactly one action or visual mission plan",
      });
    }
    if (value.visualMission && value.visualMission.robotId !== value.proposal.robotId) {
      context.addIssue({
        code: "custom",
        path: ["visualMission", "robotId"],
        message: "visual mission robot does not match proposal",
      });
    }
  });

export const planningTurnResultSchema = z.discriminatedUnion("status", [
  needsInputResultSchema,
  cannotPlanResultSchema,
  draftReadyResultSchema,
]);

export const supervisorDecisionSchema = z.discriminatedUnion("status", [
  z
    .object({
      status: z.literal("route"),
      selectedAgentIds: z.array(id).min(1).max(8),
      rationale: z.string().trim().min(1).max(2_000),
    })
    .strict(),
  needsInputResultSchema,
  cannotPlanResultSchema,
]);

export type PlanningEnvelope = z.infer<typeof planningEnvelopeSchema>;
export type PlanFragment = z.infer<typeof planFragmentSchema>;
export type PlanProposal = z.infer<typeof planProposalSchema>;
export type PlanningTurnResult = z.infer<typeof planningTurnResultSchema>;
export type SupervisorDecision = z.infer<typeof supervisorDecisionSchema>;
export type SceneContext = z.infer<typeof sceneContextSchema>;
export type SpatialIntent = z.infer<typeof spatialIntentSchema>;
export type NavigationPayload = z.infer<typeof navigationPayloadSchema>;
export type VisualMissionStage = z.infer<typeof visualMissionStageSchema>;
export type VisualMissionSpec = z.infer<typeof visualMissionSpecSchema>;
export type VisualMissionIntent = z.infer<typeof visualMissionIntentSchema>;
export type VisualResolution = z.infer<typeof visualResolutionSchema>;
export type VisualStageRequest = z.infer<typeof visualStageRequestSchema>;
export type VisualStageResult = z.infer<typeof visualStageResultSchema>;

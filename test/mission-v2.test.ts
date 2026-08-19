import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import {
  missionBriefSchema,
  missionNodeSchema,
  planRevisionSchema,
  worldRecordSchema,
  type MissionNode,
} from "../src/mission/contracts.js";
import { MissionEngine } from "../src/mission/engine.js";

const future = (seconds = 3_600) => new Date(Date.now() + seconds * 1_000).toISOString();

function envelope() {
  return {
    schemaVersion: "h-ide.mission-envelope/2" as const,
    robotId: "robot-1",
    robotBootId: "boot-1",
    localizationEpochId: "epoch-1",
    capabilityManifestDigest: `sha256:${"a".repeat(64)}`,
    robotProfileDigest: `sha256:${"b".repeat(64)}`,
    workstationCapabilityDigest: `sha256:${"c".repeat(64)}`,
    allowedSkillIds: ["vision.observe", "navigation.navigate_to_pose", "navigation.rotate"],
    maximumRiskLevel: "medium" as const,
    allowedTargetCategories: ["box", "robot"],
    allowedRegionIds: ["kitchen"],
    sceneRefs: [{ sceneId: "scene-1", revision: 1 }],
    geofences: [],
    maximumGoalDistanceM: 5,
    maximumCumulativeDistanceM: 20,
    maximumSpeedMps: 0.2,
    maximumDurationSeconds: 600,
    maximumAutoRevisions: 3,
    maximumStageAttempts: 3,
    maximumScanYawRad: Math.PI * 2,
    modelPins: [
      { role: "text_planner" as const, provider: "dashscope", model: "qwen3.8-max", promptVersion: "mission-v2" },
      { role: "visual_grounder" as const, provider: "dashscope", model: "qwen3-vl-plus", promptVersion: "grounding-v2" },
    ],
    expiresAt: future(),
  };
}

function capabilities() {
  return [
    {
      skillId: "vision.observe",
      skillVersion: "2",
      description: "Ground a target from the current RGB-D observation",
      kind: "observation" as const,
      riskLevel: "low" as const,
      executor: "local" as const,
      executorProfile: "h-ide-vision",
      requiredSensorIds: ["head_rgbd"],
    },
    {
      skillId: "navigation.navigate_to_pose",
      skillVersion: "2",
      description: "Navigate through Nav2 to one resolved pose",
      kind: "motion" as const,
      riskLevel: "medium" as const,
      executor: "nav2" as const,
      executorProfile: "nav2.navigate_to_pose.default_bt",
      requiredSensorIds: ["head_rgbd"],
    },
    {
      skillId: "navigation.rotate",
      skillVersion: "2",
      description: "Rotate the fixed-head mobile base through Nav2 Spin",
      kind: "motion" as const,
      riskLevel: "medium" as const,
      executor: "nav2" as const,
      executorProfile: "nav2.behaviors.spin",
      requiredSensorIds: [],
    },
  ];
}

function brief() {
  return missionBriefSchema.parse({
    requestId: "request-1",
    missionId: "mission-1",
    robotId: "robot-1",
    requestedBy: "tester",
    objective: "找到银白色机器人并停在旁边",
    locale: "zh-CN",
    modality: "text",
    contextVersion: "context-1",
    envelope: envelope(),
    capabilities: capabilities(),
    history: [],
    worldRecords: [],
  });
}

function nodes(): MissionNode[] {
  return [
    {
      nodeId: "observe-target",
      skillId: "vision.observe",
      skillVersion: "2",
      purpose: "定位银白色机器人",
      intent: { targetDescription: "银白色机器人", targetCategory: "robot" },
      dependsOn: [],
      preconditionIds: [],
      postconditionIds: ["target-grounded"],
      riskLevel: "low" as const,
    },
    {
      nodeId: "approach-target",
      skillId: "navigation.navigate_to_pose",
      skillVersion: "2",
      purpose: "导航到已验证目标旁边",
      intent: { targetRecord: "target-grounded", relation: "beside" },
      dependsOn: ["observe-target"],
      preconditionIds: ["target-grounded"],
      postconditionIds: ["arrival-verified"],
      riskLevel: "medium" as const,
    },
  ];
}

function revision(overrides: Record<string, unknown> = {}) {
  return planRevisionSchema.parse({
    schemaVersion: "h-ide.plan-revision/2",
    revisionId: "revision-1",
    missionId: "mission-1",
    sequence: 1,
    parentRevisionId: null,
    objective: "找到银白色机器人并停在旁边",
    nodes: nodes(),
    reasonCode: "initial_approval",
    reason: "用户批准初始任务",
    changeSummary: ["初始任务"],
    disposition: "human_approved",
    authorizationId: "auth-1",
    createdBy: "tester",
    createdAt: new Date().toISOString(),
    ...overrides,
  });
}

describe("Mission V2 planning", () => {
  it("keeps planner intent semantic and rejects every physical-motion bypass", () => {
    for (const intent of [
      { goalPose: { x: 1, y: 2, yaw: 0 } },
      { command: { cmd_vel: { linearVelocity: 0.2 } } },
      { path: [{ x: 1, y: 2 }] },
      { rosAction: "/spin" },
    ]) {
      expect(() => missionNodeSchema.parse({ ...nodes()[0], intent })).toThrow(/executable|geometry/i);
    }
  });

  it("creates a reviewable DAG using only envelope skills", async () => {
    const runtime = {
      draftMission: vi.fn().mockResolvedValue({
        objective: brief().objective,
        nodes: nodes(),
        assumptions: [],
        warnings: ["目标坐标必须由视觉几何服务确定"],
      }),
      replanMission: vi.fn(),
    };
    const draft = await new MissionEngine(runtime).createDraft(brief(), new AbortController().signal);

    expect(draft.planId).toBe("mission-1");
    expect(draft.nodes.map((node) => node.skillId)).toEqual([
      "vision.observe",
      "navigation.navigate_to_pose",
    ]);
    expect(JSON.stringify(draft.nodes)).not.toMatch(/goalPose|cmd_vel|waypoints|rosAction/iu);
  });

  it("never promotes Qwen output directly to a world fact", () => {
    expect(() => worldRecordSchema.parse({
      schemaVersion: "h-ide.world-record/2",
      recordId: "record-1",
      missionId: "mission-1",
      robotId: "robot-1",
      kind: "fact",
      subject: "target",
      predicate: "visible",
      value: true,
      source: "qwen",
      sourceRef: "analysis-1",
      confidence: 0.99,
      observedAt: new Date().toISOString(),
      evidenceIds: [],
    })).toThrow(/Qwen|fact/i);
  });

  it("allows only a conservative low-risk repair to inherit approval", async () => {
    const repairedNodes = nodes();
    repairedNodes[0] = {
      ...repairedNodes[0]!,
      intent: { targetDescription: "银白色机器人", targetCategory: "robot", retry: "new_frame" },
    };
    const runtime = {
      draftMission: vi.fn(),
      replanMission: vi.fn().mockResolvedValue({
        objective: revision().objective,
        nodes: [repairedNodes[0]],
        reasonCode: "observation_changed",
        reason: "目标需要使用新画面重新识别",
        changeSummary: ["重新观察目标"],
      }),
    };
    const result = await new MissionEngine(runtime).replan({
      requestId: "request-replan-1",
      missionId: "mission-1",
      robotId: "robot-1",
      requestedBy: "runtime",
      objective: revision().objective,
      contextVersion: "context-2",
      envelope: envelope(),
      capabilities: capabilities(),
      activeRevision: revision({ nodes: [nodes()[0]] }),
      worldRecords: [],
      trigger: "observation_updated",
      triggerDetail: "target lost",
      autoRevisionCount: 0,
      authorizationId: "auth-1",
    }, new AbortController().signal);

    expect(result.disposition).toBe("policy_inherited");
    expect(result.authorizationId).toBe("auth-1");
    expect(result.revisionId).toBe("rev-request-replan-1");
  });

  it("requires human review for a changed navigation node", async () => {
    const changed = nodes();
    changed[1] = { ...changed[1]!, intent: { targetRecord: "new-target", relation: "beside" } };
    const runtime = {
      draftMission: vi.fn(),
      replanMission: vi.fn().mockResolvedValue({
        objective: revision().objective,
        nodes: changed,
        reasonCode: "target_changed",
        reason: "导航目标语义发生变化",
        changeSummary: ["更换导航目标"],
      }),
    };
    const result = await new MissionEngine(runtime).replan({
      requestId: "request-replan-2",
      missionId: "mission-1",
      robotId: "robot-1",
      requestedBy: "runtime",
      objective: revision().objective,
      contextVersion: "context-2",
      envelope: envelope(),
      capabilities: capabilities(),
      activeRevision: revision(),
      worldRecords: [],
      trigger: "verification_failed",
      triggerDetail: "arrival verification failed",
      autoRevisionCount: 0,
      authorizationId: "auth-1",
    }, new AbortController().signal);

    expect(result.disposition).toBe("human_required");
    expect(result.authorizationId).toBeNull();
  });

  it("pauses immediately on degraded safety without asking the model", async () => {
    const runtime = { draftMission: vi.fn(), replanMission: vi.fn() };
    const decision = await new MissionEngine(runtime).handleEvent({
      missionId: "mission-1",
      robotId: "robot-1",
      eventType: "safety.changed",
      payload: { safetyMode: "estop" },
      correlationId: "event-1",
      occurredAt: new Date().toISOString(),
      activeRevision: revision(),
      autoRevisionCount: 0,
    }, {
      envelope: envelope(),
      capabilities: capabilities(),
      objective: revision().objective,
      requestedBy: "runtime",
      contextVersion: "context-2",
      worldRecords: [],
    }, new AbortController().signal);

    expect(decision.kind).toBe("pause");
    expect(runtime.replanMission).not.toHaveBeenCalled();
  });

  it("scores candidates with trusted facts but never executes the winner", () => {
    const engine = new MissionEngine({ draftMission: vi.fn(), replanMission: vi.fn() });
    const result = engine.scoreCandidates({
      missionId: "mission-1",
      robotId: "robot-1",
      envelope: envelope(),
      worldRecords: [{
        schemaVersion: "h-ide.world-record/2",
        recordId: "geometry-1",
        missionId: "mission-1",
        robotId: "robot-1",
        kind: "fact",
        subject: "silver-robot",
        predicate: "grounded",
        value: true,
        source: "geometry",
        sourceRef: "visual-result-1",
        confidence: 0.9,
        observedAt: new Date().toISOString(),
        evidenceIds: ["depth-1"],
      }],
      candidates: [{
        candidateId: "observe-again",
        skillId: "vision.observe",
        skillVersion: "2",
        purpose: "确认目标",
        intent: { targetRecord: "silver-robot" },
        riskLevel: "low",
        modelScore: 0.8,
        evidenceRecordIds: ["geometry-1"],
      }],
    });

    expect(result.selectedCandidateId).toBe("observe-again");
    expect(result).not.toHaveProperty("command");
    expect(result).not.toHaveProperty("skillRequest");
  });

  it("keeps Mission schemas compatible with DashScope structured output", () => {
    const schema = z.toJSONSchema(missionBriefSchema);
    const forbidden: string[] = [];
    const visit = (value: unknown, path = "$") => {
      if (!value || typeof value !== "object") return;
      const record = value as Record<string, unknown>;
      if ("patternProperties" in record || "propertyNames" in record) forbidden.push(path);
      Object.entries(record).forEach(([key, child]) => visit(child, `${path}.${key}`));
    };
    visit(schema);
    expect(forbidden).toEqual([]);
  });
});

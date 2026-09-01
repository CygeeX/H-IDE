import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  currentMissionVisualAmbiguity,
  latestMissionVisualResult,
  MissionVisualCandidateSelection,
  MissionVisualSummary,
  parseMissionVisualResult,
  type MissionVisualResult,
  visualTaskRequestSha256,
} from "./mission-visual-evidence";
import type { MissionRecordV2 } from "./contracts";

function result(overrides: Partial<MissionVisualResult>): MissionVisualResult {
  return {
    analysisId: "analysis-1",
    mode: "scene_description",
    status: "succeeded",
    source: {
      robotId: "robot-current",
      robotBootId: "boot-current",
      gatewayInstanceId: "gateway-current",
      observationSessionId: "observation-current",
      localizationEpochId: "epoch-current",
      sourceMode: "synthetic",
      sourceId: "mujoco:g1",
      bundleId: "bundle-current",
      sensorId: "head_rgbd",
      capturedAt: "2026-08-19T00:00:00Z",
      calibrationSha256: "b".repeat(64),
      sceneId: null,
      sceneRevision: null,
    },
    versions: {
      vlmProvider: "qwen",
      vlmModel: "qwen3-vl-plus",
      promptVersion: "grounding-v2",
      visualAlgorithmVersion: "rgbd-target-v2",
      approachPolicyVersion: "v1",
      approachPolicySha256: "c".repeat(64),
      segmentationProvider: "sam2",
    },
    hypotheses: [],
    ambiguityCandidates: [],
    geometryCandidates: [],
    observationSummary: "当前画面可见：操作台（右侧下方）。",
    directionalClear: null,
    rejectionReason: null,
    evidence: [],
    warnings: [],
    ...overrides,
  };
}

function rawVisualResult(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schemaVersion: "h-ide.visual-result/2",
    analysisId: "analysis-current",
    requestId: "skill-mission-current-1-observe-current-1",
    requestSha256: "a".repeat(64),
    missionId: "mission-current",
    revisionId: "revision-current",
    nodeId: "observe-current",
    attempt: 1,
    mode: "object_approach",
    status: "needs_input",
    source: {
      robotId: "robot-current",
      robotBootId: "boot-current",
      gatewayInstanceId: "gateway-current",
      sourceMode: "synthetic",
      sourceId: "mujoco:g1",
      bundleId: "bundle-current",
      sensorId: "head_rgbd",
      observationSessionId: "observation-current",
      localizationEpochId: "epoch-current",
      capturedAt: "2026-08-19T00:00:00Z",
      calibrationSha256: "b".repeat(64),
    },
    versions: {
      vlmProvider: "qwen",
      vlmModel: "qwen3-vl-plus",
      promptVersion: "grounding-v2",
      visualAlgorithmVersion: "rgbd-target-v2",
      approachPolicyVersion: "v1",
      approachPolicySha256: "c".repeat(64),
      segmentationProvider: "sam2",
    },
    hypotheses: [],
    ambiguityCandidates: [{
      candidateId: "candidate-current",
      label: "当前候选",
      box: [10, 20, 100, 200],
      score: 0.9,
    }],
    geometryCandidates: [],
    observationSummary: null,
    directionalClear: null,
    rejectionReason: "ambiguous_target",
    evidence: [],
    warnings: [],
    createdAt: "2026-08-19T00:00:01Z",
    ...overrides,
  };
}

function rawGeometryCandidate(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  const verified = overrides.verified ?? true;
  return {
    geometryCandidateId: "geometry-current",
    source: "deterministic_rgbd",
    role: "landmark",
    verified,
    landmarkPose: { frameId: "map", x: 1, y: 2, z: 0, yaw: 0 },
    targetPose: null,
    safetyGates: [{ name: "stable_depth", passed: verified }],
    evidenceIds: [],
    ...overrides,
  };
}

function skillExecution(visual: Record<string, unknown>): Record<string, unknown> {
  const source = visual.source as Record<string, unknown>;
  const visualStatus = String(visual.status);
  const terminalStatus = visualStatus === "needs_input" || visualStatus === "rejected"
    ? "rejected"
    : visualStatus;
  const mode = String(visual.mode);
  const input: Record<string, unknown> = {
    mode,
    instruction: "观察当前目标",
  };
  if (["object_approach", "landmark_observation", "arrival_verification"].includes(mode)) {
    input.targetDescription = "当前目标";
  }
  if (mode === "directional_free_space") {
    input.direction = "forward";
    input.requestedDistanceM = 1;
  }
  if (visual.selectedCandidateId != null || visual.selectionSourceAnalysisId != null) {
    input.candidateId = visual.selectedCandidateId;
    input.selectionSourceAnalysisId = visual.selectionSourceAnalysisId;
  }
  const request = {
    schemaVersion: "h-ide.skill-request/2",
    requestId: visual.requestId,
    missionId: visual.missionId,
    revisionId: visual.revisionId,
    nodeId: visual.nodeId,
    robotId: source.robotId,
    skillId: "vision.observe",
    skillVersion: "2.0.0",
    input,
    capabilityManifestDigest: `sha256:${"d".repeat(64)}`,
    authorizationId: "auth-current",
    idempotencyKey: visual.requestId,
    issuedAt: "2026-08-19T00:00:00Z",
    expiresAt: "2026-08-19T00:05:00Z",
  };
  const visualTaskRequest = {
    schemaVersion: "h-ide.visual-task/2",
    requestId: visual.requestId,
    missionId: visual.missionId,
    revisionId: visual.revisionId,
    nodeId: visual.nodeId,
    authorizationId: "auth-current",
    attempt: visual.attempt,
    traceId: visual.requestId,
    mode,
    instruction: input.instruction,
    targetDescription: input.targetDescription ?? null,
    direction: input.direction ?? null,
    requestedDistanceM: input.requestedDistanceM ?? null,
    expectedTargetPose: input.expectedTargetPose ?? null,
    verificationMaximumDistanceM: input.verificationMaximumDistanceM ?? 1.25,
    verificationMaximumYawRad: input.verificationMaximumYawRad ?? 0.35,
    robotId: source.robotId,
    robotBootId: source.robotBootId,
    localizationEpochId: source.localizationEpochId,
    robotProfileDigest: `sha256:${"e".repeat(64)}`,
    capabilityManifestDigest: `sha256:${"d".repeat(64)}`,
    sensorId: source.sensorId,
    requestedSourceMode: source.sourceMode,
    sceneId: source.sceneId ?? null,
    sceneRevision: source.sceneRevision ?? null,
    candidateId: input.candidateId ?? null,
    selectionSourceAnalysisId: input.selectionSourceAnalysisId ?? null,
  };
  const requestSha256 = visualTaskRequestSha256(visualTaskRequest);
  if (!requestSha256) throw new Error("visual task fixture must be canonical");
  visual.requestSha256 = requestSha256;
  const evidenceIds = Array.isArray(visual.evidence)
    ? visual.evidence.map((item) => (item as Record<string, unknown>).sha256)
    : [];
  const terminal = {
    requestId: visual.requestId,
    status: terminalStatus,
    completedAt: "2026-08-19T00:00:01Z",
    result: { visualResult: visual },
    evidenceIds,
    errorCode: visualStatus === "succeeded" ? null : `VISUAL_${visualStatus.toUpperCase()}`,
    detail: visualStatus === "succeeded"
      ? null
      : String(visual.rejectionReason || "visual task did not succeed"),
  };
  return {
    requestId: visual.requestId,
    missionId: visual.missionId,
    revisionId: visual.revisionId,
    nodeId: visual.nodeId,
    status: terminalStatus,
    request,
    result: terminal,
  };
}

function ambiguityMission(): MissionRecordV2 {
  const current = rawVisualResult();
  const stale = rawVisualResult({
    analysisId: "analysis-stale",
    requestId: "skill-mission-current-1-observe-stale-1",
    revisionId: "revision-stale",
    nodeId: "observe-stale",
    ambiguityCandidates: [{
      candidateId: "candidate-stale",
      label: "旧候选",
      box: [30, 40, 110, 210],
      score: 0.8,
    }],
  });
  const currentExecution = skillExecution(current);
  const staleExecution = skillExecution(stale);
  return {
    missionId: "mission-current",
    robotId: "robot-current",
    status: "needs_input",
    plan: {
      planType: "mission",
      schemaVersion: "h-ide.mission/2",
      planId: "mission-current",
      robotId: "robot-current",
      objective: "观察当前目标",
      nodes: [{
        nodeId: "observe-current",
        skillId: "vision.observe",
        skillVersion: "2.0.0",
        purpose: "观察当前目标",
        intent: {},
        dependsOn: [],
        preconditionIds: [],
        postconditionIds: [],
        riskLevel: "low",
      }],
      envelope: {
        schemaVersion: "h-ide.mission-envelope/2",
        robotId: "robot-current",
        robotBootId: "boot-current",
        localizationEpochId: "epoch-current",
        capabilityManifestDigest: `sha256:${"d".repeat(64)}`,
        robotProfileDigest: `sha256:${"e".repeat(64)}`,
        workstationCapabilityDigest: `sha256:${"f".repeat(64)}`,
        allowedSkillIds: ["vision.observe"],
        maximumRiskLevel: "medium",
        maximumGoalDistanceM: 5,
        maximumCumulativeDistanceM: 20,
        maximumSpeedMps: 0.2,
        maximumDurationSeconds: 600,
        maximumAutoRevisions: 3,
        maximumStageAttempts: 3,
        maximumScanYawRad: 6.28,
        sceneRefs: [],
        expiresAt: "2026-08-19T00:10:00Z",
      },
      createdAt: "2026-08-19T00:00:00Z",
      expiresAt: "2026-08-19T00:10:00Z",
    },
    authorization: { authorizationId: "auth-current" },
    activeRevision: {
      revisionId: "revision-current",
      sequence: 1,
      objective: "观察当前目标",
      disposition: "human_approved",
      changeSummary: [],
      createdAt: "2026-08-19T00:00:00Z",
      nodes: [{
        nodeId: "observe-current",
        skillId: "vision.observe",
        skillVersion: "2.0.0",
        purpose: "观察当前目标",
        intent: {},
        dependsOn: [],
        preconditionIds: [],
        postconditionIds: [],
        riskLevel: "low",
      }],
    },
    snapshot: {
      missionId: "mission-current",
      robotId: "robot-current",
      status: "needs_input",
      lastSequence: 10,
      activeRevisionId: "revision-current",
      activeNodeId: "observe-current",
      activeSkillRequestId: null,
      autoRevisionCount: 0,
      updatedAt: "2026-08-19T00:00:00Z",
      errorCode: "VISUAL_NEEDS_INPUT",
    },
    revisions: [],
    events: [{
      eventId: "event-current-terminal",
      missionId: "mission-current",
      sequence: 9,
      eventType: "skill.failed",
      occurredAt: "2026-08-19T00:00:01Z",
      correlationId: current.requestId as string,
      revisionId: "revision-current",
      nodeId: "observe-current",
      payload: structuredClone(currentExecution.result),
    }],
    worldRecords: [],
    nodeStates: [{
      missionId: "mission-current",
      revisionId: "revision-current",
      nodeId: "observe-current",
      status: "rejected",
      attempt: 1,
      state: { skillRequestId: current.requestId, resultStatus: "rejected" },
      updatedAt: "2026-08-19T00:00:00Z",
    }],
    // Deliberately put the stale projection last: array order is not identity.
    skillExecutions: [currentExecution, staleExecution],
  } as unknown as MissionRecordV2;
}

function missionWithLatestVisual(visual: Record<string, unknown>): MissionRecordV2 {
  const mission = ambiguityMission();
  const visualStatus = String(visual.status);
  const terminalStatus = visualStatus === "needs_input" || visualStatus === "rejected"
    ? "rejected"
    : visualStatus;
  const execution = skillExecution(visual);
  mission.skillExecutions = [execution];
  mission.nodeStates[0] = {
    ...mission.nodeStates[0],
    status: terminalStatus,
    attempt: Number(visual.attempt),
    state: { skillRequestId: visual.requestId, resultStatus: terminalStatus },
  };
  mission.events = [{
    ...mission.events[0],
    eventType: terminalStatus === "succeeded" ? "skill.succeeded" : "skill.failed",
    correlationId: String(visual.requestId),
    payload: structuredClone(execution.result) as Record<string, unknown>,
  }];
  return mission;
}

function pointMissionAtVisual(
  mission: MissionRecordV2,
  visual: Record<string, unknown>,
  sequence: number,
): void {
  const execution = skillExecution(visual);
  const terminalStatus = String(execution.status);
  mission.skillExecutions.push(execution);
  mission.nodeStates[0] = {
    ...mission.nodeStates[0],
    status: terminalStatus,
    attempt: Number(visual.attempt),
    state: { skillRequestId: visual.requestId, resultStatus: terminalStatus },
  };
  mission.events.push({
    ...mission.events[0],
    eventId: `event-visual-${sequence}`,
    sequence,
    eventType: terminalStatus === "succeeded" ? "skill.succeeded" : "skill.failed",
    correlationId: String(visual.requestId),
    payload: structuredClone(execution.result) as Record<string, unknown>,
  });
}

afterEach(cleanup);

describe("Mission visual result summary", () => {
  it("shows a scene observation without claiming that a semantic target failed", () => {
    render(<MissionVisualSummary
      result={result({})}
      observationOnly
      onOpen={vi.fn()}
    />);

    expect(screen.getByText(/当前画面可见：操作台/)).toBeInTheDocument();
    expect(screen.getByText(/场景描述不生成坐标/)).toBeInTheDocument();
    expect(screen.queryByText("未识别到语义目标")).not.toBeInTheDocument();
  });

  it("shows blocked free space as a skipped navigation branch", () => {
    render(<MissionVisualSummary
      result={result({
        mode: "directional_free_space",
        observationSummary: null,
        directionalClear: false,
        geometryCandidates: [{
          geometryCandidateId: "geometry-1",
          role: "free_space",
          verified: false,
          targetPose: { frameId: "map", x: 1, y: 0, z: 0, yaw: 0 },
          safetyGates: [{
            name: "directional_clearance_m",
            passed: false,
            measured: 0.98,
            threshold: 1.5,
          }],
        }],
      })}
      observationOnly={false}
      onOpen={vi.fn()}
    />);

    expect(screen.getByText("前方净空不足")).toBeInTheDocument();
    expect(screen.getByText("0.98 m / 1.50 m")).toBeInTheDocument();
    expect(screen.getByText("净空不足，已跳过本次导航")).toBeInTheDocument();
  });

  it("submits the immutable candidate and source analysis instead of an ordinal", () => {
    const onSelect = vi.fn();
    render(<MissionVisualCandidateSelection
      result={result({
        status: "needs_input",
        rejectionReason: "ambiguous_target",
        ambiguityCandidates: [{
          candidateId: "candidate-left",
          label: "左侧银白色机器人",
          box: [100, 100, 260, 440],
          score: 0.91,
        }],
      })}
      busy={false}
      onSelect={onSelect}
    />);

    fireEvent.click(screen.getByRole("button", { name: /左侧银白色机器人/ }));

    expect(onSelect).toHaveBeenCalledWith(
      "candidate-left",
      "analysis-1",
      "左侧银白色机器人",
    );
    expect(screen.getByText(/不会直接触发移动/)).toBeInTheDocument();
  });

  it("disables candidate submission while an intervention is pending", () => {
    const onSelect = vi.fn();
    render(<MissionVisualCandidateSelection
      result={result({
        ambiguityCandidates: [{
          candidateId: "candidate-left",
          label: "左侧银白色机器人",
          box: [100, 100, 260, 440],
          score: 0.91,
        }],
      })}
      busy
      onSelect={onSelect}
    />);

    const button = screen.getByRole("button", { name: /左侧银白色机器人/ });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("renders numbered candidate boxes and crops from the same evidence image", async () => {
    const onLoadEvidence = vi.fn().mockResolvedValue("blob:candidate-evidence");
    render(<MissionVisualCandidateSelection
      result={result({
        status: "needs_input",
        ambiguityCandidates: [
          {
            candidateId: "candidate-left",
            label: "左侧椅子",
            box: [80, 100, 220, 420],
            score: 0.93,
          },
          {
            candidateId: "candidate-right",
            label: "右侧椅子",
            box: [360, 100, 520, 420],
            score: 0.88,
          },
        ],
        evidence: [{
          kind: "input_rgb",
          sha256: "f".repeat(64),
          mediaType: "image/jpeg",
        }],
      })}
      busy={false}
      missionId="mission-candidates"
      onLoadEvidence={onLoadEvidence}
      onSelect={vi.fn()}
    />);

    expect(await screen.findByRole("img", { name: "视觉候选编号图" })).toBeInTheDocument();
    expect(screen.getByLabelText("带编号候选框的视觉凭证")).toHaveTextContent("12");
    expect(document.querySelectorAll(".candidate-crop")).toHaveLength(2);
    expect(onLoadEvidence).toHaveBeenCalledWith(
      "mission-candidates",
      "analysis-1",
      "f".repeat(64),
    );
  });
});

describe("Mission visual ambiguity binding", () => {
  it("uses the active node request identity instead of skill execution array order", () => {
    const mission = ambiguityMission();

    expect(latestMissionVisualResult(mission)?.analysisId).toBe("analysis-current");
    expect(currentMissionVisualAmbiguity(mission)).toMatchObject({
      analysisId: "analysis-current",
      ambiguityCandidates: [{ candidateId: "candidate-current" }],
    });
  });

  it("accepts a canonical attempt-two ambiguity that is bound to the selected source candidate", () => {
    const mission = missionWithLatestVisual(rawVisualResult({
      analysisId: "analysis-retry-two",
      requestId: "skill-mission-current-1-observe-current-2",
      attempt: 2,
      selectedCandidateId: "candidate-from-attempt-one",
      selectionSourceAnalysisId: "analysis-attempt-one",
    }));

    expect(currentMissionVisualAmbiguity(mission)).toMatchObject({
      analysisId: "analysis-retry-two",
      status: "needs_input",
    });
  });

  it("matches the authoritative Python canonical VisualTaskRequest/2 digest", () => {
    expect(visualTaskRequestSha256({
      schemaVersion: "h-ide.visual-task/2",
      requestId: "skill-mission-current-1-observe-current-1",
      missionId: "mission-current",
      revisionId: "revision-current",
      nodeId: "observe-current",
      authorizationId: "auth-current",
      attempt: 1,
      traceId: "skill-mission-current-1-observe-current-1",
      mode: "object_approach",
      instruction: "观察当前目标",
      targetDescription: "当前目标",
      direction: null,
      requestedDistanceM: null,
      expectedTargetPose: null,
      verificationMaximumDistanceM: 1.25,
      verificationMaximumYawRad: 0.35,
      robotId: "robot-current",
      robotBootId: "boot-current",
      localizationEpochId: "epoch-current",
      robotProfileDigest: `sha256:${"e".repeat(64)}`,
      capabilityManifestDigest: `sha256:${"d".repeat(64)}`,
      sensorId: "head_rgbd",
      requestedSourceMode: "synthetic",
      sceneId: null,
      sceneRevision: null,
      candidateId: null,
      selectionSourceAnalysisId: null,
    })).toBe("8cee689bf7ab26431535887150eab6a22a5a03873768123bea3d27a93a05c27f");
    expect(visualTaskRequestSha256({
      attempt: 1,
      requestedDistanceM: 1,
      expectedTargetPose: {
        frameId: "map",
        x: 1,
        y: 0.00001,
        z: -0,
        yaw: 1e16,
      },
    })).toBe("d795afac2621ac37b8bc65c622079d7f60c25ae77394c19775365d10142d0e02");
  });

  it("orders multiple active-revision visual pointers by terminal event sequence", () => {
    const mission = ambiguityMission();
    const second = rawVisualResult({
      analysisId: "analysis-second-current",
      requestId: "skill-mission-current-1-observe-second-1",
      nodeId: "observe-second",
      ambiguityCandidates: [{
        candidateId: "candidate-second",
        label: "第二个视觉节点候选",
        box: [20, 30, 120, 220],
        score: 0.85,
      }],
    });
    mission.activeRevision.nodes.push({
      ...mission.activeRevision.nodes[0],
      nodeId: "observe-second",
    });
    mission.nodeStates.push({
      ...mission.nodeStates[0],
      nodeId: "observe-second",
      state: { skillRequestId: second.requestId, resultStatus: "rejected" },
    });
    const secondExecution = skillExecution(second);
    mission.events.push({
      ...mission.events[0],
      eventId: "event-second-terminal",
      sequence: 10,
      correlationId: String(second.requestId),
      nodeId: "observe-second",
      payload: structuredClone(secondExecution.result) as Record<string, unknown>,
    });
    // Put the newest durable result first so reverse-array selection chooses incorrectly.
    mission.skillExecutions.unshift(secondExecution);

    expect(latestMissionVisualResult(mission)?.analysisId).toBe("analysis-second-current");
  });

  it("fails closed when any active execution identity or terminal status is inconsistent", () => {
    const wrongVisualNode = ambiguityMission();
    const currentExecution = wrongVisualNode.skillExecutions[0];
    const terminal = currentExecution.result as Record<string, unknown>;
    const terminalResult = terminal.result as Record<string, unknown>;
    terminalResult.visualResult = {
      ...(terminalResult.visualResult as Record<string, unknown>),
      nodeId: "observe-forged",
    };
    expect(currentMissionVisualAmbiguity(wrongVisualNode)).toBeNull();

    const wrongExecutionStatus = ambiguityMission();
    wrongExecutionStatus.skillExecutions[0].status = "running";
    expect(currentMissionVisualAmbiguity(wrongExecutionStatus)).toBeNull();

    const duplicateRequest = ambiguityMission();
    duplicateRequest.skillExecutions.push({ ...duplicateRequest.skillExecutions[0] });
    expect(currentMissionVisualAmbiguity(duplicateRequest)).toBeNull();

    const activeRequestStillRunning = ambiguityMission();
    activeRequestStillRunning.snapshot.activeSkillRequestId = "request-current";
    expect(currentMissionVisualAmbiguity(activeRequestStillRunning)).toBeNull();

    const nonVisualActiveNode = ambiguityMission();
    (nonVisualActiveNode.activeRevision.nodes[0] as { skillId: string }).skillId = "navigation.navigate_to_pose";
    expect(currentMissionVisualAmbiguity(nonVisualActiveNode)).toBeNull();
  });

  const boundIdentityCases: Array<[string, (mission: MissionRecordV2) => void]> = [
    ["request robot", (mission) => {
      ((mission.skillExecutions[0].request as Record<string, unknown>)).robotId = "robot-forged";
    }],
    ["capability manifest", (mission) => {
      ((mission.skillExecutions[0].request as Record<string, unknown>)).capabilityManifestDigest = `sha256:${"0".repeat(64)}`;
    }],
    ["request mode", (mission) => {
      const request = mission.skillExecutions[0].request as Record<string, unknown>;
      (request.input as Record<string, unknown>).mode = "scene_description";
    }],
    ["visual source boot", (mission) => {
      const terminal = mission.skillExecutions[0].result as Record<string, unknown>;
      const resultBody = terminal.result as Record<string, unknown>;
      const visual = resultBody.visualResult as Record<string, unknown>;
      (visual.source as Record<string, unknown>).robotBootId = "boot-forged";
    }],
    ["node attempt", (mission) => { mission.nodeStates[0].attempt = 2; }],
    ["terminal event identity", (mission) => { mission.events[0].nodeId = "observe-forged"; }],
  ];
  it.each(boundIdentityCases)("rejects a bound visual result with wrong %s identity", (_name, corrupt) => {
    const mission = ambiguityMission();
    corrupt(mission);
    expect(latestMissionVisualResult(mission)).toBeNull();
    expect(currentMissionVisualAmbiguity(mission)).toBeNull();
  });

  it("rejects a fully self-consistent request chain whose canonical request id is forged", () => {
    const mission = ambiguityMission();
    const terminal = mission.skillExecutions[0].result as Record<string, unknown>;
    const resultBody = terminal.result as Record<string, unknown>;
    const forgedVisual = structuredClone(resultBody.visualResult) as Record<string, unknown>;
    const forgedRequestId = "forged-mission-current-1-observe-current-1";
    forgedVisual.requestId = forgedRequestId;
    const forgedExecution = skillExecution(forgedVisual);
    mission.skillExecutions[0] = forgedExecution;
    mission.nodeStates[0].state = {
      skillRequestId: forgedRequestId,
      resultStatus: "rejected",
    };
    mission.events[0].correlationId = forgedRequestId;
    mission.events[0].payload = structuredClone(forgedExecution.result) as Record<string, unknown>;

    expect(latestMissionVisualResult(mission)).toBeNull();
    expect(currentMissionVisualAmbiguity(mission)).toBeNull();
  });

  it("rejects a mismatched terminal event payload and a well-formed but wrong request digest", () => {
    const mismatchedEvent = ambiguityMission();
    mismatchedEvent.events[0].payload.status = "succeeded";
    expect(latestMissionVisualResult(mismatchedEvent)).toBeNull();

    const wrongDigest = ambiguityMission();
    const terminal = wrongDigest.skillExecutions[0].result as Record<string, unknown>;
    const resultBody = terminal.result as Record<string, unknown>;
    const visual = resultBody.visualResult as Record<string, unknown>;
    visual.requestSha256 = "0".repeat(64);
    wrongDigest.events[0].payload = structuredClone(terminal);
    expect(latestMissionVisualResult(wrongDigest)).toBeNull();
    expect(currentMissionVisualAmbiguity(wrongDigest)).toBeNull();
  });

  it("suppresses old evidence while an active visual attempt is non-terminal", () => {
    const mission = missionWithLatestVisual(rawVisualResult({ analysisId: "analysis-old-valid" }));
    const running = skillExecution(rawVisualResult({
      analysisId: "analysis-running",
      requestId: "skill-mission-current-1-observe-current-2",
      attempt: 2,
    }));
    running.status = "running";
    running.result = null;
    mission.skillExecutions.push(running);
    mission.snapshot.activeNodeId = "observe-current";
    mission.snapshot.activeSkillRequestId = "skill-mission-current-1-observe-current-2";
    mission.nodeStates[0] = {
      ...mission.nodeStates[0],
      status: "running",
      attempt: 2,
      state: { skillRequestId: "skill-mission-current-1-observe-current-2" },
    };

    expect(latestMissionVisualResult(mission)).toBeNull();
  });

  it("keeps the last bound visual evidence while a non-visual skill is active", () => {
    const mission = ambiguityMission();
    mission.activeRevision.nodes.push({
      ...mission.activeRevision.nodes[0],
      nodeId: "navigate-current",
      skillId: "navigation.navigate_to_pose",
    });
    mission.snapshot.activeNodeId = "navigate-current";
    mission.snapshot.activeSkillRequestId = "skill-mission-current-1-navigate-current-1";
    mission.skillExecutions.push({
      requestId: mission.snapshot.activeSkillRequestId,
      missionId: "mission-current",
      revisionId: "revision-current",
      nodeId: "navigate-current",
      status: "running",
      request: {
        schemaVersion: "h-ide.skill-request/2",
        requestId: mission.snapshot.activeSkillRequestId,
        missionId: "mission-current",
        revisionId: "revision-current",
        nodeId: "navigate-current",
        robotId: "robot-current",
        skillId: "navigation.navigate_to_pose",
        skillVersion: "2.0.0",
      },
      result: null,
    });

    expect(latestMissionVisualResult(mission)?.analysisId).toBe("analysis-current");
  });

  it("does not show old evidence when an active non-visual pointer cannot be proven", () => {
    const mission = ambiguityMission();
    mission.activeRevision.nodes.push({
      ...mission.activeRevision.nodes[0],
      nodeId: "navigate-current",
      skillId: "navigation.navigate_to_pose",
    });
    mission.snapshot.activeNodeId = "navigate-current";
    mission.snapshot.activeSkillRequestId = "skill-mission-current-1-navigate-current-1";
    mission.skillExecutions.push({
      requestId: mission.snapshot.activeSkillRequestId,
      missionId: "mission-current",
      revisionId: "revision-current",
      nodeId: "navigate-current",
      status: "running",
      request: {
        requestId: mission.snapshot.activeSkillRequestId,
        revisionId: "revision-current",
        nodeId: "navigate-current",
        robotId: "robot-current",
        skillId: "navigation.navigate_to_pose",
        skillVersion: "2.0.0",
      },
      result: null,
    });

    expect(latestMissionVisualResult(mission)).toBeNull();
  });

  it.each([
    ["blank analysis id", { analysisId: "   " }],
    ["duplicate canonical candidate ids", {
      ambiguityCandidates: [
        { candidateId: "candidate-a", label: "候选 A", box: [1, 2, 3, 4], score: 0.8 },
        { candidateId: " candidate-a ", label: "候选 B", box: [5, 6, 7, 8], score: 0.7 },
      ],
    }],
    ["one malformed candidate in an otherwise valid list", {
      ambiguityCandidates: [
        { candidateId: "candidate-a", label: "候选 A", box: [1, 2, 3, 4], score: 0.8 },
        { candidateId: "candidate-b", label: "候选 B", box: [5, 6, 7], score: 0.7 },
      ],
    }],
    ["one malformed hypothesis in an otherwise valid result", { hypotheses: [42] }],
    ["a hypothesis with a negative box coordinate", {
      hypotheses: [{
        hypothesisId: "hypothesis-invalid",
        source: "qwen",
        label: "非法框",
        pixelBox: [-1, 2, 3, 4],
        score: 0.5,
      }],
    }],
    ["a hypothesis with a reversed box", {
      hypotheses: [{
        hypothesisId: "hypothesis-invalid",
        source: "qwen",
        label: "非法框",
        pixelBox: [4, 2, 3, 8],
        score: 0.5,
      }],
    }],
    ["a hypothesis score above one", {
      hypotheses: [{
        hypothesisId: "hypothesis-invalid",
        source: "qwen",
        label: "非法分数",
        pixelBox: [1, 2, 3, 4],
        score: 1.01,
      }],
    }],
    ["a fractional ambiguity box", {
      ambiguityCandidates: [{
        candidateId: "candidate-invalid",
        label: "非法候选",
        box: [1.5, 2, 3, 4],
        score: 0.5,
      }],
    }],
  ])("rejects %s without exposing a partial result", (_name, overrides) => {
    expect(parseMissionVisualResult(
      rawVisualResult(overrides as Record<string, unknown>),
    )).toBeNull();
  });

  const strictContractCases: Array<[string, (visual: Record<string, unknown>) => void]> = [
    ["requestSha256", (visual) => { visual.requestSha256 = "not-a-sha"; }],
    ["request identity", (visual) => { delete visual.requestId; }],
    ["attempt", (visual) => { visual.attempt = 0; }],
    ["createdAt", (visual) => { visual.createdAt = "2026-08-19 00:00:01"; }],
    ["source robot identity", (visual) => {
      delete (visual.source as Record<string, unknown>).robotId;
    }],
    ["source observation session", (visual) => {
      delete (visual.source as Record<string, unknown>).observationSessionId;
    }],
    ["source localization epoch", (visual) => {
      delete (visual.source as Record<string, unknown>).localizationEpochId;
    }],
    ["sourceMode literal", (visual) => {
      (visual.source as Record<string, unknown>).sourceMode = "oracle";
    }],
    ["calibrationSha256", (visual) => {
      (visual.source as Record<string, unknown>).calibrationSha256 = "bad";
    }],
    ["promptVersion", (visual) => {
      delete (visual.versions as Record<string, unknown>).promptVersion;
    }],
    ["approachPolicySha256", (visual) => {
      (visual.versions as Record<string, unknown>).approachPolicySha256 = "bad";
    }],
    ["evidence sha256", (visual) => {
      visual.evidence = [{ kind: "rgb_keyframe", sha256: "bad", mediaType: "image/jpeg" }];
    }],
    ["evidence kind", (visual) => {
      visual.evidence = [{ kind: " ", sha256: "e".repeat(64), mediaType: "image/jpeg" }];
    }],
    ["evidence media type", (visual) => {
      visual.evidence = [{ kind: "rgb_keyframe", sha256: "e".repeat(64), mediaType: " " }];
    }],
  ];
  it.each(strictContractCases)("rejects an invalid VisualTaskResult/2 %s field", (_name, corrupt) => {
    const visual = rawVisualResult();
    corrupt(visual);
    expect(parseMissionVisualResult(visual)).toBeNull();
  });

  it("canonicalizes analysis and candidate text before exposing a selection", () => {
    const visual = rawVisualResult({
      analysisId: " analysis-canonical ",
      ambiguityCandidates: [{
        candidateId: " candidate-canonical ",
        label: " 当前候选 ",
        box: [10, 20, 100, 200],
        score: null,
      }],
    });

    expect(parseMissionVisualResult(visual)).toMatchObject({
      analysisId: "analysis-canonical",
      ambiguityCandidates: [{
        candidateId: "candidate-canonical",
        label: "当前候选",
      }],
    });
  });

  it("does not fall back to an older result when the newest visual payload is malformed", () => {
    const mission = missionWithLatestVisual(rawVisualResult({ analysisId: "analysis-old-valid" }));
    pointMissionAtVisual(mission, rawVisualResult({
      analysisId: "analysis-new-malformed",
      requestId: "skill-mission-current-1-observe-current-2",
      attempt: 2,
      mode: "unsupported_visual_mode",
    }), 10);

    expect(latestMissionVisualResult(mission)).toBeNull();

    const missingPayload = missionWithLatestVisual(rawVisualResult({ analysisId: "analysis-old-valid" }));
    const missingVisual = rawVisualResult({
      analysisId: "analysis-new-missing",
      requestId: "skill-mission-current-1-observe-current-2",
      attempt: 2,
    });
    pointMissionAtVisual(missingPayload, missingVisual, 10);
    const incompleteExecution = missingPayload.skillExecutions.at(-1)!;
    (incompleteExecution.result as Record<string, unknown>).result = {};
    expect(latestMissionVisualResult(missingPayload)).toBeNull();
  });

  it.each([
    ["an unsupported mode", { mode: "unsupported_visual_mode" }],
    ["a non-terminal status", { status: "resolving" }],
    ["needs_input without candidates", { ambiguityCandidates: [] }],
    ["successful semantic grounding without verified geometry", {
      status: "succeeded",
      ambiguityCandidates: [],
      rejectionReason: null,
    }],
    ["successful scene description without a summary", {
      mode: "scene_description",
      status: "succeeded",
      ambiguityCandidates: [],
      observationSummary: "   ",
      rejectionReason: null,
    }],
    ["successful scene description with fabricated geometry", {
      mode: "scene_description",
      status: "succeeded",
      ambiguityCandidates: [],
      observationSummary: "场景摘要",
      geometryCandidates: [rawGeometryCandidate()],
      rejectionReason: null,
    }],
    ["directionalClear inconsistent with deterministic geometry", {
      mode: "directional_free_space",
      status: "succeeded",
      ambiguityCandidates: [],
      directionalClear: false,
      geometryCandidates: [rawGeometryCandidate({
        role: "free_space",
        targetPose: { frameId: "map", x: 1, y: 0, z: 0, yaw: 0 },
      })],
      rejectionReason: null,
    }],
    ["successful semantic grounding with an unverified candidate", {
      status: "succeeded",
      ambiguityCandidates: [],
      geometryCandidates: [rawGeometryCandidate({ verified: false })],
      rejectionReason: null,
    }],
    ["a successful result carrying a rejection reason", {
      status: "succeeded",
      ambiguityCandidates: [],
      geometryCandidates: [rawGeometryCandidate()],
      rejectionReason: "must-not-survive",
    }],
    ["a rejected result without a rejection reason", {
      status: "rejected",
      ambiguityCandidates: [],
      rejectionReason: null,
    }],
    ["a failed result without a rejection reason", {
      status: "failed",
      ambiguityCandidates: [],
      rejectionReason: null,
    }],
    ["a non-scene result carrying observationSummary", {
      observationSummary: "不应出现的场景摘要",
    }],
    ["a non-directional result carrying directionalClear", {
      directionalClear: true,
    }],
  ])("rejects terminal-invariant violation: %s", (_name, overrides) => {
    expect(parseMissionVisualResult(
      rawVisualResult(overrides as Record<string, unknown>),
    )).toBeNull();
  });

  it.each([
    ["scene description", {
      mode: "scene_description",
      status: "succeeded",
      ambiguityCandidates: [],
      observationSummary: "当前可见会议桌。",
      rejectionReason: null,
    }],
    ["directional free space", {
      mode: "directional_free_space",
      status: "succeeded",
      ambiguityCandidates: [],
      directionalClear: true,
      geometryCandidates: [rawGeometryCandidate({
        role: "free_space",
        targetPose: { frameId: "map", x: 1, y: 0, z: 0, yaw: 0 },
      })],
      rejectionReason: null,
    }],
    ["blocked directional free space", {
      mode: "directional_free_space",
      status: "succeeded",
      ambiguityCandidates: [],
      directionalClear: false,
      geometryCandidates: [rawGeometryCandidate({
        role: "free_space",
        verified: false,
        targetPose: { frameId: "map", x: 1, y: 0, z: 0, yaw: 0 },
      })],
      rejectionReason: null,
    }],
    ["verified semantic geometry", {
      status: "succeeded",
      ambiguityCandidates: [],
      geometryCandidates: [rawGeometryCandidate()],
      rejectionReason: null,
    }],
    ["explicit rejection", {
      status: "rejected",
      ambiguityCandidates: [],
      rejectionReason: "target_not_visible",
    }],
  ])("accepts a backend-valid terminal result: %s", (_name, overrides) => {
    expect(parseMissionVisualResult(
      rawVisualResult(overrides as Record<string, unknown>),
    )).not.toBeNull();
  });
});

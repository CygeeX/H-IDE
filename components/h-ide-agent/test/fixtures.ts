export function envelope(overrides: Record<string, unknown> = {}) {
  const base = {
    taskRequest: {
      schemaVersion: "1.0",
      taskId: "task-1",
      conversationId: "conversation-1",
      robotId: "robot-1",
      requestedBy: "tester",
      modality: "text",
      text: "向前移动后打开夹爪",
      locale: "zh-CN",
      transcriptionId: null,
      revisionOfDraftId: null,
      createdAt: new Date().toISOString(),
    },
    contextVersion: "ctx-1",
    history: [
      { role: "user", content: "向前移动后打开夹爪", createdAt: new Date().toISOString() },
    ],
    robotState: {
      robotId: "robot-1",
      status: "online",
      connected: true,
      safetyMode: "normal",
      details: {},
    },
    capabilitySnapshot: {
      agent: "robot-agent",
      protocol: "fake",
      schema: { type: "object" },
      capabilities: ["move.forward", "stop", "gripper.open"],
      safetyMode: "normal",
      robotId: "robot-1",
    },
    commandCatalog: [
      {
        commandKey: "move.forward",
        version: "1",
        kind: "motion",
        description: null,
        payloadSchema: { type: "object" },
      },
      {
        commandKey: "gripper.open",
        version: "1",
        kind: "non_motion",
        description: null,
        payloadSchema: { type: "object" },
      },
    ],
  };
  const result = { ...base, ...overrides } as typeof base;
  if ("taskRequest" in overrides) {
    result.taskRequest = { ...base.taskRequest, ...(overrides.taskRequest as Record<string, unknown>) } as typeof base.taskRequest;
  }
  if ("robotState" in overrides) {
    result.robotState = { ...base.robotState, ...(overrides.robotState as Record<string, unknown>) } as typeof base.robotState;
  }
  return result;
}

export function motionFragment() {
  return {
    agentId: "motion-worker",
    contribution: "motion",
    summary: "移动规划",
    steps: [
      {
        stepId: "move",
        commandKey: "move.forward",
        payload: {},
        dependsOn: [],
        rationale: "接近目标",
      },
    ],
    assumptions: [],
    warnings: [],
  };
}

export function manipulationFragment() {
  return {
    agentId: "manipulation-worker",
    contribution: "manipulation",
    summary: "夹爪规划",
    steps: [
      {
        stepId: "open",
        commandKey: "gripper.open",
        payload: {},
        dependsOn: ["move"],
        rationale: "打开夹爪",
      },
    ],
    assumptions: [],
    warnings: [],
  };
}

export function planProposal() {
  return {
    robotId: "robot-1",
    summary: "先移动，再打开夹爪",
    steps: [
      { ...motionFragment().steps[0], contributedBy: "motion-worker" },
      { ...manipulationFragment().steps[0], contributedBy: "manipulation-worker" },
    ],
    assumptions: [],
    warnings: [],
    contributions: [motionFragment(), manipulationFragment()],
    contextVersion: "ctx-1",
  };
}

export function visualMission(overrides: Record<string, unknown> = {}) {
  return {
    missionId: "task-visual-1",
    robotId: "robot-1",
    instruction: "走到黄色箱子旁边",
    locale: "zh-CN",
    targetDescription: "黄色箱子",
    requestedSourceMode: "synthetic",
    fusionMode: "vision_only",
    stages: [{
      stageId: "approach-1",
      kind: "approach",
      targetDescription: "黄色箱子",
      goalPurpose: "approach",
    }],
    robotBootId: "fixture-boot-1",
    localizationEpochId: "fixture-map-epoch-1",
    sensorIds: ["head_rgbd"],
    allowedTargetCategories: ["box"],
    unsupportedCapabilities: ["path_planning", "direct_robot_control"],
    maximumGoalDistanceM: 5,
    maximumCumulativeDistanceM: 20,
    maximumDurationSeconds: 600,
    vlmProvider: "FakeVlmProvider",
    vlmModel: "fake-fixture-v1",
    promptVersion: "visual-grounding-v1",
    visualAlgorithmVersion: "rgbd-median-v1",
    approachPolicyVersion: "v1",
    approachPolicySha256: "a97bcf51b108b7c5c5241a82e2710eadb2619f12a1aba6e328d77b5474f35fc0",
    maximumSpeedMps: 0.2,
    reviewExpiresAt: new Date(Date.now() + 300_000).toISOString(),
    ...overrides,
  };
}

export function visualResolution(overrides: Record<string, unknown> = {}) {
  return {
    visualResolutionId: "vr-fixture-1",
    missionId: "task-visual-1",
    stageId: "approach-1",
    attempt: 1,
    traceId: "trace-visual-1",
    status: "succeeded",
    executable: true,
    targetPhrase: "黄色箱子",
    source: {
      robotId: "robot-1",
      robotBootId: "fixture-boot-1",
      gatewayInstanceId: "fixture-gateway-1",
      sensorId: "head_rgbd",
      observationSessionId: "fixture-observation-1",
      localizationEpochId: "fixture-map-epoch-1",
      sourceMode: "synthetic",
      sourceId: "fixture-sha256",
      bundleId: "bundle-1",
      capturedAt: new Date().toISOString(),
      calibrationSha256: "b".repeat(64),
    },
    versions: {
      vlmProvider: "FakeVlmProvider",
      vlmModel: "fake-fixture-v1",
      promptVersion: "visual-grounding-v1",
      visualAlgorithmVersion: "rgbd-median-v1",
      approachPolicyVersion: "v1",
      approachPolicySha256: "a97bcf51b108b7c5c5241a82e2710eadb2619f12a1aba6e328d77b5474f35fc0",
      segmentationModel: "box-mask-v1",
      poseModel: null,
    },
    landmarkPose: { frameId: "map", x: 2, y: 0, z: 0, yaw: 0 },
    goalPose: { frameId: "map", x: 1.25, y: 0, z: 0, yaw: 0 },
    goalPurpose: "approach",
    safetyGates: [{ name: "stable_frames", passed: true, measured: 3, threshold: 3 }],
    ambiguityCandidates: [],
    warnings: [],
    rejectionReason: null,
    evidence: [],
    createdAt: new Date().toISOString(),
    ...overrides,
  };
}

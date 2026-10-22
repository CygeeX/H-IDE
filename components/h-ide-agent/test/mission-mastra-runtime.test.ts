import { describe, expect, it, vi } from "vitest";
import { MastraMissionRuntime } from "../src/mission/mastra-runtime.js";

function capability(skillId: string, description: string) {
  return {
    skillId,
    skillVersion: "2",
    description,
    kind: "observation" as const,
    riskLevel: "low" as const,
    executor: "local" as const,
    executorProfile: `${skillId}.default`,
    requiredSensorIds: [],
  };
}

function missionInput() {
  const capabilities = [
    capability("vision.observe", "观察当前 RGB-D 画面并识别视觉目标"),
    capability("inspection.scan_station", "在 Office 巡检站点旋转检查并采集场景描述"),
    capability("mission.report", "汇总站点观察并生成 Office 巡检报告"),
    ...Array.from({ length: 8 }, (_, index) => capability(
      `unrelated.skill_${index}`,
      `处理不相关任务 ${index}`,
    )),
  ];
  return {
    missionId: "mission-1",
    requestedBy: "user-1",
    objective: "检查办公室并生成巡检报告",
    capabilities,
    envelope: {
      capabilityManifestDigest: `sha256:${"a".repeat(64)}`,
      workstationCapabilityDigest: `sha256:${"b".repeat(64)}`,
      allowedSkillIds: capabilities.map((item) => item.skillId),
    },
  };
}

function semanticNode() {
  return {
    nodeId: "observe-target",
    skillId: "vision.observe",
    skillVersion: "2",
    purpose: "Observe the target",
    intent: { task: "scene_description" },
    dependsOn: [],
    preconditionIds: [],
    postconditionIds: ["target-observed"],
    riskLevel: "low" as const,
  };
}

describe("Mastra Mission memory scope", () => {
  it("keeps one stable owner when a user-created Mission is replanned by the runtime", async () => {
    const agent = {
      generate: vi.fn()
        .mockResolvedValueOnce({
          object: {
            objective: "Observe the target",
            nodes: [semanticNode()],
            assumptions: [],
            warnings: [],
          },
        })
        .mockResolvedValueOnce({
          object: {
            objective: "Observe the target again",
            nodes: [semanticNode()],
            reasonCode: "observation_updated",
            reason: "A newer observation is available",
            changeSummary: ["refresh-observation"],
          },
        }),
    };
    const runtime = new MastraMissionRuntime(agent as never);
    const signal = new AbortController().signal;

    const input = missionInput();
    await runtime.draftMission(input as never, signal);
    await runtime.replanMission({
      ...input,
      requestedBy: "mission-runtime",
      triggerDetail: "A newer observation is available",
      activeRevision: { nodes: [semanticNode()] },
    } as never, signal);

    const draftMemory = agent.generate.mock.calls[0]![1]!.memory;
    const replanMemory = agent.generate.mock.calls[1]![1]!.memory;
    expect(draftMemory).toEqual({
      thread: "mission:v2:mission-1",
      resource: "h-ide:mission:mission-1",
    });
    expect(replanMemory).toEqual(draftMemory);
  });

  it("sends only BM25 candidates to the model without mutating the full authorization", async () => {
    const agent = {
      generate: vi.fn().mockResolvedValue({
        object: {
          objective: "检查办公室并生成巡检报告",
          nodes: [{
            ...semanticNode(),
            skillId: "inspection.scan_station",
            purpose: "执行办公室巡检",
          }],
          assumptions: [],
          warnings: [],
        },
      }),
    };
    const input = missionInput();
    const originalAllowedSkillIds = [...input.envelope.allowedSkillIds];
    const runtime = new MastraMissionRuntime(agent as never);

    await runtime.draftMission(input as never, new AbortController().signal);

    const prompt = agent.generate.mock.calls[0]![0] as string;
    const promptInput = JSON.parse(prompt.split("Mission brief: ")[1]!) as {
      capabilities: Array<{ skillId: string }>;
      envelope: { allowedSkillIds: string[] };
    };
    const candidateSkillIds = promptInput.capabilities.map((item) => item.skillId);
    expect(candidateSkillIds).toContain("inspection.scan_station");
    expect(candidateSkillIds).toContain("mission.report");
    expect(promptInput.capabilities.length).toBeLessThan(input.capabilities.length);
    expect(promptInput.envelope.allowedSkillIds).toEqual(candidateSkillIds);
    expect(input.capabilities).toHaveLength(11);
    expect(input.envelope.allowedSkillIds).toEqual(originalAllowedSkillIds);
  });
});

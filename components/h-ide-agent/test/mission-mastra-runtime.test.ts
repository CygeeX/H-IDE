import { describe, expect, it, vi } from "vitest";
import { MastraMissionRuntime } from "../src/mission/mastra-runtime.js";

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

    await runtime.draftMission({
      missionId: "mission-1",
      requestedBy: "user-1",
    } as never, signal);
    await runtime.replanMission({
      missionId: "mission-1",
      requestedBy: "mission-runtime",
    } as never, signal);

    const draftMemory = agent.generate.mock.calls[0]![1]!.memory;
    const replanMemory = agent.generate.mock.calls[1]![1]!.memory;
    expect(draftMemory).toEqual({
      thread: "mission:v2:mission-1",
      resource: "h-ide:mission:mission-1",
    });
    expect(replanMemory).toEqual(draftMemory);
  });
});

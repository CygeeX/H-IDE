import { describe, expect, it } from "vitest";
import {
  planningTurnResultSchema,
  visualMissionSpecSchema,
  visualStageRequestSchema,
} from "../src/contracts.js";
import { visualMission } from "./fixtures.js";

describe("visual mission contracts", () => {
  it("accepts a semantic mission with no coordinate, path, or robot command", () => {
    const mission = visualMissionSpecSchema.parse(visualMission());
    expect(mission.allowedTargetCategories).toEqual(["box"]);
    const wire = JSON.stringify(mission);
    expect(wire).not.toContain("goalPose");
    expect(wire).not.toContain("cmd_vel");
    expect(wire.toLowerCase()).not.toContain("waypoints");
  });

  it("rejects LLM-invented execution fields and requires exactly one draft kind", () => {
    expect(() => visualMissionSpecSchema.parse({
      ...visualMission(),
      goalPose: { frameId: "map", x: 1, y: 2, z: 0, yaw: 0 },
    })).toThrow();
    const emptyProposal = {
      robotId: "robot-1",
      summary: "visual",
      steps: [],
      assumptions: [],
      warnings: [],
      contributions: [],
      contextVersion: "ctx-1",
    };
    expect(() => planningTurnResultSchema.parse({
      status: "draft_ready",
      assistantMessage: "invalid",
      proposal: emptyProposal,
    })).toThrow(/exactly one|action|visual/i);
    expect(planningTurnResultSchema.parse({
      status: "draft_ready",
      assistantMessage: "review",
      proposal: emptyProposal,
      visualMission: visualMission(),
    }).status).toBe("draft_ready");
  });

  it("binds a stage to the immutable reviewed mission", () => {
    const mission = visualMission();
    expect(visualStageRequestSchema.parse({
      mission,
      stage: mission.stages[0],
      attempt: 1,
    }).attempt).toBe(1);
    expect(() => visualStageRequestSchema.parse({
      mission,
      stage: { ...mission.stages[0], targetDescription: "另一个目标" },
      attempt: 1,
    })).toThrow(/pinned|stage/i);
    expect(() => visualStageRequestSchema.parse({
      mission,
      stage: mission.stages[0],
      attempt: 2,
      candidateId: "candidate-a",
    })).toThrow(/selection|candidate/i);
  });
});

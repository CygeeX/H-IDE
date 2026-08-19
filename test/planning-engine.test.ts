import { describe, expect, it, vi } from "vitest";
import { PlanningEngine } from "../src/planning/engine.js";
import { envelope, manipulationFragment, motionFragment, planProposal } from "./fixtures.js";

describe("Supervisor–Worker planning engine", () => {
  it("routes one request to multiple specialists and keeps their dependency", async () => {
    const runtime = {
      classify: vi.fn().mockResolvedValue({
        status: "route",
        selectedAgentIds: ["motion-worker", "manipulation-worker"],
        rationale: "需要移动和夹爪能力",
      }),
      runWorker: vi.fn(async (agentId: string) =>
        agentId === "motion-worker" ? motionFragment() : manipulationFragment(),
      ),
      merge: vi.fn().mockResolvedValue(planProposal()),
    };
    const engine = new PlanningEngine(runtime);

    const result = await engine.plan(envelope(), new AbortController().signal);

    expect(result.status).toBe("draft_ready");
    if (result.status === "draft_ready") {
      expect(result.proposal.steps[1]!.dependsOn).toEqual(["move"]);
      expect(result.proposal.contributions.map((item) => item.agentId)).toEqual([
        "motion-worker",
        "manipulation-worker",
      ]);
    }
    expect(runtime.runWorker).toHaveBeenCalledTimes(2);
  });

  it("fails the whole turn when any selected worker is malformed", async () => {
    const runtime = {
      classify: vi.fn().mockResolvedValue({
        status: "route",
        selectedAgentIds: ["motion-worker", "manipulation-worker"],
        rationale: "multi-worker",
      }),
      runWorker: vi.fn(async (agentId: string) =>
        agentId === "motion-worker" ? motionFragment() : { broken: true },
      ),
      merge: vi.fn(),
    };
    const engine = new PlanningEngine(runtime);

    await expect(engine.plan(envelope(), new AbortController().signal)).rejects.toThrow();
    expect(runtime.merge).not.toHaveBeenCalled();
  });

  it("rejects a merge that omits a worker step or rewrites its contribution", async () => {
    const omitted = planProposal();
    omitted.steps = omitted.steps.slice(0, 1);
    omitted.contributions[0]!.summary = "structuring agent rewrote this fragment";
    const runtime = {
      classify: vi.fn().mockResolvedValue({
        status: "route",
        selectedAgentIds: ["motion-worker", "manipulation-worker"],
        rationale: "multi-worker",
      }),
      runWorker: vi.fn(async (agentId: string) =>
        agentId === "motion-worker" ? motionFragment() : manipulationFragment(),
      ),
      merge: vi.fn().mockResolvedValue(omitted),
    };

    await expect(
      new PlanningEngine(runtime).plan(envelope(), new AbortController().signal),
    ).rejects.toThrow(/omit|rewrite|fragment/i);
  });

  it("returns clarification without calling a worker", async () => {
    const runtime = {
      classify: vi.fn().mockResolvedValue({
        status: "needs_input",
        assistantMessage: "请说明移动距离。",
      }),
      runWorker: vi.fn(),
      merge: vi.fn(),
    };
    const result = await new PlanningEngine(runtime).plan(envelope(), new AbortController().signal);
    expect(result).toEqual({ status: "needs_input", assistantMessage: "请说明移动距离。" });
    expect(runtime.runWorker).not.toHaveBeenCalled();
  });
});

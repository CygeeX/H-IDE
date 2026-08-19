import { describe, expect, it, vi } from "vitest";
import { PlanningEngine } from "../src/planning/engine.js";
import { VisionWorker } from "../src/vision/worker.js";
import { envelope, visualMission, visualResolution } from "./fixtures.js";

describe("vision-worker", () => {
  it("routes a visual request into a semantic VisualMissionSpec", async () => {
    const runtime = {
      classify: vi.fn().mockResolvedValue({
        status: "route",
        selectedAgentIds: ["vision-worker"],
        rationale: "current RGB-D evidence is required",
      }),
      runWorker: vi.fn(),
      runVisualWorker: vi.fn().mockResolvedValue(visualMission()),
      merge: vi.fn(),
    };
    const result = await new PlanningEngine(runtime).plan(
      envelope({
        taskRequest: {
          taskId: "task-visual-1",
          text: "走到黄色箱子旁边",
        },
        robotState: {
          details: {
            robotBootId: "fixture-boot-1",
            localizationEpochId: "fixture-map-epoch-1",
          },
        },
      }),
      new AbortController().signal,
    );
    expect(result.status).toBe("draft_ready");
    if (result.status === "draft_ready") {
      expect(result.visualMission?.targetDescription).toBe("黄色箱子");
      expect(result.proposal.steps).toEqual([]);
    }
    expect(runtime.runWorker).not.toHaveBeenCalled();
    expect(runtime.merge).not.toHaveBeenCalled();
  });

  it("returns only the canonical resolution ID and explanation", async () => {
    const client = {
      createResolution: vi.fn().mockResolvedValue(visualResolution()),
      getResolution: vi.fn(),
      cancelResolution: vi.fn(),
    };
    const result = await new VisionWorker(client).resolve({
      mission: visualMission(),
      stage: visualMission().stages[0],
      attempt: 1,
    }, new AbortController().signal);
    expect(result.status).toBe("succeeded");
    expect(result.visualResolutionId).toBe("vr-fixture-1");
    expect(result).not.toHaveProperty("goalPose");
    expect(client.getResolution).not.toHaveBeenCalled();
  });

  it("fails closed on stale identity and propagates cancellation", async () => {
    const staleClient = {
      createResolution: vi.fn().mockResolvedValue(visualResolution({
        source: { ...visualResolution().source, localizationEpochId: "stale-epoch" },
      })),
      getResolution: vi.fn(),
      cancelResolution: vi.fn(),
    };
    await expect(new VisionWorker(staleClient).resolve({
      mission: visualMission(), stage: visualMission().stages[0], attempt: 1,
    }, new AbortController().signal)).rejects.toThrow(/stale|identity/i);

    const controller = new AbortController();
    const cancelClient = {
      createResolution: vi.fn().mockResolvedValue(visualResolution({
        status: "queued", executable: false, landmarkPose: null, goalPose: null,
        safetyGates: [], rejectionReason: "pending",
      })),
      getResolution: vi.fn((_id: string, signal: AbortSignal) => new Promise<never>((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      })),
      cancelResolution: vi.fn().mockResolvedValue(visualResolution({ status: "cancelled", executable: false })),
    };
    const pending = new VisionWorker(cancelClient, 1).resolve({
      mission: visualMission(), stage: visualMission().stages[0], attempt: 1,
    }, controller.signal);
    setTimeout(() => controller.abort(new Error("test cancel")), 5);
    await expect(pending).rejects.toThrow(/cancel/i);
    expect(cancelClient.cancelResolution).toHaveBeenCalledWith("vr-fixture-1");
  });

  it("binds a selected ambiguity candidate to a new canonical attempt", async () => {
    const selected = visualResolution({
      attempt: 2,
      selectedCandidateId: "candidate-a",
      selectionSourceVisualResolutionId: "vr-ambiguous-1",
    });
    const client = {
      createResolution: vi.fn().mockResolvedValue(selected),
      getResolution: vi.fn(),
      cancelResolution: vi.fn(),
    };
    const result = await new VisionWorker(client).resolve({
      mission: visualMission(),
      stage: visualMission().stages[0],
      attempt: 2,
      candidateId: "candidate-a",
      selectionSourceVisualResolutionId: "vr-ambiguous-1",
    }, new AbortController().signal);
    expect(result.status).toBe("succeeded");
    expect(client.createResolution).toHaveBeenCalledTimes(1);
  });
});

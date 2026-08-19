import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MissionVisualSummary, type MissionVisualResult } from "./mission-visual-evidence";

function result(overrides: Partial<MissionVisualResult>): MissionVisualResult {
  return {
    analysisId: "analysis-1",
    mode: "scene_description",
    status: "succeeded",
    source: {
      sourceMode: "synthetic",
      sourceId: "mujoco:g1",
      sensorId: "head_rgbd",
      capturedAt: "2026-08-19T00:00:00Z",
    },
    versions: {
      vlmProvider: "qwen",
      vlmModel: "qwen3-vl-plus",
      visualAlgorithmVersion: "rgbd-target-v2",
      segmentationProvider: "sam2",
    },
    hypotheses: [],
    geometryCandidates: [],
    observationSummary: "当前画面可见：操作台（右侧下方）。",
    directionalClear: null,
    rejectionReason: null,
    evidence: [],
    warnings: [],
    ...overrides,
  };
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
});

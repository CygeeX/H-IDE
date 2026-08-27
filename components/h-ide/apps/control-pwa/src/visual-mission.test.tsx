import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AssistantDraft, VisualMissionRecord } from "./contracts";
import { VisualMissionCard } from "./visual-mission-card";

function draft(status = "confirmed"): AssistantDraft {
  return {
    draftId: "visual-draft-1",
    status,
    plan: {
      planType: "visual_mission",
      planId: "visual-draft-1",
      robotId: "robot-1",
      expiresAt: "2099-01-01T00:00:00Z",
      mission: {
        missionId: "visual-draft-1",
        robotId: "robot-1",
        instruction: "走到黄色箱子旁边",
        locale: "zh-CN",
        targetDescription: "黄色箱子",
        requestedSourceMode: "synthetic",
        fusionMode: "vision_only",
        stages: [{ stageId: "approach-1", kind: "approach", targetDescription: "黄色箱子", goalPurpose: "approach" }],
        robotBootId: "simulator-boot-1",
        localizationEpochId: "epoch-1",
        sensorIds: ["head_rgbd"],
        allowedTargetCategories: ["box"],
        unsupportedCapabilities: ["path_planning", "manipulation"],
        maximumGoalDistanceM: 5,
        maximumCumulativeDistanceM: 20,
        maximumDurationSeconds: 600,
        maximumSpeedMps: 0.2,
        vlmProvider: "FakeVlmProvider",
        vlmModel: "fake-fixture-v1",
        promptVersion: "visual-grounding-v1",
        visualAlgorithmVersion: "rgbd-median-v1",
        approachPolicyVersion: "v1",
        approachPolicySha256: "a".repeat(64),
        reviewExpiresAt: "2099-01-01T00:00:00Z",
      },
    },
  };
}

function mission(mode: VisualMissionRecord["mode"] = "lab_execute"): VisualMissionRecord {
  const plan = draft().plan;
  if (plan.planType !== "visual_mission") throw new Error("fixture is invalid");
  return {
    missionId: "visual-draft-1",
    draftId: "visual-draft-1",
    robotId: "robot-1",
    status: "approved",
    plan,
    state: { currentStageIndex: 0, cancelRequested: false, receipts: [] },
    stages: [],
    mode,
    releaseState: "code_complete_robot_agent_pending",
    motion: mode === "lab_execute" ? "simulated" : "not_executed",
  };
}

function ambiguousMission(): VisualMissionRecord {
  return {
    ...mission(),
    status: "needs_input",
    stages: [{
      stageId: "approach-1",
      attempt: 1,
      status: "needs_input",
      visualResolutionId: "vr-ambiguous-1",
      state: {
        ambiguityCandidates: [
          { candidateId: "table-1", label: "桌子", box: [1, 2, 30, 40], score: 0.91 },
          { candidateId: "table-2", label: "桌子", box: [31, 2, 60, 40], score: 0.80 },
        ],
      },
    }],
  };
}

const robot = { robotId: "robot-1", status: "online" as const, connected: true, safetyMode: "normal" as const };

afterEach(cleanup);

describe("visual mission card", () => {
  it("separates review from simulated target dispatch", () => {
    const onRun = vi.fn();
    render(<VisualMissionCard
      draft={draft()}
      robot={robot}
      mission={mission()}
      busy={false}
      expired={false}
      reviewable={false}
      expiry="5 分钟"
      onConfirm={vi.fn()}
      onRun={onRun}
      onCancel={vi.fn()}
      onReplan={vi.fn()}
    />);
    expect(screen.getByText(/运动为模拟/)).toBeInTheDocument();
    expect(screen.getByText(/确认后生成坐标/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "开始模拟目标解析与发送" }));
    expect(onRun).toHaveBeenCalledTimes(1);
  });

  it("keeps observe mode non-executable and exposes an accessible evidence drawer", () => {
    render(<VisualMissionCard
      draft={draft()}
      robot={robot}
      mission={mission("observe")}
      busy={false}
      expired={false}
      reviewable={false}
      expiry="5 分钟"
      onConfirm={vi.fn()}
      onRun={vi.fn()}
      onCancel={vi.fn()}
      onReplan={vi.fn()}
    />);
    expect(screen.queryByRole("button", { name: "开始模拟目标解析与发送" })).not.toBeInTheDocument();
    expect(screen.getByText(/observe 模式只允许审核/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看视觉证据" }));
    expect(screen.getByRole("dialog", { name: "视觉任务依据" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "关闭视觉证据" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("offers immutable ambiguity candidates without dispatching until selection", () => {
    const onSelectCandidate = vi.fn();
    render(<VisualMissionCard
      draft={draft()}
      robot={robot}
      mission={ambiguousMission()}
      busy={false}
      expired={false}
      reviewable={false}
      expiry="5 分钟"
      onConfirm={vi.fn()}
      onRun={vi.fn()}
      onSelectCandidate={onSelectCandidate}
      onCancel={vi.fn()}
      onReplan={vi.fn()}
    />);

    expect(screen.queryByRole("button", { name: "开始模拟目标解析与发送" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /选择候选 1：桌子/ }));
    expect(onSelectCandidate).toHaveBeenCalledWith("table-1");
  });
});

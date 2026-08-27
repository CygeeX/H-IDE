import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  AuthPrincipal,
  RobotState,
  SceneInspectionSession,
  SceneLandmark,
  SceneMemoryCapabilities,
} from "./contracts";
import {
  roughLandmarkLocation,
  SceneMemoryWorkspace,
  type SceneMemoryWorkspaceProps,
} from "./scene-memory-workspace";

const principal: AuthPrincipal = {
  userId: "scene-user",
  username: "scene-user",
  displayName: "场景用户",
  role: "USER",
  siteId: "office",
  authenticationMethod: "session",
  scopes: ["api:access", "robots:read", "assistant:use", "scene-memory:use"],
};

const robots: RobotState[] = [{
  robotId: "g1-sim",
  name: "办公室机器人",
  status: "online",
  connected: true,
  safetyMode: "normal",
  details: { localizationEpochId: "office-epoch-secret", semanticProvider: "qwen" },
}];

const capabilities: SceneMemoryCapabilities = {
  mode: "SCENE_TEACH",
  stateFlow: ["stopped", "captured", "named", "confirmed"],
  semanticProvider: "qwen",
  allowedSources: ["qwen_vlm", "rgbd", "tf"],
  retiredSources: ["pcd_annotation", "oracle", "foundationpose", "fake"],
  authority: "observation_only",
  executionEligible: false,
  navigationDispatch: "forbidden",
  humanConfirmationRequired: true,
};

const inspection: SceneInspectionSession = {
  inspectionId: "inspection-office",
  ownerUserId: "scene-user",
  siteId: "office",
  robotId: "g1-sim",
  purpose: "办公室巡检",
  mapFrameId: "map",
  localizationEpochId: "office-epoch-secret",
  status: "open",
  recordVersion: 2,
  createdAt: "2026-08-19T08:00:00Z",
  updatedAt: "2026-08-19T08:05:00Z",
  mode: "SCENE_TEACH",
  motionAllowed: false,
  navigationDispatched: false,
  checkpoints: [],
};

function landmark(status: "candidate" | "confirmed" | "superseded", name: string, index: number): SceneLandmark {
  return {
    landmarkId: `landmark-${index}`,
    rootLandmarkId: `landmark-${index}`,
    revision: 1,
    ownerUserId: "scene-user",
    siteId: "office",
    inspectionId: inspection.inspectionId,
    checkpointId: `checkpoint-${index}`,
    canonicalName: name,
    description: status === "candidate" ? "走廊尽头的饮水机" : null,
    status,
    sourceKind: "SCENE_TEACH",
    authority: "observation_only",
    executionEligible: false,
    navigationDispatched: false,
    aliases: status === "superseded" ? [] : [{
      aliasId: `alias-${index}`,
      landmarkId: `landmark-${index}`,
      alias: `${name}别名`,
      status: "active",
      createdAt: "2026-08-19T08:04:00Z",
      updatedAt: "2026-08-19T08:04:00Z",
    }],
    currentObservation: {
      observationId: `observation-${index}`,
      landmarkId: `landmark-${index}`,
      inspectionId: inspection.inspectionId,
      checkpointId: `checkpoint-${index}`,
      status,
      mapPose: { frameId: "map", x: 2, y: 1, z: 0, yaw: 0, observedAt: "2026-08-19T08:04:00Z" },
      covariance: Array(36).fill(0.05),
      localizationEpochId: "office-epoch-secret",
      robotPose: { frameId: "map", x: 0, y: 0, z: 0, yaw: 0, observedAt: "2026-08-19T08:04:00Z" },
      frameId: "rgb-1",
      frameSha256: "a".repeat(64),
      evidenceSha256: "b".repeat(64),
      evidenceRef: "evidence/private-reference",
      vlmProvider: "qwen",
      vlmModel: "qwen3-vl-plus",
      bbox: { xMin: 0.1, yMin: 0.2, xMax: 0.5, yMax: 0.8 },
      depth: { sensorId: "head-depth", samplingMethod: "median_mask" },
      tfProvenance: { sourceFrame: "head_camera", targetFrame: "map" },
      confidence: 0.91,
      supportCount: 3,
      authority: "observation_only",
      executionEligible: false,
      recordVersion: 1,
      createdAt: "2026-08-19T08:04:00Z",
      updatedAt: "2026-08-19T08:04:00Z",
    },
    recordVersion: 1,
    createdAt: "2026-08-19T08:04:00Z",
    updatedAt: "2026-08-19T08:04:00Z",
  };
}

function apiFixture(options: { inspections?: SceneInspectionSession[]; landmarks?: SceneLandmark[] } = {}) {
  let inspectionValues = options.inspections ?? [inspection];
  const landmarkValues = options.landmarks ?? [
    landmark("candidate", "饮水机", 1),
    landmark("confirmed", "会议桌", 2),
    landmark("superseded", "旧前台", 3),
  ];
  const apiFetchMock = vi.fn(async (path: string, init?: RequestInit) => {
    if (path.endsWith("/capabilities")) return capabilities;
    if (path.includes("/inspections?") && !init?.method) return inspectionValues;
    if (path.includes("/landmarks?") && !init?.method) return landmarkValues;
    if (path === "/api/v2/scene-memory/inspections" && init?.method === "POST") {
      const created = { ...inspection, inspectionId: "inspection-created", recordVersion: 1 };
      inspectionValues = [created];
      return created;
    }
    if (path.includes("/inspections/")) {
      const id = path.split("/inspections/")[1]?.split("/")[0];
      return inspectionValues.find((item) => item.inspectionId === id) ?? inspection;
    }
    if (path.includes("/landmarks/") && init?.method) return {};
    throw new Error(`unexpected API request: ${path}`);
  });
  return {
    apiFetch: apiFetchMock as unknown as SceneMemoryWorkspaceProps["apiFetch"],
    apiFetchMock,
    apiFetchBlob: vi.fn(async () => new Blob(["image"], { type: "image/jpeg" })),
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("scene teaching workspace", () => {
  it("turns coordinates into a user-readable relative location", () => {
    expect(roughLandmarkLocation(
      { frameId: "map", x: 2, y: 1, z: 0, yaw: 0 },
      { frameId: "map", x: 0, y: 0, z: 0, yaw: 0 },
    )).toBe("在本次记录位置前方偏左侧约 2.0 米");
    expect(roughLandmarkLocation(null, null)).toBe("已保存大致位置");
  });

  it("shows the four-step safe flow and hides coordinates from an ordinary user", async () => {
    const fixture = apiFixture();
    render(<SceneMemoryWorkspace
      principal={principal}
      robots={robots}
      selectedRobotId="g1-sim"
      developerMode={false}
      apiFetch={fixture.apiFetch}
      apiFetchBlob={fixture.apiFetchBlob}
      onBackToChat={vi.fn()}
    />);

    expect(await screen.findByText("移动并停稳")).toBeInTheDocument();
    expect(screen.getByText("观察场景")).toBeInTheDocument();
    expect(screen.getByText("命名并确认")).toBeInTheDocument();
    expect(screen.getByText("饮水机")).toBeInTheDocument();
    expect(screen.getByText("会议桌")).toBeInTheDocument();
    expect(screen.getByText("旧前台")).toBeInTheDocument();
    expect(screen.getByText(/当前缺少服务端一键采集编排端点/)).toBeInTheDocument();
    expect(screen.getAllByText(/不会直接驱动机器人/).length).toBeGreaterThan(0);
    expect(screen.queryByText("开发者来源、坐标与定位纪元")).not.toBeInTheDocument();
    expect(screen.queryByText("office-epoch-secret")).not.toBeInTheDocument();
    expect(screen.queryByText(/x 2\.000/)).not.toBeInTheDocument();
  });

  it("requires human confirmation before accepting, changing aliases, or invalidating a candidate", async () => {
    const candidate = landmark("candidate", "饮水机", 1);
    const fixture = apiFixture({ landmarks: [candidate] });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.spyOn(window, "prompt").mockReturnValue("识别错误");
    render(<SceneMemoryWorkspace
      principal={principal}
      robots={robots}
      selectedRobotId="g1-sim"
      developerMode={false}
      apiFetch={fixture.apiFetch}
      apiFetchBlob={fixture.apiFetchBlob}
      onBackToChat={vi.fn()}
    />);

    fireEvent.click(await screen.findByRole("button", { name: "确认记住" }));
    await waitFor(() => expect(fixture.apiFetchMock).toHaveBeenCalledWith(
      "/api/v2/scene-memory/landmarks/landmark-1/confirm",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ expectedRecordVersion: 1 }) }),
    ));
    expect(confirm).toHaveBeenCalled();

    await waitFor(() => expect(screen.getByRole("button", { name: "移除别名 饮水机别名" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "移除别名 饮水机别名" }));
    await waitFor(() => expect(fixture.apiFetchMock).toHaveBeenCalledWith(
      "/api/v2/scene-memory/landmarks/landmark-1/aliases/alias-1",
      { method: "DELETE" },
    ));

    await waitFor(() => expect(screen.getByRole("button", { name: "作废" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "作废" }));
    await waitFor(() => expect(fixture.apiFetchMock).toHaveBeenCalledWith(
      "/api/v2/scene-memory/landmarks/landmark-1/invalidate",
      expect.objectContaining({ method: "POST" }),
    ));
    expect(confirm).toHaveBeenCalledTimes(3);
  });

  it("creates an inspection only with the Robot Agent localization epoch", async () => {
    const fixture = apiFixture({ inspections: [], landmarks: [] });
    render(<SceneMemoryWorkspace
      principal={principal}
      robots={robots}
      selectedRobotId="g1-sim"
      developerMode={false}
      apiFetch={fixture.apiFetch}
      apiFetchBlob={fixture.apiFetchBlob}
      onBackToChat={vi.fn()}
    />);

    fireEvent.click(await screen.findByRole("button", { name: "开始新的巡检" }));
    await waitFor(() => expect(fixture.apiFetchMock).toHaveBeenCalledWith(
      "/api/v2/scene-memory/inspections",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          robotId: "g1-sim",
          localizationEpochId: "office-epoch-secret",
          mapFrameId: "map",
          purpose: "办公室巡检",
        }),
      }),
    ));
  });

  it("reveals provenance and exact map coordinates only in developer mode", async () => {
    const fixture = apiFixture({ landmarks: [landmark("confirmed", "会议桌", 2)] });
    render(<SceneMemoryWorkspace
      principal={{ ...principal, role: "DEVELOPER" }}
      robots={robots}
      selectedRobotId="g1-sim"
      developerMode
      apiFetch={fixture.apiFetch}
      apiFetchBlob={fixture.apiFetchBlob}
      onBackToChat={vi.fn()}
    />);

    expect(await screen.findByText("开发者来源、坐标与定位纪元")).toBeInTheDocument();
    expect(screen.getByText(/x 2\.000 · y 1\.000/)).toBeInTheDocument();
    expect(screen.getAllByText("office-epoch-secret")).toHaveLength(2);
    expect(screen.getByText("qwen · qwen3-vl-plus")).toBeInTheDocument();
  });
});

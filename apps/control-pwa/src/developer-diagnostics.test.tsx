import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DeveloperDiagnostics } from "./developer-diagnostics";

afterEach(cleanup);


function snapshot() {
  return {
    schemaVersion: "h-ide.developer-diagnostics/1" as const,
    observedAt: "2026-08-19T12:00:00+00:00",
    target: { robotId: "g1-sim", kind: "simulation" as const },
    summary: {
      status: "degraded" as const,
      classification: "physical_fault" as const,
      ready: false,
      authorizationRequired: false,
      motionAuthorized: false,
      holdReason: null,
      fault: "gait tracking envelope exceeded",
      processAliveIsNotReadiness: true as const,
      detail: "One or more required execution dependencies are not ready",
    },
    release: {
      hIde: "0.1.0",
      agent: "0.2.0",
      vision: "0.1.0",
      simulation: "office-v2",
      rosDistro: "jazzy",
    },
    components: {
      hIde: { configured: true, alive: true, ready: true, detail: "authenticated diagnostics BFF is responding" },
      agent: { configured: true, alive: true, ready: true, latencyMs: 4, reasons: [] },
      vision: {
        configured: true,
        alive: true,
        ready: true,
        latencyMs: 18,
        provider: "qwen",
        model: "qwen3-vl-plus",
        segmentation: "sam2",
        profile: "pure_vlm_product_profile",
        latestAnalysis: {
          recordedAt: "2026-08-19T11:59:00+00:00",
          latencyMs: 1320,
          status: "completed",
          evidenceCount: 3,
          evidenceKinds: ["rgb_input", "semantic_json", "sam2_mask"],
        },
      },
      gateway: {
        configured: true,
        alive: true,
        ready: true,
        connected: true,
        identityMatches: true,
        protocol: "robot-agent.v1",
        executionBackend: "websocket_inbound",
        controlState: "SAFE_HOLD",
        observationStatus: "USABLE",
      },
      robotAgent: {
        configured: true,
        alive: true,
        ready: false,
        identityMatches: true,
        commandReadiness: { ready: false, mode: "BLOCKED", reason: "required_dependency_unavailable" },
        transport: { ready: true, reason: null },
        control: {
          ready: false,
          state: "SAFE_HOLD",
          mode: "BLOCKED",
          standby: false,
          leaseValid: false,
          leaseRemainingMs: null,
          activeCommandCount: 1,
          reason: "active_commands_without_valid_lease",
        },
      },
      nav2: {
        configured: true,
        alive: true,
        ready: false,
        reason: "one_or_more_nav2_actions_unavailable",
        actions: {
          "navigation.navigate_to_pose": { ready: false, reason: "action_server_missing" },
          "navigation.rotate": { ready: true, reason: null },
        },
      },
      localization: {
        configured: true,
        poseReady: true,
        poseAgeMs: 34,
        poseMaximumAgeMs: 500,
        tfReady: null,
        tfReason: "not_exposed_by_robot_agent_diagnostics_v1",
        reason: null,
      },
      observation: {
        configured: true,
        ready: true,
        status: "USABLE",
        reason: null,
        captureAgeMs: 42,
        healthAgeMs: 21,
        gatewayAckAgeMs: 17,
        maximumAgeMs: 750,
      },
      sonic: {
        configured: true,
        alive: true,
        ready: false,
        motionReady: false,
        mode: "FAULT",
        nav2State: "fault_latched",
        controlClock: "simulation_time_50hz",
        fault: "gait tracking envelope exceeded",
      },
      plant: {
        configured: true,
        alive: true,
        ready: false,
        status: "failed",
        captureSequence: 202,
        rgbReady: true,
        depthReady: true,
        rgbFrames: 202,
        depthFrames: 202,
        supportMode: "safety_stop",
        safetyStopLatched: true,
        fault: "MuJoCo physics time rolled back",
      },
    },
  };
}


function standbySnapshot() {
  const base = snapshot();
  return {
    ...base,
    summary: {
      status: "standby" as const,
      classification: "authorization_hold" as const,
      ready: true,
      authorizationRequired: true,
      motionAuthorized: false,
      holdReason: "plant safety_stop requires explicit control.acquire rearm",
      fault: null,
      processAliveIsNotReadiness: true as const,
      detail: "Physical safe-stop is confirmed; execution requires explicit authorization",
    },
    components: {
      ...base.components,
      robotAgent: {
        ...base.components.robotAgent,
        ready: true,
        standby: true,
        commandReadiness: { ready: true, mode: "STANDBY", reason: null },
        control: {
          ready: true,
          state: "SAFE_HOLD",
          mode: "STANDBY",
          active: false,
          standby: true,
          canAcquire: true,
          authorityRevoked: true,
          safeStopState: "confirmed",
          safeStopEpoch: 7,
          leaseValid: false,
          leaseRemainingMs: null,
          activeCommandCount: 0,
          reason: "standby_waiting_for_control_acquire",
        },
        safeStop: {
          ready: true,
          authorityRevoked: true,
          state: "confirmed",
          epoch: 7,
          consistent: true,
        },
      },
      nav2: {
        ...base.components.nav2,
        ready: true,
        reason: null,
        actions: {
          "navigation.navigate_to_pose": { ready: true, reason: null },
          "navigation.rotate": { ready: true, reason: null },
        },
      },
      sonic: {
        ...base.components.sonic,
        ready: true,
        standby: true,
        motionReady: true,
        mode: "IDLE",
        nav2State: "safety_stop",
        controlClock: "simulation_time_50hz_motion_ready",
        staticHoldActive: true,
        commandZero: true,
        navigationTaskActive: false,
        stopPending: false,
        supportMode: "safety_stop",
        supportSafetyLatched: true,
        faultTracePresent: false,
        faultTraceSampleCount: 0,
        lastErrorKnown: true,
        lastError: "plant safety_stop requires explicit control.acquire rearm",
        fault: null,
      },
      plant: {
        ...base.components.plant,
        ready: true,
        standby: true,
        status: "running",
        supportMode: "safety_stop",
        safetyStopLatched: true,
        faultKnown: true,
        fault: null,
      },
    },
  };
}


describe("DeveloperDiagnostics", () => {
  it("renders confirmed SAFE_HOLD as healthy authorization standby", async () => {
    const apiFetch = vi.fn().mockResolvedValue(standbySnapshot());
    render(<DeveloperDiagnostics apiFetch={apiFetch} robotId="g1-sim" pollIntervalMs={60_000} />);

    expect(await screen.findByText("安全待命，执行前需授权")).toBeInTheDocument();
    expect(screen.getAllByText("STANDBY").length).toBeGreaterThan(0);
    expect(screen.getByText("plant safety_stop requires explicit control.acquire rearm")).toBeInTheDocument();
    expect(screen.getAllByText("standby_waiting_for_control_acquire").length).toBeGreaterThan(0);
    expect(screen.getByText("已撤销")).toBeInTheDocument();
    expect(screen.getByText("全零")).toBeInTheDocument();
    expect(screen.getByText("已锁存")).toBeInTheDocument();
    expect(screen.queryByText("Plant latched fault")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /授权|获取控制|执行|移动/ })).not.toBeInTheDocument();
  });

  it("renders precise readiness failures and only calls the authenticated BFF", async () => {
    const apiFetch = vi.fn().mockResolvedValue(snapshot());
    render(<DeveloperDiagnostics apiFetch={apiFetch} robotId="g1-sim" pollIntervalMs={60_000} />);

    expect(await screen.findByText("检测到物理执行故障")).toBeInTheDocument();
    expect(apiFetch).toHaveBeenCalledWith("/api/v2/developer/diagnostics?robotId=g1-sim");
    expect(screen.getByText("required_dependency_unavailable")).toBeInTheDocument();
    expect(screen.getByText("active_commands_without_valid_lease")).toBeInTheDocument();
    expect(screen.getByText("action_server_missing")).toBeInTheDocument();
    expect(screen.getAllByText("gait tracking envelope exceeded").length).toBeGreaterThan(0);
    expect(screen.getByText("MuJoCo physics time rolled back")).toBeInTheDocument();
    expect(screen.getByText("qwen / qwen3-vl-plus")).toBeInTheDocument();
    expect(screen.getByText("1320 ms")).toBeInTheDocument();
    expect(screen.getByText("rgb_input · semantic_json · sam2_mask")).toBeInTheDocument();
    expect(screen.getByText("Robot Agent v1 未暴露")).toBeInTheDocument();
    expect(screen.getByText(/process alive ≠ ready/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /故障注入|急停|执行|移动/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("keeps the last snapshot visible when a later read-only refresh fails", async () => {
    const apiFetch = vi.fn()
      .mockResolvedValueOnce(snapshot())
      .mockRejectedValueOnce(new Error("diagnostics timeout"));
    render(<DeveloperDiagnostics apiFetch={apiFetch} robotId="g1-sim" pollIntervalMs={60_000} />);
    expect(await screen.findByText("检测到物理执行故障")).toBeInTheDocument();

    screen.getByRole("button", { name: "刷新只读状态" }).click();
    await waitFor(() => expect(screen.getByText("刷新失败，当前显示上一份快照")).toBeInTheDocument());
    expect(screen.getByText("diagnostics timeout")).toBeInTheDocument();
    expect(screen.getByText("Robot Agent Gateway")).toBeInTheDocument();
  });
});

import { describe, expect, it, vi } from "vitest";
import { loadConfig } from "../src/config.js";
import { planningEnvelopeSchema } from "../src/contracts.js";
import { createAgentSystem } from "../src/mastra/agents.js";
import { MastraPlanningRuntime } from "../src/planning/mastra-runtime.js";
import { envelope, manipulationFragment, motionFragment } from "./fixtures.js";

describe("Mastra agent system", () => {
  it("keeps readiness degraded for weak or placeholder internal tokens", () => {
    expect(loadConfig({ MASTRA_INTERNAL_TOKEN: "dev-internal-token" }).internalTokenConfigured)
      .toBe(false);
    expect(loadConfig({ MASTRA_INTERNAL_TOKEN: "short-token" }).internalTokenConfigured)
      .toBe(false);
    expect(loadConfig({
      MASTRA_INTERNAL_TOKEN: "0123456789abcdef0123456789abcdef",
    }).internalTokenConfigured).toBe(true);
  });

  it("reports a missing credential for configured OpenAI models", () => {
    const environment = {
      MASTRA_INTERNAL_TOKEN: "test-internal-token",
      SUPERVISOR_MODEL: "openai/test-model",
      STRUCTURING_MODEL: "openai/test-model",
    };
    expect(loadConfig(environment).modelCredentialReasons).toContain(
      "OPENAI_API_KEY is required by configured models",
    );
    expect(loadConfig({ ...environment, OPENAI_API_KEY: "test-key" }).modelCredentialReasons)
      .toEqual([]);
  });

  it("reports a missing credential for configured DeepSeek models", () => {
    const environment = {
      MASTRA_INTERNAL_TOKEN: "test-internal-token",
      SUPERVISOR_MODEL: "deepseek/deepseek-v4-flash",
      WORKER_MODEL: "deepseek/deepseek-v4-flash",
      STRUCTURING_MODEL: "deepseek/deepseek-v4-flash",
    };
    expect(loadConfig(environment).modelCredentialReasons).toContain(
      "DEEPSEEK_API_KEY is required by configured models",
    );
    expect(loadConfig({ ...environment, DEEPSEEK_API_KEY: "test-key" }).modelCredentialReasons)
      .toEqual([]);
  });

  it("reports a missing DashScope credential for Alibaba China models", () => {
    const environment = {
      MASTRA_INTERNAL_TOKEN: "test-internal-token",
      SUPERVISOR_MODEL: "alibaba-cn/qwen3.8-max",
      WORKER_MODEL: "alibaba-cn/qwen3.8-max",
      STRUCTURING_MODEL: "alibaba-cn/qwen3.8-max",
    };
    expect(loadConfig(environment).modelCredentialReasons).toContain(
      "DASHSCOPE_API_KEY is required by configured models",
    );
    expect(loadConfig({ ...environment, DASHSCOPE_API_KEY: "test-key" }).modelCredentialReasons)
      .toEqual([]);
  });

  it("rejects external workers that collide with built-ins or each other", () => {
    const worker = {
      id: "motion-worker",
      name: "Collision",
      description: "Should not replace the local worker",
      url: "https://agents.example.com",
    };
    expect(() => loadConfig({
      A2A_ALLOWED_ORIGINS: "https://agents.example.com",
      A2A_WORKERS_JSON: JSON.stringify([worker]),
    })).toThrow(/reserved|duplicate|collision/i);
    expect(() => loadConfig({
      A2A_ALLOWED_ORIGINS: "https://agents.example.com",
      A2A_WORKERS_JSON: JSON.stringify([
        { ...worker, id: "inspection-worker" },
        { ...worker, id: "inspection-worker" },
      ]),
    })).toThrow(/duplicate|unique/i);
  });

  it("keeps the supervisor as a pure router without callable worker tools", async () => {
    const config = loadConfig({
      MASTRA_INTERNAL_TOKEN: "test-internal-token",
      SUPERVISOR_MODEL: "openai/test-model",
      STRUCTURING_MODEL: "openai/test-model",
      PLANNING_DATABASE_URL: "file::memory:",
      MASTRA_MEMORY_URL: "file::memory:",
    });

    const system = createAgentSystem(config);

    expect(await system.agents.supervisorAgent.listAgents()).toEqual({});
    expect(Object.keys(system.agents)).toEqual([
      "supervisorAgent",
      "motionWorker",
      "manipulationWorker",
      "visionWorker",
      "structuringAgent",
      "missionExecutiveAgent",
    ]);
    expect(Object.keys(await system.agents.motionWorker.listTools())).toContain(
      "resolveSpatialTarget",
    );
    expect(Object.keys(await system.agents.manipulationWorker.listTools())).not.toContain(
      "resolveSpatialTarget",
    );
    expect(Object.keys(await system.agents.visionWorker.listTools())).toEqual([]);
  });

  it("describes each available worker in the supervisor classification prompt", async () => {
    const supervisor = {
      generate: vi.fn().mockResolvedValue({
        object: {
          status: "route",
          selectedAgentIds: ["inspection-worker"],
          rationale: "inspection requested",
        },
      }),
    };
    const runtime = new MastraPlanningRuntime(
      supervisor as never,
      { generate: vi.fn() } as never,
      new Map(),
      new Map(),
      new Map([["inspection-worker", "Plans visual inspection tasks."]]),
    );

    await runtime.classify(
      planningEnvelopeSchema.parse(envelope()),
      new AbortController().signal,
    );

    expect(supervisor.generate.mock.calls[0]![0]).toContain(
      "inspection-worker: Plans visual inspection tasks.",
    );
  });

  it("deterministically routes a visible target without a bound PCD scene", async () => {
    const supervisor = { generate: vi.fn() };
    const runtime = new MastraPlanningRuntime(
      supervisor as never,
      { generate: vi.fn() } as never,
      new Map(),
      new Map(),
      new Map([
        ["motion-worker", "Plans motion commands."],
        ["vision-worker", "Creates a visual target mission."],
      ]),
    );

    const result = await runtime.classify(
      planningEnvelopeSchema.parse(envelope({
        taskRequest: { text: "找到银白色的机器人并去机器人的旁边" },
      })),
      new AbortController().signal,
    );

    expect(result).toMatchObject({
      status: "route",
      selectedAgentIds: ["vision-worker"],
    });
    expect(supervisor.generate).not.toHaveBeenCalled();
  });

  it("keeps unsupported visual multi-action requests under supervisor control", async () => {
    const supervisor = {
      generate: vi.fn().mockResolvedValue({
        object: {
          status: "cannot_plan",
          assistantMessage: "当前版本不支持抓取。",
          reason: "manipulation is unavailable",
          missingCapabilities: ["manipulation"],
        },
      }),
    };
    const runtime = new MastraPlanningRuntime(
      supervisor as never,
      { generate: vi.fn() } as never,
      new Map(),
      new Map(),
      new Map([["vision-worker", "Creates a visual target mission."]]),
    );

    const result = await runtime.classify(
      planningEnvelopeSchema.parse(envelope({
        taskRequest: { text: "找到银白色机器人并抓起来" },
      })),
      new AbortController().signal,
    );

    expect(result.status).toBe("cannot_plan");
    expect(supervisor.generate).toHaveBeenCalledTimes(1);
  });

  it("leaves open-ended camera questions to the supervisor", async () => {
    const supervisor = {
      generate: vi.fn().mockResolvedValue({
        object: {
          status: "needs_input",
          assistantMessage: "请说明要在画面中寻找的目标。",
          question: "你想寻找什么目标？",
        },
      }),
    };
    const runtime = new MastraPlanningRuntime(
      supervisor as never,
      { generate: vi.fn() } as never,
      new Map(),
      new Map(),
      new Map([["vision-worker", "Creates a visual target mission."]]),
    );

    const result = await runtime.classify(
      planningEnvelopeSchema.parse(envelope({
        taskRequest: { text: "看看当前画面" },
      })),
      new AbortController().signal,
    );

    expect(result.status).toBe("needs_input");
    expect(supervisor.generate).toHaveBeenCalledTimes(1);
  });

  it.each(["去厨房旁边等候", "找到厨房并在那里等待"])(
    "does not steal a named PCD location from the spatial supervisor route: %s",
    async (text) => {
      const supervisor = {
        generate: vi.fn().mockResolvedValue({
          object: {
            status: "route",
            selectedAgentIds: ["motion-worker"],
            rationale: "named scene location",
          },
        }),
      };
      const runtime = new MastraPlanningRuntime(
        supervisor as never,
        { generate: vi.fn() } as never,
        new Map(),
        new Map(),
        new Map([
          ["motion-worker", "Plans motion commands."],
          ["vision-worker", "Creates a visual target mission."],
        ]),
      );

      const result = await runtime.classify(
        planningEnvelopeSchema.parse(envelope({
          taskRequest: { text },
        })),
        new AbortController().signal,
      );

      expect(result).toMatchObject({ selectedAgentIds: ["motion-worker"] });
      expect(supervisor.generate).toHaveBeenCalledTimes(1);
    },
  );

  it("retries a transient inference transport failure with the same planning input", async () => {
    const supervisor = {
      generate: vi.fn()
        .mockRejectedValueOnce(
          new Error(
            "Cannot connect to API: Client network socket disconnected before secure TLS connection was established",
          ),
        )
        .mockResolvedValue({
          object: {
            status: "route",
            selectedAgentIds: ["inspection-worker"],
            rationale: "inspection requested",
          },
        }),
    };
    const runtime = new MastraPlanningRuntime(
      supervisor as never,
      { generate: vi.fn() } as never,
      new Map(),
      new Map(),
      new Map([["inspection-worker", "Plans visual inspection tasks."]]),
    );

    const result = await runtime.classify(
      planningEnvelopeSchema.parse(envelope()),
      new AbortController().signal,
    );

    expect(result.status).toBe("route");
    expect(supervisor.generate).toHaveBeenCalledTimes(2);
    expect(supervisor.generate.mock.calls[1]![0]).toBe(supervisor.generate.mock.calls[0]![0]);
  });

  it("does not retry a non-transport inference failure", async () => {
    const supervisor = {
      generate: vi.fn().mockRejectedValue(new Error("structured output schema is invalid")),
    };
    const runtime = new MastraPlanningRuntime(
      supervisor as never,
      { generate: vi.fn() } as never,
      new Map(),
      new Map(),
      new Map([["inspection-worker", "Plans visual inspection tasks."]]),
    );

    await expect(runtime.classify(
      planningEnvelopeSchema.parse(envelope()),
      new AbortController().signal,
    )).rejects.toThrow(/schema/);
    expect(supervisor.generate).toHaveBeenCalledTimes(1);
  });

  it("limits a built-in manipulation worker to its owned command catalog", async () => {
    const fragment = manipulationFragment();
    const worker = {
      generate: vi.fn().mockResolvedValue({
        object: {
          ...fragment,
          steps: fragment.steps.map((step) => ({
            ...step,
            spatialIntent: null,
          })),
        },
      }),
    };
    const runtime = new MastraPlanningRuntime(
      { generate: vi.fn() } as never,
      { generate: vi.fn() } as never,
      new Map([["manipulation-worker", worker as never]]),
      new Map(),
      new Map([["manipulation-worker", "Plans manipulation commands."]]),
    );

    await runtime.runWorker(
      "manipulation-worker",
      planningEnvelopeSchema.parse(envelope()),
      new AbortController().signal,
    );

    const prompt = worker.generate.mock.calls[0]![0] as string;
    const scopedEnvelope = JSON.parse(prompt.split("\n").at(-1)!);
    expect(scopedEnvelope.commandCatalog.map((item: { commandKey: string }) => item.commandKey))
      .toEqual(["gripper.open"]);
    expect(scopedEnvelope.capabilitySnapshot.capabilities).toEqual(["gripper.open"]);
  });

  it("hides catalog commands that the current robot does not advertise", async () => {
    const fragment = manipulationFragment();
    const worker = { generate: vi.fn().mockResolvedValue({
      object: {
        ...fragment,
        steps: fragment.steps.map((step) => ({
          ...step,
          spatialIntent: null,
        })),
      },
    }) };
    const runtime = new MastraPlanningRuntime(
      { generate: vi.fn() } as never, { generate: vi.fn() } as never,
      new Map([["manipulation-worker", worker as never]]), new Map(),
      new Map([["manipulation-worker", "Plans manipulation commands."]]),
    );

    await runtime.runWorker(
      "manipulation-worker",
      planningEnvelopeSchema.parse(envelope({
        capabilitySnapshot: {
          agent: "robot-agent", protocol: "robot-agent.v1", schema: { type: "object" },
          capabilities: ["gripper.open"], safetyMode: "normal", robotId: "robot-1",
        },
      })),
      new AbortController().signal,
    );

    const prompt = worker.generate.mock.calls[0]![0] as string;
    const scopedEnvelope = JSON.parse(prompt.split("\n").at(-1)!);
    expect(scopedEnvelope.commandCatalog.map((item: { commandKey: string }) => item.commandKey))
      .toEqual(["gripper.open"]);
  });

  it("rejects a built-in manipulation worker command outside its scope", async () => {
    const fragment = manipulationFragment();
    fragment.steps[0]!.commandKey = "move.forward";
    const workerFragment = {
      ...fragment,
      steps: fragment.steps.map((step) => ({
        ...step,
        spatialIntent: null,
      })),
    };
    const runtime = new MastraPlanningRuntime(
      { generate: vi.fn() } as never,
      { generate: vi.fn() } as never,
      new Map([["manipulation-worker", { generate: vi.fn().mockResolvedValue({ object: workerFragment }) } as never]]),
      new Map(),
      new Map([["manipulation-worker", "Plans manipulation commands."]]),
    );

    await expect(runtime.runWorker(
      "manipulation-worker",
      planningEnvelopeSchema.parse(envelope()),
      new AbortController().signal,
    )).rejects.toThrow(/scope|owned|command/i);
  });

  it("materializes only the static-PCD intent explicitly selected by the worker", async () => {
    const sceneContext = {
      sceneRef: { sceneId: "scene-demo", revision: 1 },
      bounds: { min: { x: -20, y: -20, z: -1 }, max: { x: 20, y: 20, z: 3 } },
      annotations: [{
        locationId: "kitchen",
        name: "厨房",
        aliases: ["烹饪区"],
        polygonXY: [{ x: 2, y: 2 }, { x: 6, y: 2 }, { x: 6, y: 6 }],
        zMin: 0,
        zMax: 2.5,
        entryPose: { frameId: "map" as const, x: 2, y: 3, z: 0, yaw: 0 },
      }],
      currentPose: null,
      selectedPoint: null,
    };
    const worker = {
      generate: vi.fn().mockResolvedValue({
        object: {
          agentId: "motion-worker",
          contribution: "motion",
          summary: "选择厨房静态候选点",
          steps: [{
            stepId: "go-kitchen",
            commandKey: "navigation.navigate_to_pose",
            payload: {},
            spatialIntent: { kind: "semantic", locationId: "kitchen" },
            dependsOn: [],
            rationale: "采纳 PCD 标注的厨房入口候选点",
          }],
          assumptions: [],
          warnings: ["执行前需确认候选点当前未被占用"],
        },
      }),
    };
    const runtime = new MastraPlanningRuntime(
      { generate: vi.fn() } as never,
      { generate: vi.fn() } as never,
      new Map([["motion-worker", worker as never]]),
      new Map(),
      new Map([["motion-worker", "Plans motion commands."]]),
    );
    const planningEnvelope = planningEnvelopeSchema.parse(envelope({
      taskRequest: { text: "去厨房", sceneContext },
      capabilitySnapshot: {
        agent: "robot-agent",
        protocol: "robot-agent.v1",
        schema: { type: "object" },
        capabilities: ["navigation.navigate_to_pose"],
        safetyMode: "normal",
        robotId: "robot-1",
      },
      commandCatalog: [{
        commandKey: "navigation.navigate_to_pose",
        version: "1",
        kind: "motion",
        payloadSchema: { type: "object" },
      }],
      sceneContext,
    }));

    const fragment = await runtime.runWorker(
      "motion-worker",
      planningEnvelope,
      new AbortController().signal,
    );

    expect(fragment.steps[0]!.payload).toMatchObject({
      goalPose: { frameId: "map", x: 2, y: 3, z: 0, yaw: 0 },
      spatialResolution: {
        intent: { kind: "semantic", locationId: "kitchen" },
        resolverVersion: "scene-core/1",
      },
    });
    const prompt = worker.generate.mock.calls[0]![0] as string;
    const scopedEnvelope = JSON.parse(prompt.split("\n").at(-1)!);
    expect(scopedEnvelope.commandCatalog[0].payloadSchema).toEqual({
      type: "object",
      additionalProperties: false,
      properties: {},
    });
  });

  it("rejects a navigation step that does not select a spatial intent", async () => {
    const worker = {
      generate: vi.fn().mockResolvedValue({
        object: {
          agentId: "motion-worker",
          contribution: "motion",
          summary: "无效候选",
          steps: [{
            stepId: "go",
            commandKey: "navigation.navigate_to_pose",
            payload: {},
            spatialIntent: null,
            dependsOn: [],
            rationale: "无效",
          }],
          assumptions: [],
          warnings: [],
        },
      }),
    };
    const sceneContext = {
      sceneRef: { sceneId: "scene-demo", revision: 1 },
      bounds: { min: { x: -2, y: -2, z: -1 }, max: { x: 2, y: 2, z: 2 } },
      annotations: [],
      currentPose: null,
      selectedPoint: null,
    };
    const runtime = new MastraPlanningRuntime(
      { generate: vi.fn() } as never,
      { generate: vi.fn() } as never,
      new Map([["motion-worker", worker as never]]),
      new Map(),
      new Map([["motion-worker", "Plans motion commands."]]),
    );

    await expect(runtime.runWorker(
      "motion-worker",
      planningEnvelopeSchema.parse(envelope({
        taskRequest: { text: "去目标点", sceneContext },
        capabilitySnapshot: {
          agent: "robot-agent", protocol: "robot-agent.v1", schema: { type: "object" },
          capabilities: ["navigation.navigate_to_pose"], safetyMode: "normal", robotId: "robot-1",
        },
        commandCatalog: [{
          commandKey: "navigation.navigate_to_pose", version: "1", kind: "motion",
          payloadSchema: { type: "object" },
        }],
        sceneContext,
      })),
      new AbortController().signal,
    )).rejects.toThrow(/did not select a spatial intent/i);
  });

  it("deterministically preserves worker fields while applying merge advice", async () => {
    const structuringAgent = {
      generate: vi.fn().mockResolvedValue({
        object: {
          summary: "先移动，再打开夹爪",
          stepOrder: ["move", "open"],
          dependencies: [{ stepId: "open", dependsOn: ["move"] }],
          assumptions: ["使用默认移动距离"],
          warnings: [],
        },
      }),
    };
    const runtime = new MastraPlanningRuntime(
      { generate: vi.fn() } as never,
      structuringAgent as never,
      new Map(),
      new Map(),
      new Map(),
    );

    const proposal = await runtime.merge(
      planningEnvelopeSchema.parse(envelope()),
      [motionFragment(), manipulationFragment()],
      new AbortController().signal,
    );

    expect(proposal.steps).toEqual([
      { ...motionFragment().steps[0], contributedBy: "motion-worker" },
      {
        ...manipulationFragment().steps[0],
        dependsOn: ["move"],
        contributedBy: "manipulation-worker",
      },
    ]);
    expect(proposal.contributions).toEqual([motionFragment(), manipulationFragment()]);
  });

  it("does not ask the structuring model to reorder one immutable worker step", async () => {
    const structuringAgent = {
      generate: vi.fn().mockRejectedValue(new Error("must not be called")),
    };
    const runtime = new MastraPlanningRuntime(
      { generate: vi.fn() } as never,
      structuringAgent as never,
      new Map(),
      new Map(),
      new Map(),
    );
    const fragment = motionFragment();

    const proposal = await runtime.merge(
      planningEnvelopeSchema.parse(envelope()),
      [fragment],
      new AbortController().signal,
    );

    expect(structuringAgent.generate).not.toHaveBeenCalled();
    expect(proposal.steps).toEqual([
      { ...fragment.steps[0], contributedBy: fragment.agentId },
    ]);
    expect(proposal.contributions).toEqual([fragment]);
  });

  it("rejects merge advice that omits a worker step", async () => {
    const runtime = new MastraPlanningRuntime(
      { generate: vi.fn() } as never,
      { generate: vi.fn().mockResolvedValue({
        object: {
          summary: "incomplete",
          stepOrder: ["move"],
          dependencies: [],
          assumptions: [],
          warnings: [],
        },
      }) } as never,
      new Map(),
      new Map(),
      new Map(),
    );

    await expect(runtime.merge(
      planningEnvelopeSchema.parse(envelope()),
      [motionFragment(), manipulationFragment()],
      new AbortController().signal,
    )).rejects.toThrow(/omit|step|worker/i);
  });
});

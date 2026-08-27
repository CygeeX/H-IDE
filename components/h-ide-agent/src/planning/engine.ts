import { isDeepStrictEqual } from "node:util";
import {
  planFragmentSchema,
  planProposalSchema,
  planningEnvelopeSchema,
  planningTurnResultSchema,
  supervisorDecisionSchema,
  type PlanFragment,
  type PlanProposal,
  type PlanningEnvelope,
  type PlanningTurnResult,
  type SupervisorDecision,
  type VisualMissionSpec,
  visualMissionSpecSchema,
} from "../contracts.js";
import {
  NAVIGATION_COMMAND_KEY,
  validateNavigationPayload,
} from "../mastra/tools/spatial.js";

export interface PlanningRuntime {
  classify(envelope: PlanningEnvelope, signal: AbortSignal): Promise<unknown>;
  runWorker(agentId: string, envelope: PlanningEnvelope, signal: AbortSignal): Promise<unknown>;
  merge(
    envelope: PlanningEnvelope,
    fragments: PlanFragment[],
    signal: AbortSignal,
  ): Promise<unknown>;
  runVisualWorker?(envelope: PlanningEnvelope, signal: AbortSignal): Promise<unknown>;
}

function throwIfAborted(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException("planning turn aborted", "AbortError");
}

export class PlanningEngine {
  constructor(private readonly runtime: PlanningRuntime) {}

  async plan(input: unknown, signal: AbortSignal): Promise<PlanningTurnResult> {
    throwIfAborted(signal);
    const envelope = planningEnvelopeSchema.parse(input);
    const decision: SupervisorDecision = supervisorDecisionSchema.parse(
      await this.runtime.classify(envelope, signal),
    );
    throwIfAborted(signal);
    if (decision.status !== "route") {
      return planningTurnResultSchema.parse(decision);
    }

    const selectedAgentIds = [...new Set(decision.selectedAgentIds)];
    if (selectedAgentIds.length !== decision.selectedAgentIds.length) {
      throw new Error("supervisor selected the same worker more than once");
    }
    if (selectedAgentIds.includes("vision-worker")) {
      if (selectedAgentIds.length !== 1) {
        return planningTurnResultSchema.parse({
          status: "cannot_plan",
          assistantMessage: "当前视觉定位 V1 只支持静态可见目标，不能与其他动作组合执行。",
          reason: "visual mission cannot be combined with action workers in V1",
          missingCapabilities: ["visual_multi_stage_action_v1_1"],
        });
      }
      if (!this.runtime.runVisualWorker) {
        throw new Error("vision-worker is not configured");
      }
      const visualMission: VisualMissionSpec = visualMissionSpecSchema.parse(
        await this.runtime.runVisualWorker(envelope, signal),
      );
      if (visualMission.robotId !== envelope.taskRequest.robotId) {
        throw new Error("visual mission targets a different robot");
      }
      throwIfAborted(signal);
      const proposal = planProposalSchema.parse({
        robotId: envelope.taskRequest.robotId,
        summary: `视觉定位任务：${visualMission.targetDescription}`,
        steps: [],
        assumptions: [],
        warnings: [
          "仅生成目标点；路径规划、避障与运动执行由 Robot Agent/Nav2 负责。",
          ...visualMission.unsupportedCapabilities.map((item) => `本版本不支持：${item}`),
        ],
        contributions: [],
        contextVersion: envelope.contextVersion,
      });
      return planningTurnResultSchema.parse({
        status: "draft_ready",
        assistantMessage: `已生成待审核视觉任务：${visualMission.targetDescription}`,
        proposal,
        visualMission,
      });
    }
    const fragments = await Promise.all(
      selectedAgentIds.map(async (agentId) => {
        const fragment = planFragmentSchema.parse(
          await this.runtime.runWorker(agentId, envelope, signal),
        );
        if (fragment.agentId !== agentId) {
          throw new Error(`worker ${agentId} returned a fragment owned by ${fragment.agentId}`);
        }
        this.validateSpatialSteps(envelope, fragment.steps);
        return fragment;
      }),
    );
    throwIfAborted(signal);

    const proposal: PlanProposal = planProposalSchema.parse(
      await this.runtime.merge(envelope, fragments, signal),
    );
    this.validateMergedProposal(envelope, selectedAgentIds, fragments, proposal);
    return planningTurnResultSchema.parse({
      status: "draft_ready",
      assistantMessage: `已生成待审核计划：${proposal.summary}`,
      proposal,
    });
  }

  private validateMergedProposal(
    envelope: PlanningEnvelope,
    selectedAgentIds: string[],
    fragments: PlanFragment[],
    proposal: PlanProposal,
  ) {
    if (proposal.robotId !== envelope.taskRequest.robotId) {
      throw new Error("merged plan targets a different robot");
    }
    if (proposal.contextVersion !== envelope.contextVersion) {
      throw new Error("merged plan used stale context");
    }
    const contributed = proposal.contributions.map((fragment) => fragment.agentId);
    if (
      contributed.length !== selectedAgentIds.length ||
      selectedAgentIds.some((agentId) => !contributed.includes(agentId))
    ) {
      throw new Error("merged plan silently omitted a selected worker contribution");
    }
    const fragmentByAgent = new Map(fragments.map((fragment) => [fragment.agentId, fragment]));
    for (const contribution of proposal.contributions) {
      const original = fragmentByAgent.get(contribution.agentId);
      if (!original || !isDeepStrictEqual(contribution, original)) {
        throw new Error(`merged plan rewrote worker fragment: ${contribution.agentId}`);
      }
    }
    const sourceSteps = new Map<string, { fragment: PlanFragment; step: PlanFragment["steps"][number] }>();
    for (const fragment of fragments) {
      for (const step of fragment.steps) {
        if (sourceSteps.has(step.stepId)) {
          throw new Error(`worker fragments reuse step ID: ${step.stepId}`);
        }
        sourceSteps.set(step.stepId, { fragment, step });
      }
    }
    if (proposal.steps.length !== sourceSteps.size) {
      throw new Error("merged plan omitted or invented a worker step");
    }
    for (const merged of proposal.steps) {
      const source = sourceSteps.get(merged.stepId);
      if (
        !source ||
        merged.contributedBy !== source.fragment.agentId ||
        merged.commandKey !== source.step.commandKey ||
        merged.rationale !== source.step.rationale ||
        !isDeepStrictEqual(merged.payload, source.step.payload) ||
        source.step.dependsOn.some((dependency) => !merged.dependsOn.includes(dependency))
      ) {
        throw new Error(`merged plan rewrote or invented worker step: ${merged.stepId}`);
      }
    }
    const commands = new Set(envelope.commandCatalog.map((command) => command.commandKey));
    const capabilities = new Set(envelope.capabilitySnapshot.capabilities);
    for (const step of proposal.steps) {
      if (!commands.has(step.commandKey)) {
        throw new Error(`merged plan uses unknown command: ${step.commandKey}`);
      }
      if (!capabilities.has(step.commandKey)) {
        throw new Error(`robot is missing capability: ${step.commandKey}`);
      }
    }
    this.validateSpatialSteps(envelope, proposal.steps);
  }

  private validateSpatialSteps(
    envelope: PlanningEnvelope,
    steps: Array<{ commandKey: string; payload: Record<string, unknown> }>,
  ) {
    for (const step of steps) {
      if (step.commandKey !== NAVIGATION_COMMAND_KEY) continue;
      if (!envelope.sceneContext) {
        throw new Error("navigation.navigate_to_pose requires a published H-IDE scene context");
      }
      validateNavigationPayload(envelope.sceneContext, step.payload);
    }
  }
}

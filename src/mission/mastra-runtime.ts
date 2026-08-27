import type { Agent } from "@mastra/core/agent";
import {
  missionDraftModelOutputSchema,
  missionRevisionModelOutputSchema,
  type MissionBrief,
  type MissionReplanRequest,
} from "./contracts.js";
import type { MissionPlanningRuntime } from "./engine.js";
import { withInferenceRetry } from "../resilience/inference-retry.js";

type StructuredAgent = Pick<Agent, "generate">;

// DeepSeek V4 Flash enables its reasoning channel by default.  Mission
// planning is a bounded schema-generation step, not a long-form reasoning
// task; leaving the default on can spend the whole HTTP deadline in hidden
// reasoning before emitting a small DAG.  Keep the configured V4 Flash model
// while explicitly selecting the fast, non-thinking mode for this pass.
const DEEPSEEK_FAST_PROVIDER_OPTIONS = {
  deepseek: { thinking: { type: "disabled" as const } },
};

// Some OpenAI-compatible providers (including the DeepSeek endpoint used by
// the local integration profile) append JSON-schema bookkeeping fields such as
// ``$schema`` or an empty key to an otherwise valid structured response.  The
// provider-facing schema must strip those fields so Mastra can finish the
// generation.  We still parse the returned value with the strict domain
// schemas below before it is accepted by MissionEngine.
const missionDraftStructuredOutputSchema = missionDraftModelOutputSchema.strip();
const missionRevisionStructuredOutputSchema = missionRevisionModelOutputSchema.strip();

function missionMemory(input: { missionId: string; requestedBy: string }) {
  return {
    thread: `mission:${input.missionId}`,
    resource: `h-ide:${input.requestedBy}`,
  };
}

const AUTHORITY_BOUNDARY = [
  "You are H-IDE's semantic Mission Executive, not a robot controller.",
  "Return semantic skill nodes only. Never emit coordinates, poses, paths, waypoints, velocities, cmd_vel, ROS topics/actions/services, motor commands, or executable payloads.",
  "Use only exact allowedSkillIds from the MissionEnvelope and keep every node at or below maximumRiskLevel.",
  "Any base translation or orientation change, including an in-place scan, must be represented by an allowed navigation skill; Robot Agent/Nav2 owns all path planning and base motion. The robot head/camera is fixed and cannot rotate independently.",
  "Qwen/VLM records are hypotheses only. Only geometry, Robot Agent, scene, operator, or verifier records marked as facts may satisfy a physical precondition.",
  "PCD/scene records are static priors. They cannot prove current occupancy, reachability, or dynamic safety.",
  "If the objective explicitly says observe only, do not move, or do not rotate, never add a navigation skill—not even as a placeholder or non-executing semantic node.",
  "For target grounding use a vision intent such as task=describe_and_ground and targetDescription=<semantic target>; for an unrestricted scene summary use task=scene_description.",
  "Never emit target_region, target_description, or entry_pose. For a published place use the exact typed h-ide.place-intent/1 shape from the supplied world records, including locationId, anchorRole, targetRecordId, candidateTargetRecordIds, and an explicit completion predicate.",
  "Use explicit preconditionIds and postconditionIds so H-IDE can observe, execute, verify, and replan continuously.",
].join("\n");

export class MastraMissionRuntime implements MissionPlanningRuntime {
  constructor(private readonly missionExecutive: StructuredAgent) {}

  async draftMission(input: MissionBrief, signal: AbortSignal) {
    const output = await withInferenceRetry(async () => {
      const value = await this.missionExecutive.generate(
        [
          AUTHORITY_BOUNDARY,
          "Create one reviewable Mission DAG for the user's objective.",
          "Human approval is outside the DAG and is enforced by H-IDE. Never add a wait, approval, or confirmation node, and never disguise such control flow as a navigation or other executable skill. Return only work that may execute after the draft is approved.",
          "For a visual or landmark target, first add an observation/grounding node when such a skill is allowed, then a semantic navigation node that refers to the grounded target by record/category/description—not by geometry.",
          "Do not add speculative manipulation, dynamic following, exploration, or path-planning skills when they are absent from allowedSkillIds.",
          "Every node riskLevel must be exactly one lowercase token: low, medium, high, or critical, and must exactly match the capability riskLevel for that skill.",
          "Keep intent semantic only. For a relative base move, use fields such as motion=translate_forward, distance_m, and reference=base_link; never use x, y, z, pose, goalPose, relative_translation, path, waypoint, cmd_vel, or ROS fields.",
          "Every dependency must reference another returned nodeId and the graph must be acyclic.",
          `Mission brief: ${JSON.stringify(input)}`,
        ].join("\n"),
        {
          structuredOutput: { schema: missionDraftStructuredOutputSchema },
          abortSignal: signal,
          memory: missionMemory(input),
          modelSettings: { temperature: 0 },
          providerOptions: DEEPSEEK_FAST_PROVIDER_OPTIONS,
        },
      );
      if (value.error) throw value.error;
      return value;
    }, signal);
    return missionDraftModelOutputSchema.parse(output.object);
  }

  async replanMission(input: MissionReplanRequest, signal: AbortSignal) {
    const output = await withInferenceRetry(async () => {
      const value = await this.missionExecutive.generate(
        [
          AUTHORITY_BOUNDARY,
          "Return a full semantic replacement DAG for the active Mission revision after the supplied trigger.",
          "Human approval is outside the DAG and is enforced by H-IDE. Never add a wait, approval, or confirmation node, and never disguise such control flow as a navigation or other executable skill.",
          "Do not mutate the active revision in place. Preserve node IDs for unchanged work and use new IDs for semantically changed work.",
          "A new observation may invalidate a target hypothesis, but it cannot bypass verification or directly produce executable geometry.",
          "Prefer another observation or a bounded Nav2 rotation skill when the target is not visible; never invent a moving-head action.",
          "Every node riskLevel must be exactly one lowercase token: low, medium, high, or critical, and must exactly match the capability riskLevel for that skill.",
          "Keep intent semantic only. Never use x, y, z, pose, goalPose, relative_translation, path, waypoint, cmd_vel, or ROS fields.",
          "The reasonCode and changeSummary must identify the actual trigger and changed semantic nodes.",
          `Replan request: ${JSON.stringify(input)}`,
        ].join("\n"),
        {
          structuredOutput: { schema: missionRevisionStructuredOutputSchema },
          abortSignal: signal,
          memory: missionMemory(input),
          modelSettings: { temperature: 0 },
          providerOptions: DEEPSEEK_FAST_PROVIDER_OPTIONS,
        },
      );
      if (value.error) throw value.error;
      return value;
    }, signal);
    return missionRevisionModelOutputSchema.parse(output.object);
  }
}

import {
  visualResolutionSchema,
  visualStageRequestSchema,
  visualStageResultSchema,
  type VisualResolution,
  type VisualStageResult,
} from "../contracts.js";
import type { VisionResolutionClient } from "./client.js";

const TERMINAL = new Set(["succeeded", "needs_input", "rejected", "cancelled", "failed"]);

export interface VisualStageResolver {
  resolve(input: unknown, signal: AbortSignal): Promise<VisualStageResult>;
}

export class VisionWorker implements VisualStageResolver {
  constructor(
    private readonly client: VisionResolutionClient,
    private readonly pollIntervalMs = 200,
    private readonly maximumPolls = 150,
  ) {}

  async resolve(input: unknown, signal: AbortSignal): Promise<VisualStageResult> {
    const request = visualStageRequestSchema.parse(input);
    let resolution: VisualResolution | undefined;
    try {
      resolution = visualResolutionSchema.parse(await this.client.createResolution(request, signal));
      this.validateBinding(request, resolution);
      for (let poll = 0; !TERMINAL.has(resolution.status); poll += 1) {
        if (poll >= this.maximumPolls) throw new Error("visual resolution polling timed out");
        await this.wait(signal);
        resolution = visualResolutionSchema.parse(
          await this.client.getResolution(resolution.visualResolutionId, signal),
        );
        this.validateBinding(request, resolution);
      }
      return visualStageResultSchema.parse({
        status: resolution.status,
        visualResolutionId: resolution.visualResolutionId,
        ambiguityCandidates: resolution.ambiguityCandidates,
        explanation: this.explain(resolution),
      });
    } catch (error) {
      if (signal.aborted && resolution) {
        await this.client.cancelResolution(resolution.visualResolutionId).catch(() => undefined);
      }
      throw error;
    }
  }

  private validateBinding(
    request: ReturnType<typeof visualStageRequestSchema.parse>,
    resolution: VisualResolution,
  ) {
    const mission = request.mission;
    if (
      resolution.missionId !== mission.missionId ||
      resolution.stageId !== request.stage.stageId ||
      resolution.attempt !== request.attempt
    ) {
      throw new Error("H-IDE Vision returned a resolution for another mission stage");
    }
    if (request.candidateId) {
      if (
        resolution.selectedCandidateId !== request.candidateId
        || resolution.selectionSourceVisualResolutionId
          !== request.selectionSourceVisualResolutionId
      ) {
        throw new Error("H-IDE Vision returned a resolution for another candidate selection");
      }
    } else if (
      resolution.selectedCandidateId != null
      || resolution.selectionSourceVisualResolutionId != null
    ) {
      throw new Error("H-IDE Vision returned an unrequested candidate selection");
    }
    if (
      resolution.source.robotId !== mission.robotId ||
      resolution.source.robotBootId !== mission.robotBootId ||
      resolution.source.localizationEpochId !== mission.localizationEpochId ||
      !mission.sensorIds.includes(resolution.source.sensorId) ||
      resolution.source.sourceMode !== mission.requestedSourceMode
    ) {
      throw new Error("H-IDE Vision returned a resolution with stale observation identity");
    }
    const versions = resolution.versions;
    if (
      versions.vlmProvider !== mission.vlmProvider ||
      versions.vlmModel !== mission.vlmModel ||
      versions.promptVersion !== mission.promptVersion ||
      versions.visualAlgorithmVersion !== mission.visualAlgorithmVersion ||
      versions.approachPolicyVersion !== mission.approachPolicyVersion ||
      versions.approachPolicySha256 !== mission.approachPolicySha256
    ) {
      throw new Error("H-IDE Vision returned an unreviewed model or policy version");
    }
    if (mission.sceneRef) {
      if (
        resolution.source.sceneId !== mission.sceneRef.sceneId ||
        resolution.source.sceneRevision !== mission.sceneRef.revision
      ) {
        throw new Error("H-IDE Vision returned a different scene binding");
      }
    }
  }

  private explain(resolution: VisualResolution): string {
    if (resolution.status === "succeeded") {
      return "视觉服务已生成通过确定性安全校验的不可变目标解析；坐标仅由 H-IDE 运行时复读。";
    }
    if (resolution.status === "needs_input") {
      return `检测到 ${resolution.ambiguityCandidates.length} 个候选目标，需要用户选择后创建新解析。`;
    }
    return `视觉解析未产生可执行目标：${resolution.rejectionReason ?? resolution.status}`;
  }

  private wait(signal: AbortSignal): Promise<void> {
    if (signal.aborted) return Promise.reject(signal.reason);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, this.pollIntervalMs);
      signal.addEventListener(
        "abort",
        () => {
          clearTimeout(timer);
          reject(signal.reason);
        },
        { once: true },
      );
    });
  }
}

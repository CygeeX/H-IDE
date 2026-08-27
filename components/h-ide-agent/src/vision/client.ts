import {
  visualResolutionSchema,
  type VisualResolution,
  type VisualStageRequest,
} from "../contracts.js";

export interface VisionResolutionClient {
  createResolution(request: VisualStageRequest, signal: AbortSignal): Promise<VisualResolution>;
  getResolution(resolutionId: string, signal: AbortSignal): Promise<VisualResolution>;
  cancelResolution(resolutionId: string, signal?: AbortSignal): Promise<VisualResolution>;
}

type FetchLike = typeof fetch;

export class HttpVisionResolutionClient implements VisionResolutionClient {
  private readonly baseUrl: URL;

  constructor(
    baseUrl: string,
    private readonly token: string,
    private readonly timeoutMs = 30_000,
    private readonly fetchImpl: FetchLike = fetch,
  ) {
    this.baseUrl = new URL(baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`);
    if (!["http:", "https:"].includes(this.baseUrl.protocol)) {
      throw new Error("H-IDE Vision URL must use HTTP or HTTPS");
    }
    if (!token.trim()) throw new Error("H-IDE Vision agent token is required");
  }

  createResolution(request: VisualStageRequest, signal: AbortSignal) {
    const mission = request.mission;
    const sceneRef = mission.sceneRef;
    return this.request("POST", "internal/v1/visual-resolutions", signal, {
      requestId: [
        mission.missionId,
        request.stage.stageId,
        request.attempt,
        request.candidateId ?? "auto",
        request.selectionSourceVisualResolutionId ?? "initial",
      ].join(":"),
      missionId: mission.missionId,
      stageId: request.stage.stageId,
      attempt: request.attempt,
      traceId: `visual-stage:${mission.missionId}:${request.stage.stageId}:${request.attempt}`,
      instruction: mission.instruction,
      targetDescription: request.stage.targetDescription,
      robotId: mission.robotId,
      robotBootId: mission.robotBootId,
      localizationEpochId: mission.localizationEpochId,
      sensorId: mission.sensorIds[0],
      requestedSourceMode: mission.requestedSourceMode,
      ...(sceneRef ? { sceneId: sceneRef.sceneId, sceneRevision: sceneRef.revision } : {}),
      goalPurpose: request.stage.goalPurpose,
      ...(request.candidateId ? { candidateId: request.candidateId } : {}),
      ...(request.selectionSourceVisualResolutionId
        ? { selectionSourceVisualResolutionId: request.selectionSourceVisualResolutionId }
        : {}),
    });
  }

  getResolution(resolutionId: string, signal: AbortSignal) {
    return this.request(
      "GET",
      `internal/v1/visual-resolutions/${encodeURIComponent(resolutionId)}`,
      signal,
    );
  }

  cancelResolution(resolutionId: string, signal?: AbortSignal) {
    return this.request(
      "POST",
      `internal/v1/visual-resolutions/${encodeURIComponent(resolutionId)}/cancel`,
      signal ?? AbortSignal.timeout(this.timeoutMs),
    );
  }

  private async request(
    method: "GET" | "POST",
    path: string,
    signal: AbortSignal,
    body?: Record<string, unknown>,
  ): Promise<VisualResolution> {
    const combinedSignal = AbortSignal.any([signal, AbortSignal.timeout(this.timeoutMs)]);
    const response = await this.fetchImpl(new URL(path, this.baseUrl), {
      method,
      redirect: "error",
      headers: {
        authorization: `Bearer ${this.token}`,
        ...(body ? { "content-type": "application/json" } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
      signal: combinedSignal,
    });
    if (!response.ok) {
      const detail = (await response.text()).slice(0, 500);
      throw new Error(`H-IDE Vision returned ${response.status}${detail ? `: ${detail}` : ""}`);
    }
    return visualResolutionSchema.parse(await response.json());
  }
}

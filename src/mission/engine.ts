import {
  candidateScoreRequestSchema,
  candidateScoreResultSchema,
  missionDecisionSchema,
  missionBriefSchema,
  missionDraftModelOutputSchema,
  missionEnvelopeSchema,
  missionEventContextSchema,
  missionEventTriggerSchema,
  missionPlanDraftSchema,
  missionReplanRequestSchema,
  missionRevisionModelOutputSchema,
  planRevisionSchema,
  riskRank,
  type CandidateScoreRequest,
  type CandidateScoreResult,
  type MissionBrief,
  type MissionDecision,
  type MissionEnvelope,
  type MissionEventTrigger,
  type MissionPlanDraft,
  type MissionReplanRequest,
  type MissionRevisionModelOutput,
  type PlanRevision,
  type WorldRecord,
} from "./contracts.js";

export interface MissionPlanningRuntime {
  draftMission(input: MissionBrief, signal: AbortSignal): Promise<unknown>;
  replanMission(input: MissionReplanRequest, signal: AbortSignal): Promise<unknown>;
}

export interface MissionEngineOptions {
  /** Minimum deterministic score required before a candidate may be selected. */
  minimumCandidateScore?: number;
}

function throwIfAborted(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException("mission planning aborted", "AbortError");
}

function nowIso() {
  return new Date().toISOString();
}

function unique<T>(items: T[]) {
  return new Set(items).size === items.length;
}

function changedNodeIds(
  previous: PlanRevision["nodes"],
  next: PlanRevision["nodes"],
): string[] {
  const before = new Map(previous.map((node) => [node.nodeId, JSON.stringify(node)]));
  const after = new Map(next.map((node) => [node.nodeId, JSON.stringify(node)]));
  return [...new Set([
    ...[...before.keys()].filter((id) => !after.has(id)),
    ...[...after.keys()].filter((id) => before.get(id) !== after.get(id)),
  ])];
}

function validateNodesAgainstEnvelope(
  nodes: PlanRevision["nodes"],
  envelope: MissionEnvelope,
) {
  const allowed = new Set(envelope.allowedSkillIds);
  for (const node of nodes) {
    if (!allowed.has(node.skillId)) {
      throw new Error(`mission revision uses skill outside envelope: ${node.skillId}`);
    }
    if (riskRank(node.riskLevel) > riskRank(envelope.maximumRiskLevel)) {
      throw new Error(`mission revision node exceeds envelope risk: ${node.nodeId}`);
    }
  }
}

function validateNodesAgainstCapabilities(
  nodes: PlanRevision["nodes"],
  capabilities: MissionBrief["capabilities"],
) {
  const catalog = new Map(capabilities.map((item) => [item.skillId, item]));
  for (const node of nodes) {
    const controlFlowValue = Object.entries(node.intent).find(([key, value]) => {
      if (typeof value !== "string") return false;
      if (!["action", "mode", "operation", "state", "status", "task"].includes(key.toLowerCase())) {
        return false;
      }
      const marker = value.trim().toLowerCase().replace(/[\s-]+/gu, "_");
      return /^(?:await|require|wait)(?:_for)?_(?:approval|confirmation|human|operator|user)$/u.test(marker)
        || /^(?:approval|confirmation)_(?:pending|required)$/u.test(marker);
    });
    if (controlFlowValue) {
      throw new Error(
        `mission approval is runtime control flow, not an executable skill node: ${node.nodeId}`,
      );
    }
    const capability = catalog.get(node.skillId);
    if (!capability) throw new Error(`mission uses an unpublished capability: ${node.skillId}`);
    if (capability.skillVersion !== node.skillVersion) {
      throw new Error(`mission uses a stale capability version: ${node.skillId}`);
    }
    if (capability.riskLevel !== node.riskLevel) {
      throw new Error(`mission risk does not match capability manifest: ${node.skillId}`);
    }
    if (capability.kind === "motion" && capability.executor !== "nav2") {
      throw new Error(`mission base motion capability is not owned by Nav2: ${node.skillId}`);
    }
  }
}

function autoRevisionEligible(
  request: MissionReplanRequest,
  output: MissionRevisionModelOutput,
  changed: string[],
): boolean {
  if (!request.authorizationId) return false;
  if (request.autoRevisionCount >= request.envelope.maximumAutoRevisions) return false;
  if (request.trigger === "goal_changed" || request.trigger === "operator_intervention" || request.trigger === "safety_degraded") {
    return false;
  }
  if (output.objective !== request.activeRevision.objective) return false;
  if (changed.length === 0) return false;
  const previousSkills = new Set(request.activeRevision.nodes.map((node) => node.skillId));
  const previousNodes = new Map(request.activeRevision.nodes.map((node) => [node.nodeId, node]));
  const changedNodes = output.nodes.filter((node) => {
    const previous = previousNodes.get(node.nodeId);
    return !previous || JSON.stringify(previous) !== JSON.stringify(node);
  });
  // Policy inheritance is intentionally conservative: it can only repair a
  // low-risk node using an already approved skill. Navigation orientation and
  // all new skills still go through the human gate in H-IDE.
  return changedNodes.every((node) =>
    previousSkills.has(node.skillId) && node.riskLevel === "low"
    && riskRank(node.riskLevel) <= riskRank(request.envelope.maximumRiskLevel));
}

export class MissionEngine {
  private readonly minimumCandidateScore: number;

  constructor(
    private readonly runtime: MissionPlanningRuntime,
    options: MissionEngineOptions = {},
  ) {
    this.minimumCandidateScore = options.minimumCandidateScore ?? 0.55;
  }

  /** Generate a semantic MissionPlanDraft. No physical target is accepted here. */
  async createDraft(input: unknown, signal: AbortSignal): Promise<MissionPlanDraft> {
    throwIfAborted(signal);
    const brief = missionBriefSchema.parse(input) as MissionBrief;
    const envelope = missionEnvelopeSchema.parse(brief.envelope);
    const raw = await this.runtime.draftMission(brief, signal);
    throwIfAborted(signal);
    const modelOutput = missionDraftModelOutputSchema.parse(raw);
    if (modelOutput.nodes.some((node) => !envelope.allowedSkillIds.includes(node.skillId))) {
      throw new Error("mission draft contains a skill outside the approved envelope");
    }
    validateNodesAgainstCapabilities(modelOutput.nodes, brief.capabilities);
    const createdAt = nowIso();
    const envelopeExpiry = new Date(envelope.expiresAt).getTime();
    const computedExpiry = Math.min(
      envelopeExpiry,
      Date.now() + envelope.maximumDurationSeconds * 1_000,
    );
    if (computedExpiry <= Date.now()) throw new Error("mission envelope expires before a draft can be created");
    return missionPlanDraftSchema.parse({
      planType: "mission",
      schemaVersion: "h-ide.mission/2",
      // H-IDE allocates the durable Mission identity before planning so
      // scene/world-record references cannot be rebound to another ID when
      // the draft is approved.
      planId: brief.missionId,
      robotId: brief.robotId,
      objective: modelOutput.objective,
      nodes: modelOutput.nodes,
      envelope,
      createdAt,
      expiresAt: new Date(computedExpiry).toISOString(),
    });
  }

  /**
   * Ask the executive for a semantic revision, then apply deterministic
   * envelope, DAG, revision-budget, and approval checks.
   */
  async replan(input: unknown, signal: AbortSignal): Promise<PlanRevision> {
    throwIfAborted(signal);
    const request = missionReplanRequestSchema.parse(input);
    const raw = await this.runtime.replanMission(request, signal);
    throwIfAborted(signal);
    const modelOutput = missionRevisionModelOutputSchema.parse(raw);
    validateNodesAgainstEnvelope(modelOutput.nodes, request.envelope);
    validateNodesAgainstCapabilities(modelOutput.nodes, request.capabilities);
    const changed = changedNodeIds(request.activeRevision.nodes, modelOutput.nodes);
    if (changed.length === 0 && modelOutput.objective === request.activeRevision.objective) {
      throw new Error("replan did not change the active mission");
    }
    const inherited = autoRevisionEligible(request, modelOutput, changed);
    const disposition = inherited ? "policy_inherited" : "human_required";
    const authorizationId = inherited ? request.authorizationId : null;
    const revision = planRevisionSchema.parse({
      schemaVersion: "h-ide.plan-revision/2",
      revisionId: `rev-${request.requestId}`,
      missionId: request.missionId,
      sequence: request.activeRevision.sequence + 1,
      parentRevisionId: request.activeRevision.revisionId,
      objective: modelOutput.objective,
      nodes: modelOutput.nodes,
      reasonCode: modelOutput.reasonCode,
      reason: modelOutput.reason,
      changeSummary: unique(modelOutput.changeSummary)
        ? modelOutput.changeSummary
        : [...new Set(modelOutput.changeSummary)],
      disposition,
      authorizationId,
      createdBy: "h-ide-agent",
      createdAt: nowIso(),
    });
    return revision;
  }

  /**
   * Convert an observation/progress event into a bounded replanning decision.
   * H-IDE remains the event store and physical execution authority.
   */
  async handleEvent(
    input: unknown,
    contextInput: unknown,
    signal: AbortSignal,
  ): Promise<MissionDecision> {
    throwIfAborted(signal);
    const event = missionEventTriggerSchema.parse(input);
    const context = missionEventContextSchema.parse(contextInput);
    if (event.robotId !== context.envelope.robotId) {
      throw new Error("mission event robot identity mismatch");
    }
    const payload = event.payload;
    if (event.eventType === "safety.changed") {
      const mode = typeof payload.safetyMode === "string" ? payload.safetyMode : "unknown";
      if (["degraded", "stopped", "estop", "unknown"].includes(mode)) {
        return missionDecisionSchema.parse({
          kind: "pause",
          reason: `安全状态变为 ${mode}，暂停任务并等待 Robot Agent/H-IDE 安全状态恢复。`,
        });
      }
      return missionDecisionSchema.parse({ kind: "continue", reason: "安全状态仍允许继续" });
    }
    if (event.eventType === "operator.intervention") {
      const kind = typeof payload.kind === "string" ? payload.kind : "pause";
      if (kind === "pause" || kind === "cancel") {
        return missionDecisionSchema.parse({ kind: "pause", reason: `操作员请求${kind === "pause" ? "暂停" : "取消"}任务` });
      }
    }
    if (!event.activeRevision) {
      return missionDecisionSchema.parse({ kind: "needs_review", reason: "事件没有绑定活动修订，不能自动重规划" });
    }
    const shouldReplan = event.eventType === "verification.failed"
      || event.eventType === "mission.timeout"
      || (event.eventType === "skill.result" && ["failed", "timed_out", "unknown"].includes(String(payload.status)))
      || (event.eventType === "observation.updated" && (
        payload.replan === true || payload.targetLost === true || payload.changed === true
      ))
      || (event.eventType === "operator.intervention" && ["modify_goal", "append_requirement", "answer"].includes(String(payload.kind)));
    if (!shouldReplan) {
      return missionDecisionSchema.parse({ kind: "continue", reason: "事件不满足重规划触发条件" });
    }
    if (event.autoRevisionCount >= context.envelope.maximumAutoRevisions && event.eventType !== "operator.intervention") {
      return missionDecisionSchema.parse({ kind: "needs_review", reason: "自动重规划次数已达上限，需要人工审核" });
    }
    const trigger = event.eventType === "verification.failed"
      ? "verification_failed"
      : event.eventType === "mission.timeout"
        ? "timeout"
        : event.eventType === "observation.updated"
          ? "observation_updated"
          : event.eventType === "operator.intervention"
            ? "operator_intervention"
            : "skill_failed";
    const request: MissionReplanRequest = missionReplanRequestSchema.parse({
      requestId: `replan-${event.correlationId}`,
      missionId: event.missionId,
      robotId: event.robotId,
      requestedBy: context.requestedBy,
      objective: context.objective,
      contextVersion: context.contextVersion,
      envelope: context.envelope,
      capabilities: context.capabilities,
      activeRevision: event.activeRevision,
      worldRecords: context.worldRecords ?? [],
      trigger,
      triggerDetail: typeof payload.detail === "string" ? payload.detail : event.eventType,
      autoRevisionCount: event.autoRevisionCount,
      authorizationId: typeof payload.authorizationId === "string" ? payload.authorizationId : null,
    });
    const revision = await this.replan(request, signal);
    return missionDecisionSchema.parse({ kind: "replan", request, revision });
  }

  /**
   * Small SayCan-style scoring helper. It combines model confidence with
   * trusted world evidence, but never turns a score into an execution call.
   */
  scoreCandidates(input: unknown): CandidateScoreResult {
    const request = candidateScoreRequestSchema.parse(input) as CandidateScoreRequest;
    const recordsById = new Map(request.worldRecords.map((record) => [record.recordId, record]));
    const scores = request.candidates.map((candidate) => {
      const reasons: string[] = [];
      let score = candidate.modelScore ?? 0.5;
      if (!request.envelope.allowedSkillIds.includes(candidate.skillId)) {
        return { candidateId: candidate.candidateId, score: 0, eligible: false, reasons: ["skill is outside mission envelope"] };
      }
      if (riskRank(candidate.riskLevel) > riskRank(request.envelope.maximumRiskLevel)) {
        return { candidateId: candidate.candidateId, score: 0, eligible: false, reasons: ["risk exceeds mission envelope"] };
      }
      const evidence = candidate.evidenceRecordIds.map((id) => recordsById.get(id)).filter(Boolean) as WorldRecord[];
      if (evidence.length) {
        const trusted = evidence.filter((record) => record.kind === "fact" && record.source !== "qwen");
        const confidence = (trusted.length ? trusted : evidence)
          .reduce((sum, record) => sum + record.confidence, 0) / evidence.length;
        score = score * 0.65 + confidence * 0.35;
        reasons.push(`${trusted.length}/${evidence.length} evidence records are trusted facts`);
      } else {
        score *= 0.75;
        reasons.push("no world evidence bound to candidate");
      }
      const eligible = score >= this.minimumCandidateScore;
      if (!eligible) reasons.push(`score is below ${this.minimumCandidateScore.toFixed(2)}`);
      return {
        candidateId: candidate.candidateId,
        score: Number(score.toFixed(6)),
        eligible,
        reasons,
      };
    });
    const eligible = scores.filter((item) => item.eligible).sort((a, b) => b.score - a.score);
    const selected = eligible[0];
    return candidateScoreResultSchema.parse({
      missionId: request.missionId,
      scores,
      selectedCandidateId: selected?.candidateId ?? null,
      generatedAt: nowIso(),
    });
  }
}

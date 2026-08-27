import {
  planningEnvelopeSchema,
  planningTurnResultSchema,
  type PlanningEnvelope,
  type PlanningTurnResult,
} from "../contracts.js";
import type { PlanningTurnRepository, PlanningTurnRecord } from "./repository.js";

export interface Planner {
  plan(envelope: PlanningEnvelope, signal: AbortSignal): Promise<PlanningTurnResult>;
}

export class PlanningTurnService {
  private readonly controllers = new Map<string, AbortController>();
  private readonly jobs = new Map<string, Promise<void>>();
  private readonly ready: Promise<void>;

  constructor(
    private readonly repository: PlanningTurnRepository,
    private readonly planner: Planner,
  ) {
    this.ready = this.repository.initialize().then(() => this.repository.recoverInterrupted());
  }

  async submit(input: unknown): Promise<PlanningTurnRecord> {
    await this.ready;
    const envelope = planningEnvelopeSchema.parse(input);
    const { created, record } = await this.repository.create(envelope);
    if (created) this.launch(envelope);
    return record;
  }

  async get(taskId: string, afterEvent = 0): Promise<PlanningTurnRecord> {
    await this.ready;
    const record = await this.repository.get(taskId, afterEvent);
    if (!record) throw new Error("planning turn not found");
    return record;
  }

  async cancel(taskId: string): Promise<PlanningTurnRecord> {
    await this.ready;
    const record = await this.repository.cancel(taskId);
    if (!record) throw new Error("planning turn not found");
    this.controllers.get(taskId)?.abort();
    return record;
  }

  private launch(envelope: PlanningEnvelope) {
    const taskId = envelope.taskRequest.taskId;
    const controller = new AbortController();
    this.controllers.set(taskId, controller);
    const job = this.execute(envelope, controller)
      .catch(() => undefined)
      .finally(() => {
        this.controllers.delete(taskId);
        this.jobs.delete(taskId);
      });
    this.jobs.set(taskId, job);
  }

  private async execute(envelope: PlanningEnvelope, controller: AbortController) {
    const taskId = envelope.taskRequest.taskId;
    if (!(await this.repository.claim(taskId))) return;
    try {
      const result = planningTurnResultSchema.parse(
        await this.planner.plan(envelope, controller.signal),
      );
      await this.repository.complete(taskId, result);
    } catch (error) {
      if (controller.signal.aborted) return;
      const message = error instanceof Error ? error.message : String(error);
      await this.repository.fail(taskId, message);
    }
  }

  async waitForIdle() {
    while (this.jobs.size) await Promise.allSettled([...this.jobs.values()]);
  }
}

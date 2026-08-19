import type { Client, Row } from "@libsql/client";
import { planningEnvelopeSchema, type PlanningEnvelope, type PlanningTurnResult } from "../contracts.js";

function now() {
  return new Date().toISOString();
}

function parseJson<T>(value: unknown): T | null {
  return typeof value === "string" && value ? (JSON.parse(value) as T) : null;
}

function canonicalEnvelope(input: unknown): { envelope: PlanningEnvelope; json: string } {
  const envelope = planningEnvelopeSchema.parse(input);
  return { envelope, json: JSON.stringify(envelope) };
}

export interface PlanningEvent {
  eventId: number;
  taskId: string;
  type: string;
  payload: Record<string, unknown>;
  createdAt: string;
}

export interface PlanningTurnRecord {
  taskId: string;
  conversationId: string;
  status: string;
  result: PlanningTurnResult | null;
  error: string | null;
  cancellationRequested: boolean;
  createdAt: string;
  updatedAt: string;
  events: PlanningEvent[];
  lastEventId: number;
}

export class PlanningIdempotencyConflict extends Error {}

export class PlanningTurnRepository {
  private initializePromise: Promise<void> | undefined;
  private writeTail: Promise<void> = Promise.resolve();

  constructor(private readonly client: Client) {}

  private async serializedWrite<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.writeTail;
    let release!: () => void;
    this.writeTail = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }

  initialize(): Promise<void> {
    this.initializePromise ??= this.doInitialize();
    return this.initializePromise;
  }

  private async doInitialize() {
    await this.client.batch(
      [
        `CREATE TABLE IF NOT EXISTS planning_turns (
          task_id TEXT PRIMARY KEY,
          conversation_id TEXT NOT NULL,
          envelope_json TEXT NOT NULL,
          status TEXT NOT NULL,
          result_json TEXT,
          error TEXT,
          cancellation_requested INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )`,
        `CREATE TABLE IF NOT EXISTS planning_events (
          event_id INTEGER PRIMARY KEY AUTOINCREMENT,
          task_id TEXT NOT NULL,
          type TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(task_id) REFERENCES planning_turns(task_id)
        )`,
        "CREATE INDEX IF NOT EXISTS planning_events_task_idx ON planning_events(task_id, event_id)",
      ],
      "write",
    );
  }

  async create(input: unknown): Promise<{ created: boolean; record: PlanningTurnRecord }> {
    return this.serializedWrite(() => this.createUnlocked(input));
  }

  private async createUnlocked(input: unknown): Promise<{ created: boolean; record: PlanningTurnRecord }> {
    await this.initialize();
    const { envelope, json } = canonicalEnvelope(input);
    const taskId = envelope.taskRequest.taskId;
    const createdAt = now();
    const transaction = await this.client.transaction("write");
    let created = false;
    try {
      const inserted = await transaction.execute({
        sql: `INSERT INTO planning_turns(
          task_id, conversation_id, envelope_json, status, result_json, error,
          cancellation_requested, created_at, updated_at
        ) VALUES (?, ?, ?, 'queued', NULL, NULL, 0, ?, ?)
        ON CONFLICT(task_id) DO NOTHING`,
        args: [taskId, envelope.taskRequest.conversationId, json, createdAt, createdAt],
      });
      created = inserted.rowsAffected === 1;
      if (created) {
        await transaction.execute({
          sql: "INSERT INTO planning_events(task_id, type, payload_json, created_at) VALUES (?, 'queued', ?, ?)",
          args: [taskId, JSON.stringify({ status: "queued" }), createdAt],
        });
      } else {
        const existing = await transaction.execute({
          sql: "SELECT envelope_json FROM planning_turns WHERE task_id = ?",
          args: [taskId],
        });
        if (existing.rows[0]?.envelope_json !== json) {
          throw new PlanningIdempotencyConflict(`taskId ${taskId} is bound to a different envelope`);
        }
      }
      await transaction.commit();
    } catch (error) {
      await transaction.rollback();
      throw error;
    }
    const record = await this.get(taskId, 0);
    if (!record) throw new Error("planning turn disappeared after creation");
    return { created, record };
  }

  async claim(taskId: string): Promise<boolean> {
    return this.serializedWrite(() => this.claimUnlocked(taskId));
  }

  private async claimUnlocked(taskId: string): Promise<boolean> {
    await this.initialize();
    const changedAt = now();
    const transaction = await this.client.transaction("write");
    try {
      const result = await transaction.execute({
        sql: "UPDATE planning_turns SET status = 'running', updated_at = ? " +
          "WHERE task_id = ? AND status = 'queued' AND cancellation_requested = 0",
        args: [changedAt, taskId],
      });
      if (result.rowsAffected === 1) {
        await transaction.execute({
          sql: "INSERT INTO planning_events(task_id, type, payload_json, created_at) VALUES (?, 'running', ?, ?)",
          args: [taskId, JSON.stringify({ status: "running" }), changedAt],
        });
      }
      await transaction.commit();
      return result.rowsAffected === 1;
    } catch (error) {
      await transaction.rollback();
      throw error;
    }
  }

  async complete(taskId: string, result: PlanningTurnResult): Promise<boolean> {
    return this.transition(taskId, result.status, result, null);
  }

  async fail(taskId: string, error: string): Promise<boolean> {
    return this.transition(taskId, "failed", null, error.slice(0, 2_000));
  }

  private async transition(
    taskId: string,
    status: string,
    result: PlanningTurnResult | null,
    error: string | null,
  ): Promise<boolean> {
    return this.serializedWrite(() => this.transitionUnlocked(taskId, status, result, error));
  }

  private async transitionUnlocked(
    taskId: string,
    status: string,
    result: PlanningTurnResult | null,
    error: string | null,
  ): Promise<boolean> {
    await this.initialize();
    const changedAt = now();
    const transaction = await this.client.transaction("write");
    try {
      const changed = await transaction.execute({
        sql: "UPDATE planning_turns SET status = ?, result_json = ?, error = ?, updated_at = ? " +
          "WHERE task_id = ? AND status IN ('queued', 'running') AND cancellation_requested = 0",
        args: [status, result ? JSON.stringify(result) : null, error, changedAt, taskId],
      });
      if (changed.rowsAffected === 1) {
        await transaction.execute({
          sql: "INSERT INTO planning_events(task_id, type, payload_json, created_at) VALUES (?, ?, ?, ?)",
          args: [taskId, status, JSON.stringify({ status, result, error }), changedAt],
        });
      }
      await transaction.commit();
      return changed.rowsAffected === 1;
    } catch (caught) {
      await transaction.rollback();
      throw caught;
    }
  }

  async cancel(taskId: string): Promise<PlanningTurnRecord | null> {
    return this.serializedWrite(() => this.cancelUnlocked(taskId));
  }

  private async cancelUnlocked(taskId: string): Promise<PlanningTurnRecord | null> {
    await this.initialize();
    const changedAt = now();
    const transaction = await this.client.transaction("write");
    try {
      const changed = await transaction.execute({
        sql: "UPDATE planning_turns SET status = 'cancelled', cancellation_requested = 1, updated_at = ? " +
          "WHERE task_id = ? AND status IN ('queued', 'running')",
        args: [changedAt, taskId],
      });
      if (changed.rowsAffected === 1) {
        await transaction.execute({
          sql: "INSERT INTO planning_events(task_id, type, payload_json, created_at) VALUES (?, 'cancelled', ?, ?)",
          args: [taskId, JSON.stringify({ status: "cancelled" }), changedAt],
        });
      }
      await transaction.commit();
    } catch (error) {
      await transaction.rollback();
      throw error;
    }
    return this.get(taskId, 0);
  }

  async recoverInterrupted(): Promise<void> {
    return this.serializedWrite(() => this.recoverInterruptedUnlocked());
  }

  private async recoverInterruptedUnlocked(): Promise<void> {
    await this.initialize();
    const transaction = await this.client.transaction("write");
    const changedAt = now();
    const error = "planning turn interrupted by process restart; submit a new taskId";
    try {
      const active = await transaction.execute(
        "SELECT task_id FROM planning_turns WHERE status IN ('queued', 'running')",
      );
      for (const row of active.rows) {
        const taskId = String(row.task_id);
        await transaction.execute({
          sql: "UPDATE planning_turns SET status = 'unknown', error = ?, updated_at = ? WHERE task_id = ?",
          args: [error, changedAt, taskId],
        });
        await transaction.execute({
          sql: "INSERT INTO planning_events(task_id, type, payload_json, created_at) VALUES (?, 'unknown', ?, ?)",
          args: [taskId, JSON.stringify({ status: "unknown", error }), changedAt],
        });
      }
      await transaction.commit();
    } catch (caught) {
      await transaction.rollback();
      throw caught;
    }
  }

  async get(taskId: string, afterEvent = 0): Promise<PlanningTurnRecord | null> {
    await this.initialize();
    const turn = await this.client.execute({
      sql: "SELECT * FROM planning_turns WHERE task_id = ?",
      args: [taskId],
    });
    const row = turn.rows[0];
    if (!row) return null;
    const events = await this.client.execute({
      sql: "SELECT * FROM planning_events WHERE task_id = ? AND event_id > ? ORDER BY event_id",
      args: [taskId, afterEvent],
    });
    const eventRecords = events.rows.map((event) => this.eventFromRow(event));
    // Advance only through events included in this snapshot. Reading MAX in a
    // second query can observe a concurrent insert that was absent from the
    // event rows above, causing the caller to skip that event forever.
    const lastEventId = eventRecords.reduce(
      (latest, event) => Math.max(latest, event.eventId),
      afterEvent,
    );
    return {
      taskId: String(row.task_id),
      conversationId: String(row.conversation_id),
      status: String(row.status),
      result: parseJson<PlanningTurnResult>(row.result_json),
      error: row.error == null ? null : String(row.error),
      cancellationRequested: Boolean(row.cancellation_requested),
      createdAt: String(row.created_at),
      updatedAt: String(row.updated_at),
      events: eventRecords,
      lastEventId,
    };
  }

  private eventFromRow(row: Row): PlanningEvent {
    return {
      eventId: Number(row.event_id),
      taskId: String(row.task_id),
      type: String(row.type),
      payload: parseJson<Record<string, unknown>>(row.payload_json) ?? {},
      createdAt: String(row.created_at),
    };
  }
}

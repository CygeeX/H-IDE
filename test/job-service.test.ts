import { afterEach, describe, expect, it, vi } from "vitest";
import { createClient, type Client, type InStatement } from "@libsql/client";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PlanningTurnRepository } from "../src/jobs/repository.js";
import { PlanningTurnService } from "../src/jobs/service.js";
import { envelope, planProposal } from "./fixtures.js";

describe("durable planning jobs", () => {
  let client: Client | undefined;
  let testDirectory: string | undefined;

  afterEach(() => {
    client?.close();
    if (testDirectory) rmSync(testDirectory, { recursive: true, force: true });
  });

  const databaseUrl = () => {
    testDirectory = mkdtempSync(join(tmpdir(), "h-ide-agent-test-"));
    return `file:${join(testDirectory, "planning.db")}`;
  };

  it("is idempotent and never launches the same task twice", async () => {
    client = createClient({ url: databaseUrl() });
    const repository = new PlanningTurnRepository(client);
    const planner = {
      plan: vi.fn().mockResolvedValue({
        status: "draft_ready",
        assistantMessage: "草案已生成",
        proposal: planProposal(),
      }),
    };
    const service = new PlanningTurnService(repository, planner);
    const request = envelope();

    const first = await service.submit(request);
    const second = await service.submit(request);
    expect(first.taskId).toBe("task-1");
    expect(second.taskId).toBe("task-1");
    await service.waitForIdle();

    expect(planner.plan).toHaveBeenCalledTimes(1);
    expect((await service.get("task-1", 0)).status).toBe("draft_ready");
  });

  it("marks an interrupted run unknown instead of replaying it", async () => {
    client = createClient({ url: databaseUrl() });
    const repository = new PlanningTurnRepository(client);
    await repository.initialize();
    await repository.create(envelope());
    await repository.claim("task-1");
    await repository.recoverInterrupted();

    const recovered = await repository.get("task-1", 0);
    expect(recovered?.status).toBe("unknown");
    expect(recovered?.error).toMatch(/restart/i);
  });

  it("never advances an event cursor past an event omitted by the current read", async () => {
    const baseClient = createClient({ url: databaseUrl() });
    let injectBetweenReads = false;
    let injected = false;
    const interceptedClient = new Proxy(baseClient, {
      get(target, property) {
        if (property === "execute") {
          return async (statement: InStatement) => {
            const result = await target.execute(statement);
            const sql = typeof statement === "string" ? statement : statement.sql;
            if (
              injectBetweenReads &&
              !injected &&
              sql.startsWith("SELECT * FROM planning_events")
            ) {
              injected = true;
              await target.execute({
                sql: "INSERT INTO planning_events(task_id, type, payload_json, created_at) VALUES (?, ?, ?, ?)",
                args: ["task-1", "late-event", "{}", new Date().toISOString()],
              });
            }
            return result;
          };
        }
        const value = target[property as keyof Client];
        return typeof value === "function" ? value.bind(target) : value;
      },
    }) as Client;
    client = interceptedClient;
    const repository = new PlanningTurnRepository(interceptedClient);
    await repository.create(envelope());

    injectBetweenReads = true;
    const snapshot = await repository.get("task-1", 0);
    const returnedLastId = Math.max(0, ...(snapshot?.events.map((event) => event.eventId) ?? []));

    expect(injected).toBe(true);
    expect(snapshot?.lastEventId).toBe(returnedLastId);
  });

  it("cancels a running turn and aborts its planner", async () => {
    client = createClient({ url: databaseUrl() });
    const repository = new PlanningTurnRepository(client);
    let observedAbort = false;
    const planner = {
      plan: vi.fn((_input: unknown, signal: AbortSignal) =>
        new Promise<never>((_resolve, reject) => {
          signal.addEventListener("abort", () => {
            observedAbort = true;
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
      ),
    };
    const service = new PlanningTurnService(repository, planner);
    await service.submit(envelope());
    await vi.waitFor(() => expect(planner.plan).toHaveBeenCalled());

    expect((await service.cancel("task-1")).status).toBe("cancelled");
    await service.waitForIdle();
    expect(observedAbort).toBe(true);
  });
});

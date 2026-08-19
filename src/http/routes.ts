import { timingSafeEqual } from "node:crypto";
import { registerApiRoute, type ApiRoute } from "@mastra/core/server";
import { ZodError } from "zod";
import type { PlanningTurnService } from "../jobs/service.js";
import type { PlanningTurnRecord } from "../jobs/repository.js";
import { PlanningIdempotencyConflict } from "../jobs/repository.js";
import type { VisualStageResolver } from "../vision/worker.js";

function authorized(header: string | undefined, expectedToken: string): boolean {
  if (!header?.startsWith("Bearer ")) return false;
  const supplied = Buffer.from(header.slice("Bearer ".length));
  const expected = Buffer.from(expectedToken);
  return supplied.length === expected.length && timingSafeEqual(supplied, expected);
}

function responseRecord(record: PlanningTurnRecord) {
  return record.result ? { ...record, ...record.result, result: record.result } : record;
}

type PlanningService = Pick<PlanningTurnService, "submit" | "get" | "cancel">;

export function createPlanningRoutes(
  service: PlanningService,
  internalToken: string,
  visualStageResolver?: VisualStageResolver,
): ApiRoute[] {
  const requireInternalAuth = (context: {
    req: { header(name: string): string | undefined };
    json(payload: unknown, status?: number): Response;
  }) => {
    if (authorized(context.req.header("authorization"), internalToken)) return null;
    return context.json({ error: "internal authentication required" }, 401);
  };

  const handleError = (context: { json(payload: unknown, status?: number): Response }, error: unknown) => {
    if (error instanceof PlanningIdempotencyConflict) {
      return context.json({ error: error.message }, 409);
    }
    if (error instanceof ZodError) {
      return context.json({ error: "invalid planning envelope", details: error.issues }, 422);
    }
    const message = error instanceof Error ? error.message : String(error);
    if (message.includes("not found")) return context.json({ error: message }, 404);
    return context.json({ error: message || "planning service failed" }, 500);
  };

  const routes: ApiRoute[] = [
    registerApiRoute("/internal/v1/planning-turns", {
      method: "POST",
      requiresAuth: false,
      handler: async (context: any) => {
        const denied = requireInternalAuth(context);
        if (denied) return denied;
        try {
          const record = await service.submit(await context.req.json());
          return context.json(responseRecord(record), 202);
        } catch (error) {
          return handleError(context, error);
        }
      },
    }),
    registerApiRoute("/internal/v1/planning-turns/:taskId", {
      method: "GET",
      requiresAuth: false,
      handler: async (context: any) => {
        const denied = requireInternalAuth(context);
        if (denied) return denied;
        try {
          const rawAfter = context.req.query("afterEvent") ?? "0";
          const parsedAfter = Number.parseInt(rawAfter, 10);
          const afterEvent = Number.isSafeInteger(parsedAfter) && parsedAfter >= 0 ? parsedAfter : 0;
          const record = await service.get(context.req.param("taskId"), afterEvent);
          return context.json(responseRecord(record), 200);
        } catch (error) {
          return handleError(context, error);
        }
      },
    }),
    registerApiRoute("/internal/v1/planning-turns/:taskId/cancel", {
      method: "POST",
      requiresAuth: false,
      handler: async (context: any) => {
        const denied = requireInternalAuth(context);
        if (denied) return denied;
        try {
          const record = await service.cancel(context.req.param("taskId"));
          return context.json(responseRecord(record), 200);
        } catch (error) {
          return handleError(context, error);
        }
      },
    }),
  ];
  if (visualStageResolver) {
    routes.push(registerApiRoute("/internal/v1/visual-stages/resolve", {
      method: "POST",
      requiresAuth: false,
      handler: async (context: any) => {
        const denied = requireInternalAuth(context);
        if (denied) return denied;
        const controller = new AbortController();
        try {
          const result = await visualStageResolver.resolve(await context.req.json(), controller.signal);
          return context.json(result, 200);
        } catch (error) {
          return handleError(context, error);
        }
      },
    }));
  }
  return routes;
}

export function createHealthRoutes(readiness: () => { ready: boolean; reasons: string[] }): ApiRoute[] {
  return [
    registerApiRoute("/healthz", {
      method: "GET",
      requiresAuth: false,
      handler: (context: any) => context.json({ status: "ok" }, 200),
    }),
    registerApiRoute("/readyz", {
      method: "GET",
      requiresAuth: false,
      handler: (context: any) => {
        const state = readiness();
        return context.json(
          { status: state.ready ? "ready" : "degraded", reasons: state.reasons },
          state.ready ? 200 : 503,
        );
      },
    }),
  ];
}

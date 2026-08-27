import { timingSafeEqual } from "node:crypto";
import { registerApiRoute, type ApiRoute } from "@mastra/core/server";
import { ZodError } from "zod";
import { missionEventEvaluationRequestSchema } from "./contracts.js";
import type { MissionEngine } from "./engine.js";

type MissionService = Pick<MissionEngine, "createDraft" | "replan" | "handleEvent" | "scoreCandidates">;

function authorized(header: string | undefined, expectedToken: string): boolean {
  if (!header?.startsWith("Bearer ")) return false;
  const supplied = Buffer.from(header.slice("Bearer ".length));
  const expected = Buffer.from(expectedToken);
  return supplied.length === expected.length && timingSafeEqual(supplied, expected);
}

export function createMissionRoutes(service: MissionService, internalToken: string): ApiRoute[] {
  const requireAuth = (context: any) => authorized(
    context.req.header("authorization"),
    internalToken,
  ) ? null : context.json({ error: "internal authentication required" }, 401);
  const failure = (context: any, error: unknown) => {
    if (error instanceof ZodError) {
      return context.json({ error: "invalid Mission V2 request", details: error.issues }, 422);
    }
    const message = error instanceof Error ? error.message : String(error);
    if (/outside envelope|exceeds envelope|expired|identity mismatch|did not change/i.test(message)) {
      return context.json({ error: message }, 409);
    }
    return context.json({ error: message || "Mission V2 planning failed" }, 500);
  };
  const invoke = async (
    context: any,
    operation: (body: unknown, signal: AbortSignal) => Promise<unknown> | unknown,
  ) => {
    const denied = requireAuth(context);
    if (denied) return denied;
    const controller = new AbortController();
    try {
      return context.json(await operation(await context.req.json(), controller.signal), 200);
    } catch (error) {
      return failure(context, error);
    }
  };

  return [
    registerApiRoute("/internal/v2/mission-plans", {
      method: "POST",
      requiresAuth: false,
      handler: (context: any) => invoke(context, (body, signal) => service.createDraft(body, signal)),
    }),
    registerApiRoute("/internal/v2/mission-revisions", {
      method: "POST",
      requiresAuth: false,
      handler: (context: any) => invoke(context, (body, signal) => service.replan(body, signal)),
    }),
    registerApiRoute("/internal/v2/mission-events/evaluate", {
      method: "POST",
      requiresAuth: false,
      handler: (context: any) => invoke(context, (body, signal) => {
        const request = missionEventEvaluationRequestSchema.parse(body);
        return service.handleEvent(request.event, request.context, signal);
      }),
    }),
    registerApiRoute("/internal/v2/mission-candidate-scores", {
      method: "POST",
      requiresAuth: false,
      handler: (context: any) => invoke(context, (body) => service.scoreCandidates(body)),
    }),
  ];
}


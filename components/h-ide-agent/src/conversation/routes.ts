import { timingSafeEqual } from "node:crypto";
import { registerApiRoute, type ApiRoute } from "@mastra/core/server";
import { ZodError } from "zod";
import { conversationTurnRequestSchema } from "./contracts.js";
import type { ConversationRouter } from "./router.js";

type ConversationRoutingService = Pick<ConversationRouter, "route">;

function authorized(header: string | undefined, expectedToken: string): boolean {
  if (!header?.startsWith("Bearer ")) return false;
  const supplied = Buffer.from(header.slice("Bearer ".length));
  const expected = Buffer.from(expectedToken);
  return supplied.length === expected.length && timingSafeEqual(supplied, expected);
}

export function createConversationRoutes(
  service: ConversationRoutingService,
  internalToken: string,
): ApiRoute[] {
  return [
    registerApiRoute("/internal/v2/conversation-turns", {
      method: "POST",
      requiresAuth: false,
      handler: async (context: any) => {
        if (!authorized(context.req.header("authorization"), internalToken)) {
          return context.json({ error: "internal authentication required" }, 401);
        }

        let input;
        try {
          input = conversationTurnRequestSchema.parse(await context.req.json());
        } catch (error) {
          if (error instanceof ZodError) {
            return context.json({
              error: "invalid Assistant V2 conversation turn",
              details: error.issues,
            }, 422);
          }
          return context.json({ error: "conversation request body is not valid JSON" }, 400);
        }

        try {
          const signal = context.req.raw?.signal ?? new AbortController().signal;
          return context.json(await service.route(input, signal), 200);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          return context.json({
            error: message || "Assistant V2 conversation routing failed",
          }, 502);
        }
      },
    }),
  ];
}

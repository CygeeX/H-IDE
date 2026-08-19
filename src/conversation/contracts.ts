import { z } from "zod";

const idSchema = z.string().trim().min(1).max(160);
const messageSchema = z.string().trim().min(1).max(8_000);

export const conversationIntentSchema = z.enum([
  "SOCIAL_CHAT",
  "GENERAL_QA",
  "ROBOT_OBSERVE",
  "ROBOT_ACTION",
  "SCENE_TEACH",
  "ADMIN_DEV",
]);

export const conversationHistoryMessageSchema = z
  .object({
    role: z.enum(["user", "assistant"]),
    content: messageSchema,
  })
  .strict();

/**
 * Browser-facing conversation state is reduced by H-IDE before crossing the
 * internal boundary.  The router receives only a bounded recent window plus
 * an optional server-produced summary; neither is treated as an instruction.
 */
export const conversationTurnRequestSchema = z
  .object({
    schemaVersion: z.literal("h-ide.conversation-turn/2").default("h-ide.conversation-turn/2"),
    conversationId: idSchema,
    robotId: idSchema.nullable().optional(),
    locale: z.string().trim().min(2).max(35).default("zh-CN"),
    text: messageSchema,
    history: z.array(conversationHistoryMessageSchema).max(20).default([]),
    summary: z.string().trim().min(1).max(12_000).nullable().optional(),
  })
  .strict();

/**
 * Provider-facing schema.  Keep the root a plain object and avoid z.record so
 * the generated DashScope response_format contains neither propertyNames nor
 * patternProperties.  Domain-owned routing flags are derived after parsing,
 * rather than trusted from model output.
 */
export const conversationRouterModelOutputSchema = z
  .object({
    intent: conversationIntentSchema,
    assistantMessage: z.string().trim().min(1).max(4_000),
    clarification: z.string().trim().min(1).max(1_000).nullable(),
  })
  .strict();

// Some OpenAI-compatible providers append schema bookkeeping fields to an
// otherwise valid object.  Strip those at the provider boundary, then parse
// the accepted object again with the strict schema above.
export const conversationRouterStructuredOutputSchema =
  conversationRouterModelOutputSchema.strip();

export const conversationTurnResponseSchema = z
  .object({
    schemaVersion: z.literal("h-ide.conversation-result/2"),
    conversationId: idSchema,
    robotId: idSchema.nullable().optional(),
    intent: conversationIntentSchema,
    assistantMessage: z.string().trim().min(1).max(4_000),
    needsRobot: z.boolean(),
    shouldCreateMission: z.boolean(),
    clarification: z.string().trim().min(1).max(1_000).optional(),
  })
  .strict()
  .superRefine((value, context) => {
    const robotIntent = ["ROBOT_OBSERVE", "ROBOT_ACTION", "SCENE_TEACH"].includes(value.intent);
    if (value.needsRobot !== robotIntent) {
      context.addIssue({
        code: "custom",
        path: ["needsRobot"],
        message: "needsRobot must be derived from intent",
      });
    }
    // Only a physical action enters Mission V2/PlanDraft. Observation and
    // scene teaching use their own read-only / memory routes and must not make
    // a harmless conversation look like an executable robot plan.
    const shouldCreateMission =
      value.intent === "ROBOT_ACTION" && value.clarification == null;
    if (value.shouldCreateMission !== shouldCreateMission) {
      context.addIssue({
        code: "custom",
        path: ["shouldCreateMission"],
        message: "mission creation requires an unambiguous robot intent",
      });
    }
  });

export type ConversationIntent = z.infer<typeof conversationIntentSchema>;
export type ConversationTurnRequest = z.infer<typeof conversationTurnRequestSchema>;
export type ConversationRouterModelOutput = z.infer<typeof conversationRouterModelOutputSchema>;
export type ConversationTurnResponse = z.infer<typeof conversationTurnResponseSchema>;

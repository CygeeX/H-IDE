import { createHash } from "node:crypto";
import { z } from "zod";

const idSchema = z.string().trim().min(1).max(160);
const messageSchema = z.string().trim().min(1).max(8_000);
const memoryTextSchema = z.string().trim().min(1).max(12_000);

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
 * This context is assembled by the authenticated H-IDE server, never by the
 * browser.  `CONVERSATIONAL_ONLY` is deliberately part of the wire contract:
 * summaries and snippets may resolve discourse and preferences, but are not
 * scene, localization, perception, or safety facts.
 */
export const conversationDurableSummarySchema = z
  .object({
    summaryId: idSchema,
    revision: z.number().int().nonnegative(),
    text: memoryTextSchema,
    coveredThroughTurn: z.number().int().nonnegative(),
    generatedAt: z.string().datetime({ offset: true }),
    method: z.enum(["SERVER_COMPACTION", "OPERATOR_APPROVED"]),
    contentSha256: z.string().regex(/^[0-9a-f]{64}$/),
  })
  .strict()
  .superRefine((value, context) => {
    const digest = createHash("sha256").update(value.text, "utf8").digest("hex");
    if (digest !== value.contentSha256) {
      context.addIssue({
        code: "custom",
        path: ["contentSha256"],
        message: "durable summary digest does not match its text",
      });
    }
  });

export const conversationMemorySnippetSchema = z
  .object({
    memoryId: idSchema,
    kind: z.enum([
      "USER_PREFERENCE",
      "CONVERSATION_FACT",
      "PRIOR_DECISION",
      "ONGOING_TOPIC",
      "CONVERSATION_EXCERPT",
    ]),
    text: z.string().trim().min(1).max(2_000),
    recordedAt: z.string().datetime({ offset: true }),
    sourceTurn: z.number().int().nonnegative().optional(),
    relevanceScore: z.number().min(0).max(1).optional(),
  })
  .strict();

export const semanticReferentSchema = z
  .object({
    schemaVersion: z.literal("h-ide.semantic-referent/1"),
    referentId: z.string().trim().min(1).max(200),
    sourceTaskId: z.string().trim().min(1).max(200),
    sourceMissionId: z.string().trim().min(1).max(200).nullable().optional(),
    analysisId: z.string().trim().min(1).max(200),
    hypothesisId: z.string().trim().min(1).max(200),
    label: z.string().trim().min(1).max(200),
    aliases: z.array(z.string().trim().min(1).max(200)).max(30).default([]),
    regionId: z.string().trim().min(1).max(200).nullable().optional(),
    pixelBox: z.tuple([
      z.number().int(),
      z.number().int(),
      z.number().int(),
      z.number().int(),
    ]).nullable().optional(),
    confidence: z.number().finite().min(0).max(1).nullable().optional(),
    evidence: z.array(z.object({
      kind: z.string().trim().min(1).max(80),
      sha256: z.string().regex(/^[0-9a-f]{64}$/),
      mediaType: z.string().trim().min(1).max(120),
    }).strict()).max(20).default([]),
    capturedAt: z.string().datetime({ offset: true }),
    robotBootId: z.string().trim().min(1).max(200),
    localizationEpochId: z.string().trim().min(1).max(200),
    observationSessionId: z.string().trim().min(1).max(200).nullable().optional(),
    sceneRef: z.object({
      sceneId: z.string().trim().min(1).max(200),
      revision: z.union([
        z.string().trim().min(1).max(200),
        z.number().int().nonnegative(),
      ]),
    }).strict().nullable().optional(),
    authority: z.literal("SEMANTIC_ONLY"),
    executionEligible: z.literal(false),
  })
  .strict()
  .superRefine((value, context) => {
    if (new Set(value.aliases.map((item) => item.toLocaleLowerCase())).size !== value.aliases.length) {
      context.addIssue({ code: "custom", path: ["aliases"], message: "semantic referent aliases must be unique" });
    }
    if (value.pixelBox && (value.pixelBox[0] >= value.pixelBox[2] || value.pixelBox[1] >= value.pixelBox[3])) {
      context.addIssue({ code: "custom", path: ["pixelBox"], message: "semantic referent pixelBox is invalid" });
    }
  });

export const conversationMemoryContextSchema = z
  .object({
    schemaVersion: z.literal("h-ide.conversation-memory-context/1"),
    providedBy: z.literal("H_IDE_SERVER"),
    authority: z.literal("CONVERSATIONAL_ONLY"),
    /** Stable authenticated H-IDE principal/tenant scope, not a browser ID. */
    resourceId: idSchema,
    revision: z.number().int().nonnegative(),
    durableSummary: conversationDurableSummarySchema.nullable().optional(),
    relevantMemories: z.array(conversationMemorySnippetSchema).max(8).default([]),
    semanticReferents: z.array(semanticReferentSchema).max(20).default([]),
  })
  .strict()
  .superRefine((value, context) => {
    const totalSnippetCharacters = value.relevantMemories.reduce(
      (total, memory) => total + memory.text.length,
      0,
    );
    if (totalSnippetCharacters > 12_000) {
      context.addIssue({
        code: "custom",
        path: ["relevantMemories"],
        message: "relevant memory snippets exceed the 12000 character budget",
      });
    }
  });

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
    // Advisory UI context only. H-IDE derives all executable semantics from
    // its deterministic scene/capability boundary; this field carries no
    // coordinates and must not be treated as an authority grant.
    skillId: z.enum(["inspection.office", "assistant.voice"]).nullable().optional(),
    locale: z.string().trim().min(2).max(35).default("zh-CN"),
    text: messageSchema,
    history: z.array(conversationHistoryMessageSchema).max(20).default([]),
    /** @deprecated Prefer memoryContext.durableSummary. */
    summary: z.string().trim().min(1).max(12_000).nullable().optional(),
    memoryContext: conversationMemoryContextSchema.nullable().optional(),
  })
  .strict()
  .superRefine((value, context) => {
    const recentHistoryCharacters = value.history.reduce(
      (total, message) => total + message.content.length,
      0,
    );
    if (recentHistoryCharacters > 40_000) {
      context.addIssue({
        code: "custom",
        path: ["history"],
        message: "recent conversation history exceeds the 40000 character budget",
      });
    }
  });

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
export type ConversationHistoryMessage = z.infer<typeof conversationHistoryMessageSchema>;
export type ConversationDurableSummary = z.infer<typeof conversationDurableSummarySchema>;
export type ConversationMemorySnippet = z.infer<typeof conversationMemorySnippetSchema>;
export type SemanticReferent = z.infer<typeof semanticReferentSchema>;
export type ConversationMemoryContext = z.infer<typeof conversationMemoryContextSchema>;
export type ConversationTurnRequest = z.infer<typeof conversationTurnRequestSchema>;
export type ConversationRouterModelOutput = z.infer<typeof conversationRouterModelOutputSchema>;
export type ConversationTurnResponse = z.infer<typeof conversationTurnResponseSchema>;

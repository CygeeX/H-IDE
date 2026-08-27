import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { z } from "zod";
import {
  conversationRouterStructuredOutputSchema,
  conversationTurnRequestSchema,
  conversationTurnResponseSchema,
} from "../src/conversation/contracts.js";

describe("Assistant V2 conversation contracts", () => {
  it("accepts the compact boundary and applies stable defaults", () => {
    const parsed = conversationTurnRequestSchema.parse({
      conversationId: "conversation-1",
      text: "你好",
    });
    expect(parsed).toMatchObject({
      schemaVersion: "h-ide.conversation-turn/2",
      locale: "zh-CN",
      history: [],
    });
    expect(() => conversationTurnRequestSchema.parse({
      conversationId: "conversation-1",
      text: "你好",
      executableCommand: { topic: "/cmd_vel" },
    })).toThrow();
  });

  it("accepts server-owned conversational memory while preserving the legacy summary", () => {
    const parsed = conversationTurnRequestSchema.parse({
      conversationId: "conversation-memory-1",
      text: "那我们接着聊",
      summary: "旧调用方仍可发送这个摘要。",
      memoryContext: {
        schemaVersion: "h-ide.conversation-memory-context/1",
        providedBy: "H_IDE_SERVER",
        authority: "CONVERSATIONAL_ONLY",
        resourceId: "authenticated-user-1",
        revision: 7,
        durableSummary: {
          summaryId: "summary-7",
          revision: 2,
          text: "用户正在设计一个淡紫色机器人助手界面。",
          coveredThroughTurn: 42,
          generatedAt: "2026-08-19T08:00:00.000Z",
          method: "SERVER_COMPACTION",
          contentSha256: createHash("sha256")
            .update("用户正在设计一个淡紫色机器人助手界面。", "utf8")
            .digest("hex"),
        },
        relevantMemories: [{
          memoryId: "memory-theme",
          kind: "USER_PREFERENCE",
          text: "用户偏好淡奶油色与淡紫色。",
          recordedAt: "2026-08-19T08:01:00.000Z",
          relevanceScore: 0.92,
        }],
      },
    });
    expect(parsed.summary).toBe("旧调用方仍可发送这个摘要。");
    expect(parsed.memoryContext?.authority).toBe("CONVERSATIONAL_ONLY");
    expect(parsed.memoryContext?.durableSummary?.coveredThroughTurn).toBe(42);
    expect(() => conversationTurnRequestSchema.parse({
      conversationId: "conversation-memory-bad-digest",
      text: "继续",
      memoryContext: {
        ...parsed.memoryContext,
        durableSummary: {
          ...parsed.memoryContext?.durableSummary,
          contentSha256: "0".repeat(64),
        },
      },
    })).toThrow(/digest/i);
  });

  it("rejects browser-like memory authority and oversized bounded context", () => {
    expect(() => conversationTurnRequestSchema.parse({
      conversationId: "conversation-memory-2",
      text: "继续",
      memoryContext: {
        schemaVersion: "h-ide.conversation-memory-context/1",
        providedBy: "BROWSER",
        authority: "PHYSICAL_TRUTH",
        resourceId: "user-1",
        revision: 1,
        relevantMemories: [],
      },
    })).toThrow();

    expect(() => conversationTurnRequestSchema.parse({
      conversationId: "conversation-memory-3",
      text: "继续",
      history: Array.from({ length: 6 }, (_, index) => ({
        role: index % 2 ? "assistant" : "user",
        content: `${index}${"x".repeat(7_999)}`,
      })),
    })).toThrow(/40000 character budget/i);
  });

  it("emits a DashScope-compatible root object schema", () => {
    const schema = z.toJSONSchema(conversationRouterStructuredOutputSchema);
    expect(schema.type).toBe("object");
    const forbidden: string[] = [];
    const visit = (value: unknown, path = "$") => {
      if (value == null || typeof value !== "object") return;
      const record = value as Record<string, unknown>;
      if ("patternProperties" in record || "propertyNames" in record) forbidden.push(path);
      for (const [key, child] of Object.entries(record)) visit(child, `${path}.${key}`);
    };
    visit(schema);
    expect(forbidden).toEqual([]);
  });

  it("rejects contradictory robot and mission flags", () => {
    const base = {
      schemaVersion: "h-ide.conversation-result/2",
      conversationId: "conversation-1",
      intent: "SOCIAL_CHAT",
      assistantMessage: "你好！",
      needsRobot: false,
      shouldCreateMission: false,
    } as const;
    expect(conversationTurnResponseSchema.parse(base)).toEqual(base);
    expect(() => conversationTurnResponseSchema.parse({
      ...base,
      needsRobot: true,
      shouldCreateMission: false,
    })).toThrow(/needsRobot|mission/i);
    expect(() => conversationTurnResponseSchema.parse({
      ...base,
      intent: "ROBOT_ACTION",
      needsRobot: true,
      shouldCreateMission: true,
      clarification: "要操作哪台机器人？",
    })).toThrow(/mission/i);
    expect(conversationTurnResponseSchema.parse({
      ...base,
      intent: "ROBOT_OBSERVE",
      needsRobot: true,
      shouldCreateMission: false,
    }).shouldCreateMission).toBe(false);
    expect(conversationTurnResponseSchema.parse({
      ...base,
      intent: "SCENE_TEACH",
      needsRobot: true,
      shouldCreateMission: false,
    }).shouldCreateMission).toBe(false);
  });
});

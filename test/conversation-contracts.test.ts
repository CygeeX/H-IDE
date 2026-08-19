import { describe, expect, it } from "vitest";
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

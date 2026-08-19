import { createHash } from "node:crypto";
import { LibSQLStore } from "@mastra/libsql";
import { Memory } from "@mastra/memory";
import { describe, expect, it } from "vitest";
import {
  MastraConversationMemory,
  conversationMemoryInternals,
  requestOnlyContext,
} from "../src/conversation/memory.js";
import {
  conversationTurnRequestSchema,
  conversationTurnResponseSchema,
  type ConversationTurnRequest,
} from "../src/conversation/contracts.js";

function request(turn: number, text = `第 ${turn} 轮普通对话`): ConversationTurnRequest {
  return conversationTurnRequestSchema.parse({
    conversationId: "long-conversation-1",
    locale: "zh-CN",
    text,
    memoryContext: {
      schemaVersion: "h-ide.conversation-memory-context/1",
      providedBy: "H_IDE_SERVER",
      authority: "CONVERSATIONAL_ONLY",
      resourceId: "user-42",
      revision: turn,
      relevantMemories: turn === 1
        ? [{
            memoryId: "preference-color",
            kind: "USER_PREFERENCE",
            text: "用户偏好淡紫色界面。",
            recordedAt: "2026-08-19T08:00:00.000Z",
          }]
        : [],
    },
  });
}

function response(input: ConversationTurnRequest) {
  return conversationTurnResponseSchema.parse({
    schemaVersion: "h-ide.conversation-result/2",
    conversationId: input.conversationId,
    intent: "SOCIAL_CHAT",
    assistantMessage: `已收到：${input.text}`,
    needsRobot: false,
    shouldCreateMission: false,
  });
}

describe("Assistant V2 durable conversation memory", () => {
  it("keeps a 55-turn transcript, auditable summaries, bounded recent context, and relevant recall", async () => {
    const store = new LibSQLStore({ id: "conversation-memory-long-test", url: "file::memory:" });
    const mastraMemory = new Memory({ storage: store, options: { lastMessages: 20 } });
    const memory = new MastraConversationMemory(mastraMemory);

    for (let turn = 1; turn <= 55; turn += 1) {
      const input = request(
        turn,
        turn === 1 ? "请记住我最喜欢淡紫色界面" : `第 ${turn} 轮，我们继续讨论界面设计`,
      );
      await memory.record(input, response(input), requestOnlyContext(input));
    }

    const state = await memory.inspect(request(56, "我喜欢什么颜色的界面？"));
    expect(state).toBeDefined();
    expect(state?.turnCount).toBe(55);
    expect(state?.durableSummary).toMatchObject({
      method: "SERVER_COMPACTION",
      coveredThroughTurn: 40,
      revision: 3,
    });
    expect(state?.durableSummary?.contentSha256).toMatch(/^[0-9a-f]{64}$/);
    const compactions = state?.audit.filter((event) => event.eventType === "COMPACTION") ?? [];
    expect(compactions).toHaveLength(3);
    expect(compactions.map((event) => [event.coveredFromTurn, event.coveredThroughTurn]))
      .toEqual([[1, 20], [21, 30], [31, 40]]);
    expect(compactions.every((event) => /^[0-9a-f]{64}$/.test(event.sourceDigest ?? "")))
      .toBe(true);

    // A new service instance over the same Mastra storage simulates restart.
    const restarted = new MastraConversationMemory(
      new Memory({ storage: store, options: { lastMessages: 20 } }),
    );
    const prepared = await restarted.prepare(request(56, "我喜欢什么颜色的界面？"));
    expect(prepared.recentMessages).toHaveLength(20);
    expect(prepared.audit.storedMessageCount).toBe(110);
    expect(prepared.durableSummary?.coveredThroughTurn).toBe(40);
    expect(prepared.relevantMemories.some((memory) =>
      memory.memoryId === "preference-color" || memory.text.includes("淡紫色")))
      .toBe(true);
  });

  it("does not claim a summary coverage range when the bounded source has a missing turn", () => {
    expect(conversationMemoryInternals.summarizeTurns(
      undefined,
      [],
      1,
      10,
      "conversation-with-gap",
    )).toBeUndefined();
  });

  it("uses deterministic lexical retrieval without interpreting snippets as authority", () => {
    expect(conversationMemoryInternals.lexicalScore(
      "继续讨论淡紫色界面",
      "用户偏好淡紫色界面",
    )).toBeGreaterThan(0);
    expect(conversationMemoryInternals.lexicalScore(
      "继续讨论淡紫色界面",
      "机器人当前坐标是 1,2",
    )).toBe(0);
  });

  it("removes the overlapping server history instead of duplicating the recent window", () => {
    const history = [
      { role: "user" as const, content: "我们讨论淡紫色界面。" },
      { role: "assistant" as const, content: "好的，我们继续。" },
    ];
    expect(conversationMemoryInternals.mergeRecentHistory(history, history)).toEqual(history);
    expect(conversationMemoryInternals.mergeRecentHistory(
      [{ role: "user", content: "更早的一轮" }, ...history],
      history,
    )).toEqual([{ role: "user", content: "更早的一轮" }, ...history]);
  });

  it("continues after a server summary when persistence starts mid-conversation", async () => {
    const store = new LibSQLStore({ id: "conversation-memory-midstream-test", url: "file::memory:" });
    const memory = new MastraConversationMemory(
      new Memory({ storage: store, options: { lastMessages: 20 } }),
    );
    const summaryText = "服务接入前的 50 轮对话已由 H-IDE 服务端压缩。";
    const input = conversationTurnRequestSchema.parse({
      conversationId: "midstream-conversation",
      text: "继续",
      memoryContext: {
        schemaVersion: "h-ide.conversation-memory-context/1",
        providedBy: "H_IDE_SERVER",
        authority: "CONVERSATIONAL_ONLY",
        resourceId: "user-midstream",
        revision: 10,
        durableSummary: {
          summaryId: "summary-midstream",
          revision: 4,
          text: summaryText,
          coveredThroughTurn: 50,
          generatedAt: "2026-08-19T08:00:00.000Z",
          method: "SERVER_COMPACTION",
          contentSha256: createHash("sha256").update(summaryText, "utf8").digest("hex"),
        },
        relevantMemories: [],
      },
    });
    await memory.record(input, response(input), requestOnlyContext(input));
    expect((await memory.inspect(input))?.turnCount).toBe(51);
  });
});

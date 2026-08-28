import { describe, expect, it } from "vitest";

import type { AssistantConversation, AssistantMessage, AssistantTurn } from "./contracts";
import { buildConversationTimeline, mergeConversationMessages } from "./conversation-timeline";

function message(
  messageId: string,
  taskId: string,
  role: "user" | "assistant",
  content: string,
  createdAt: string,
  turnSequence: number,
): AssistantMessage {
  return {
    messageId,
    taskId,
    turnSequence,
    role,
    content,
    modality: "text",
    createdAt,
  };
}

function turn(taskId: string, turnSequence: number): AssistantTurn {
  return {
    taskId,
    conversationId: "conversation-timeline",
    turnSequence,
    status: "completed",
  };
}

describe("conversation timeline", () => {
  it("keeps duplicate text in strict server turn order and binds Mission links", () => {
    const conversation: AssistantConversation = {
      conversationId: "conversation-timeline",
      robotId: "g1-sim",
      requestedBy: "operator",
      locale: "zh-CN",
      status: "idle",
      messages: [
        message("message-u2", "task-2", "user", "去椅子旁边", "2026-08-27T00:00:00Z", 2),
        message("message-u1", "task-1", "user", "去椅子旁边", "2026-08-27T00:00:00Z", 1),
      ],
      turns: [turn("task-2", 2), turn("task-1", 1)],
      missionLinks: [{
        taskId: "task-1",
        turnSequence: 1,
        missionDraftId: "mission-draft-1",
        missionId: "mission-1",
        createdAt: "2026-08-27T00:00:01Z",
        updatedAt: "2099-01-01T00:00:00Z",
      }],
    };

    const timeline = buildConversationTimeline(conversation, null);

    expect(timeline.map((item) => item.taskId)).toEqual(["task-1", "task-2"]);
    expect(timeline[0]?.missionLink?.missionId).toBe("mission-1");
    expect(timeline[1]?.messages[0]?.content).toBe("去椅子旁边");
  });

  it("replaces an optimistic message with its server identity", () => {
    const persisted = [
      message("message-server", "task-1", "user", "寻找电脑", "2026-08-27T00:00:01Z", 1),
    ];
    const local = [
      message("message-local", "task-1", "user", "寻找电脑", "2099-01-01T00:00:00Z", 1),
    ];

    expect(mergeConversationMessages(persisted, local)).toEqual(persisted);
  });
});

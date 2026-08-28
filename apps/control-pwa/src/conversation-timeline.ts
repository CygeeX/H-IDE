import type {
  AssistantConversation,
  AssistantMessage,
  AssistantMissionLink,
  AssistantTurn,
} from "./contracts";

export interface ConversationTimelineTurn {
  key: string;
  taskId: string | null;
  turnSequence: number | null;
  createdAt: string;
  messages: AssistantMessage[];
  turn: AssistantTurn | null;
  missionLink: AssistantMissionLink | null;
}

function finiteSequence(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 1
    ? value
    : null;
}

function earlier(left: string | undefined, right: string | undefined): string {
  if (!left) return right ?? "";
  if (!right) return left;
  return left.localeCompare(right) <= 0 ? left : right;
}

export function mergeConversationMessages(
  persisted: AssistantMessage[],
  local: AssistantMessage[],
): AssistantMessage[] {
  const result = [...persisted];
  const ids = new Set(persisted.map((item) => item.messageId));
  for (const message of local) {
    if (ids.has(message.messageId)) continue;
    const serverEquivalent = persisted.some((item) => (
      Boolean(message.taskId)
      && item.taskId === message.taskId
      && item.role === message.role
      && item.content === message.content
    ));
    if (!serverEquivalent) result.push(message);
  }
  return result;
}

export function buildConversationTimeline(
  conversation: AssistantConversation | null,
  activeTurn: AssistantTurn | null,
  localMessages: AssistantMessage[] = [],
): ConversationTimelineTurn[] {
  const persistedMessages = conversation?.messages ?? [];
  const messages = mergeConversationMessages(persistedMessages, localMessages);
  const turnsByTask = new Map<string, AssistantTurn>();
  for (const turn of [...(conversation?.turns ?? []), ...(activeTurn ? [activeTurn] : [])]) {
    const current = turnsByTask.get(turn.taskId);
    if (!current || current.status !== turn.status || !current.result) {
      turnsByTask.set(turn.taskId, turn);
    }
  }
  const linksByTask = new Map(
    (conversation?.missionLinks ?? []).map((item) => [item.taskId, item]),
  );
  const taskIds: string[] = [];
  const remember = (taskId: string | null | undefined) => {
    if (taskId && !taskIds.includes(taskId)) taskIds.push(taskId);
  };
  for (const turn of turnsByTask.values()) remember(turn.taskId);
  for (const message of messages) remember(message.taskId);
  for (const link of linksByTask.values()) remember(link.taskId);

  const groups: ConversationTimelineTurn[] = taskIds.map((taskId) => {
    const taskMessages = messages.filter((item) => item.taskId === taskId);
    const turn = turnsByTask.get(taskId) ?? null;
    const missionLink = linksByTask.get(taskId) ?? null;
    const turnSequence = finiteSequence(turn?.turnSequence)
      ?? finiteSequence(missionLink?.turnSequence)
      ?? taskMessages.map((item) => finiteSequence(item.turnSequence)).find(
        (item): item is number => item !== null,
      )
      ?? null;
    const createdAt = taskMessages.reduce(
      (value, item) => earlier(value, item.createdAt),
      "",
    ) || missionLink?.createdAt || "";
    taskMessages.sort((left, right) => (
      left.createdAt.localeCompare(right.createdAt)
      || left.messageId.localeCompare(right.messageId)
    ));
    return {
      key: `turn:${taskId}`,
      taskId,
      turnSequence,
      createdAt,
      messages: taskMessages,
      turn,
      missionLink,
    } satisfies ConversationTimelineTurn;
  });

  const orphanMessages = messages.filter((item) => !item.taskId);
  if (orphanMessages.length) {
    orphanMessages.sort((left, right) => (
      left.createdAt.localeCompare(right.createdAt)
      || left.messageId.localeCompare(right.messageId)
    ));
    groups.push({
      key: "turn:legacy-orphans",
      taskId: null,
      turnSequence: null,
      createdAt: orphanMessages[0]?.createdAt ?? "",
      messages: orphanMessages,
      turn: null,
      missionLink: null,
    });
  }

  groups.sort((left, right) => {
    if (left.turnSequence !== null && right.turnSequence !== null) {
      return left.turnSequence - right.turnSequence;
    }
    if (left.turnSequence !== null) return -1;
    if (right.turnSequence !== null) return 1;
    return left.createdAt.localeCompare(right.createdAt) || left.key.localeCompare(right.key);
  });
  return groups;
}

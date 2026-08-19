import { createHash, randomUUID } from "node:crypto";
import type { MastraDBMessage } from "@mastra/core/agent";
import type { StorageThreadType } from "@mastra/core/memory";
import type { Memory } from "@mastra/memory";
import {
  conversationDurableSummarySchema,
  conversationMemorySnippetSchema,
  conversationTurnRequestSchema,
  type ConversationDurableSummary,
  type ConversationHistoryMessage,
  type ConversationMemorySnippet,
  type ConversationTurnRequest,
  type ConversationTurnResponse,
} from "./contracts.js";

type MemoryPort = Pick<
  Memory,
  "getThreadById" | "recall" | "saveMessages" | "saveThread"
>;

const RECENT_MESSAGE_LIMIT = 20;
const RELEVANT_MEMORY_LIMIT = 8;
const RETRIEVAL_SCAN_LIMIT = 256;
const COMPACTION_START_TURNS = 30;
const COMPACTION_INTERVAL_TURNS = 10;
const SUMMARY_CHARACTER_LIMIT = 12_000;
const STORED_SNIPPET_LIMIT = 64;
const AUDIT_EVENT_LIMIT = 64;

const memoryStateKey = "hIdeConversationMemoryV1";

export interface ConversationContextAudit {
  recentMessageCount: number;
  storedMessageCount: number;
  summaryId?: string;
  summaryRevision?: number;
  selectedMemoryIds: string[];
  retrievalMethod: "NONE" | "LEXICAL_V1" | "SERVER_SELECTED" | "SERVER_AND_LEXICAL_V1";
}

export interface PreparedConversationContext {
  recentMessages: ConversationHistoryMessage[];
  durableSummary?: ConversationDurableSummary;
  relevantMemories: ConversationMemorySnippet[];
  audit: ConversationContextAudit;
}

interface ConversationMemoryAuditEvent {
  eventId: string;
  eventType: "CONTEXT_USED" | "COMPACTION";
  recordedAt: string;
  turnNumber: number;
  summaryRevision?: number;
  selectedMemoryIds?: string[];
  coveredFromTurn?: number;
  coveredThroughTurn?: number;
  sourceDigest?: string;
}

interface StoredConversationMemoryState {
  schemaVersion: "h-ide.conversation-memory-state/1";
  conversationId: string;
  resourceId: string;
  turnCount: number;
  contextRevision: number;
  durableSummary?: ConversationDurableSummary;
  memorySnippets: ConversationMemorySnippet[];
  audit: ConversationMemoryAuditEvent[];
}

interface ConversationScope {
  threadId: string;
  resourceId: string;
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function scopeFor(input: ConversationTurnRequest): ConversationScope {
  const owner = input.memoryContext?.resourceId ?? `legacy:${input.conversationId}`;
  const resourceId = `h-ide-conversation:${owner}`;
  const digest = sha256(`${resourceId}\0${input.conversationId}`).slice(0, 40);
  return { threadId: `assistant:${digest}`, resourceId };
}

function initialState(input: ConversationTurnRequest, scope: ConversationScope): StoredConversationMemoryState {
  return {
    schemaVersion: "h-ide.conversation-memory-state/1",
    conversationId: input.conversationId,
    resourceId: scope.resourceId,
    turnCount: 0,
    contextRevision: input.memoryContext?.revision ?? 0,
    ...(input.memoryContext?.durableSummary
      ? { durableSummary: input.memoryContext.durableSummary }
      : input.summary
        ? {
            durableSummary: legacySummary(input),
          }
        : {}),
    memorySnippets: input.memoryContext?.relevantMemories ?? [],
    audit: [],
  };
}

function legacySummary(input: ConversationTurnRequest): ConversationDurableSummary {
  const text = input.summary!;
  return {
    summaryId: `legacy-${sha256(`${input.conversationId}\0${text}`).slice(0, 24)}`,
    revision: 0,
    text,
    coveredThroughTurn: 0,
    generatedAt: new Date(0).toISOString(),
    method: "SERVER_COMPACTION",
    contentSha256: sha256(text),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function parseStoredState(
  thread: StorageThreadType | null,
  input: ConversationTurnRequest,
  scope: ConversationScope,
): StoredConversationMemoryState {
  const raw = isRecord(thread?.metadata) ? thread.metadata[memoryStateKey] : undefined;
  if (!isRecord(raw)
    || raw.schemaVersion !== "h-ide.conversation-memory-state/1"
    || raw.conversationId !== input.conversationId
    || raw.resourceId !== scope.resourceId
    || !Number.isInteger(raw.turnCount)
    || !Number.isInteger(raw.contextRevision)
    || !Array.isArray(raw.memorySnippets)
    || !Array.isArray(raw.audit)) {
    return initialState(input, scope);
  }
  try {
    const snippets = raw.memorySnippets.map((snippet) =>
      conversationMemorySnippetSchema.parse(snippet));
    const durableSummary = raw.durableSummary == null
      ? undefined
      : conversationDurableSummarySchema.parse(raw.durableSummary);
    const audit = raw.audit.filter(isRecord).flatMap((event) => {
      if (typeof event.eventId !== "string"
        || (event.eventType !== "CONTEXT_USED" && event.eventType !== "COMPACTION")
        || typeof event.recordedAt !== "string"
        || !Number.isInteger(event.turnNumber)) return [];
      return [event as unknown as ConversationMemoryAuditEvent];
    });
    return {
      schemaVersion: "h-ide.conversation-memory-state/1",
      conversationId: input.conversationId,
      resourceId: scope.resourceId,
      turnCount: raw.turnCount as number,
      contextRevision: raw.contextRevision as number,
      ...(durableSummary ? { durableSummary } : {}),
      memorySnippets: snippets,
      audit,
    };
  } catch {
    return initialState(input, scope);
  }
}

function messageText(message: MastraDBMessage): string | undefined {
  const text = message.content.parts
    .filter((part): part is Extract<typeof part, { type: "text" }> => part.type === "text")
    .map((part) => part.text)
    .join("\n")
    .trim();
  return text || undefined;
}

function messageTurnNumber(message: MastraDBMessage): number | undefined {
  const metadata = message.content.metadata;
  const value = isRecord(metadata) ? metadata.turnNumber : undefined;
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : undefined;
}

function storedHistory(messages: MastraDBMessage[]): ConversationHistoryMessage[] {
  return messages.flatMap((message) => {
    if (message.role !== "user" && message.role !== "assistant") return [];
    const content = messageText(message);
    return content ? [{ role: message.role, content }] : [];
  });
}

function mergeRecentHistory(
  stored: ConversationHistoryMessage[],
  provided: ConversationHistoryMessage[],
): ConversationHistoryMessage[] {
  const sameMessage = (
    left: ConversationHistoryMessage,
    right: ConversationHistoryMessage,
  ) => left.role === right.role && left.content === right.content;
  let overlap = 0;
  const maximumOverlap = Math.min(stored.length, provided.length);
  for (let length = maximumOverlap; length > 0; length -= 1) {
    const storedOffset = stored.length - length;
    if (provided.slice(0, length).every((message, index) =>
      sameMessage(stored[storedOffset + index]!, message))) {
      overlap = length;
      break;
    }
  }
  const merged: ConversationHistoryMessage[] = [];
  for (const message of [...stored, ...provided.slice(overlap)]) {
    const previous = merged.at(-1);
    if (previous && sameMessage(previous, message)) continue;
    merged.push(message);
  }
  return merged.slice(-RECENT_MESSAGE_LIMIT);
}

function lexicalTerms(value: string): Set<string> {
  const normalized = value.normalize("NFKC").toLocaleLowerCase();
  const terms = new Set(normalized.match(/[\p{L}\p{N}_-]{2,}/gu) ?? []);
  const han = [...normalized.matchAll(/[\p{Script=Han}]+/gu)].map((match) => match[0]);
  for (const run of han) {
    for (let offset = 0; offset < run.length - 1; offset += 1) {
      terms.add(run.slice(offset, offset + 2));
    }
  }
  return terms;
}

function lexicalScore(query: string, candidate: string): number {
  const queryTerms = lexicalTerms(query);
  const candidateTerms = lexicalTerms(candidate);
  if (queryTerms.size === 0 || candidateTerms.size === 0) return 0;
  let overlap = 0;
  for (const term of queryTerms) if (candidateTerms.has(term)) overlap += 1;
  return Math.min(1, overlap / Math.sqrt(queryTerms.size * candidateTerms.size));
}

function clampText(value: string, maximum: number): string {
  const compact = value.replace(/\s+/gu, " ").trim();
  if (compact.length <= maximum) return compact;
  return `${compact.slice(0, maximum - 1)}…`;
}

function messageExcerpt(message: MastraDBMessage, score: number): ConversationMemorySnippet | undefined {
  const text = messageText(message);
  const turnNumber = messageTurnNumber(message);
  if (!text || turnNumber == null) return undefined;
  return {
    memoryId: `excerpt-${sha256(`${message.id}\0${text}`).slice(0, 24)}`,
    kind: "CONVERSATION_EXCERPT",
    text: clampText(`${message.role === "user" ? "用户" : "助手"}：${text}`, 800),
    recordedAt: message.createdAt.toISOString(),
    sourceTurn: turnNumber,
    relevanceScore: Number(score.toFixed(6)),
  };
}

function selectRelevantMemories(
  query: string,
  explicit: ConversationMemorySnippet[],
  persisted: ConversationMemorySnippet[],
  olderMessages: MastraDBMessage[],
): ConversationMemorySnippet[] {
  const candidates = new Map<string, ConversationMemorySnippet>();
  for (const memory of persisted) {
    const score = lexicalScore(query, memory.text);
    if (score <= 0) continue;
    candidates.set(memory.memoryId, { ...memory, relevanceScore: Number(score.toFixed(6)) });
  }
  // H-IDE already performed the authorization-aware retrieval for explicit
  // snippets. Keep those ahead of local lexical candidates even when the
  // current turn is a pronoun-only follow-up with no overlapping tokens.
  for (const memory of explicit) {
    const score = memory.relevanceScore ?? 1;
    candidates.set(memory.memoryId, { ...memory, relevanceScore: Number(score.toFixed(6)) });
  }
  for (const message of olderMessages) {
    const text = messageText(message);
    if (!text) continue;
    const score = lexicalScore(query, text);
    if (score <= 0) continue;
    const excerpt = messageExcerpt(message, score);
    if (excerpt) candidates.set(excerpt.memoryId, excerpt);
  }
  return [...candidates.values()]
    .sort((left, right) =>
      (right.relevanceScore ?? 0) - (left.relevanceScore ?? 0)
      || right.recordedAt.localeCompare(left.recordedAt)
      || left.memoryId.localeCompare(right.memoryId))
    .slice(0, RELEVANT_MEMORY_LIMIT);
}

function incomingSummary(input: ConversationTurnRequest): ConversationDurableSummary | undefined {
  return input.memoryContext?.durableSummary ?? (input.summary ? legacySummary(input) : undefined);
}

function newerSummary(
  current: ConversationDurableSummary | undefined,
  incoming: ConversationDurableSummary | undefined,
): ConversationDurableSummary | undefined {
  if (!incoming) return current;
  if (!current) return incoming;
  if (incoming.coveredThroughTurn > current.coveredThroughTurn) return incoming;
  if (incoming.coveredThroughTurn < current.coveredThroughTurn) return current;
  return incoming.revision > current.revision ? incoming : current;
}

function mergeStoredSnippets(
  current: ConversationMemorySnippet[],
  incoming: ConversationMemorySnippet[],
): ConversationMemorySnippet[] {
  const snippets = new Map(current.map((memory) => [memory.memoryId, memory]));
  for (const memory of incoming) {
    // A relevance score is scoped to the request that selected the snippet;
    // persisting it would incorrectly make the memory relevant forever.
    const { relevanceScore: _requestScore, ...stableMemory } = memory;
    snippets.set(memory.memoryId, stableMemory);
  }
  return [...snippets.values()]
    .sort((left, right) => right.recordedAt.localeCompare(left.recordedAt))
    .slice(0, STORED_SNIPPET_LIMIT);
}

function summarizeTurns(
  previous: ConversationDurableSummary | undefined,
  messages: MastraDBMessage[],
  fromTurn: number,
  throughTurn: number,
  conversationId: string,
): ConversationDurableSummary | undefined {
  const selected = messages.filter((message) => {
    const turn = messageTurnNumber(message);
    return turn != null && turn >= fromTurn && turn <= throughTurn;
  });
  const observedTurns = new Set(selected.map(messageTurnNumber).filter((turn): turn is number => turn != null));
  for (let turn = fromTurn; turn <= throughTurn; turn += 1) {
    // Never claim an audited coverage range when the bounded storage scan has
    // a hole. The next compaction attempt can retry, or H-IDE can provide a
    // server-owned summary that explicitly covers the earlier turns.
    if (!observedTurns.has(turn)) return undefined;
  }
  const lines = selected.flatMap((message) => {
    const text = messageText(message);
    const turn = messageTurnNumber(message);
    if (!text || turn == null) return [];
    const role = message.role === "user" ? "U" : "A";
    return [`T${turn}${role}: ${clampText(text, 360)}`];
  });
  if (lines.length === 0) return undefined;
  const combined = [previous?.text, ...lines].filter(Boolean).join("\n");
  const text = combined.length <= SUMMARY_CHARACTER_LIMIT
    ? combined
    : `${combined.slice(0, 5_500)}\n…[中间内容保留在审计消息中]…\n${combined.slice(-6_400)}`;
  return {
    summaryId: `summary-${sha256(conversationId).slice(0, 24)}`,
    revision: (previous?.revision ?? 0) + 1,
    text,
    coveredThroughTurn: throughTurn,
    generatedAt: new Date().toISOString(),
    method: "SERVER_COMPACTION",
    contentSha256: sha256(text),
  };
}

function dbMessage(
  scope: ConversationScope,
  role: "user" | "assistant",
  text: string,
  turnNumber: number,
  createdAt: Date,
): MastraDBMessage {
  return {
    id: randomUUID(),
    role,
    createdAt,
    threadId: scope.threadId,
    resourceId: scope.resourceId,
    content: {
      format: 2,
      parts: [{ type: "text", text }],
      metadata: {
        hIdeKind: "assistant-conversation-turn",
        turnNumber,
        authority: "CONVERSATIONAL_ONLY",
      },
    },
  };
}

function threadWithState(
  existing: StorageThreadType | null,
  scope: ConversationScope,
  state: StoredConversationMemoryState,
  now: Date,
): StorageThreadType {
  return {
    id: scope.threadId,
    resourceId: scope.resourceId,
    title: existing?.title || "H-IDE Assistant conversation",
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
    metadata: {
      ...(existing?.metadata ?? {}),
      [memoryStateKey]: state,
    },
  };
}

export function requestOnlyContext(input: ConversationTurnRequest): PreparedConversationContext {
  const durableSummary = incomingSummary(input);
  const relevantMemories = input.memoryContext?.relevantMemories ?? [];
  return {
    recentMessages: input.history,
    ...(durableSummary ? { durableSummary } : {}),
    relevantMemories,
    audit: {
      recentMessageCount: input.history.length,
      storedMessageCount: 0,
      ...(durableSummary
        ? { summaryId: durableSummary.summaryId, summaryRevision: durableSummary.revision }
        : {}),
      selectedMemoryIds: relevantMemories.map((memory) => memory.memoryId),
      retrievalMethod: relevantMemories.length ? "SERVER_SELECTED" : "NONE",
    },
  };
}

/**
 * Explicit, manual use of Mastra Memory. We intentionally do not attach this
 * memory to Agent.generate(): automatic memory injection can be interpreted as
 * system authority. Recalled text is instead serialized under an untrusted
 * data marker by ConversationRouter.
 */
export class MastraConversationMemory {
  private readonly writeQueues = new Map<string, Promise<void>>();

  constructor(private readonly memory: MemoryPort) {}

  async prepare(input: ConversationTurnRequest): Promise<PreparedConversationContext> {
    const scope = scopeFor(input);
    const thread = await this.memory.getThreadById({
      threadId: scope.threadId,
      resourceId: scope.resourceId,
    });
    if (!thread) return requestOnlyContext(input);
    const state = parseStoredState(thread, input, scope);
    const recalled = await this.memory.recall({
      threadId: scope.threadId,
      resourceId: scope.resourceId,
      perPage: RETRIEVAL_SCAN_LIMIT,
      page: 0,
      orderBy: { field: "createdAt", direction: "DESC" },
    });
    const messages = [...recalled.messages].reverse();
    const stored = storedHistory(messages);
    const recentMessages = mergeRecentHistory(stored.slice(-RECENT_MESSAGE_LIMIT), input.history);
    const durableSummary = newerSummary(state.durableSummary, incomingSummary(input));
    const olderMessages = messages.slice(0, Math.max(0, messages.length - RECENT_MESSAGE_LIMIT));
    const serverMemories = input.memoryContext?.relevantMemories ?? [];
    const relevantMemories = selectRelevantMemories(
      input.text,
      serverMemories,
      state.memorySnippets,
      olderMessages,
    );
    const hasServer = serverMemories.length > 0;
    const hasLexical = relevantMemories.some((memory) =>
      memory.kind === "CONVERSATION_EXCERPT" || !serverMemories.some((item) => item.memoryId === memory.memoryId));
    return {
      recentMessages,
      ...(durableSummary ? { durableSummary } : {}),
      relevantMemories,
      audit: {
        recentMessageCount: recentMessages.length,
        storedMessageCount: recalled.total,
        ...(durableSummary
          ? { summaryId: durableSummary.summaryId, summaryRevision: durableSummary.revision }
          : {}),
        selectedMemoryIds: relevantMemories.map((memory) => memory.memoryId),
        retrievalMethod: hasServer && hasLexical
          ? "SERVER_AND_LEXICAL_V1"
          : hasServer
            ? "SERVER_SELECTED"
            : hasLexical
              ? "LEXICAL_V1"
              : "NONE",
      },
    };
  }

  async record(
    input: ConversationTurnRequest,
    response: ConversationTurnResponse,
    context: PreparedConversationContext,
  ): Promise<void> {
    const scope = scopeFor(input);
    const previous = this.writeQueues.get(scope.threadId) ?? Promise.resolve();
    const next = previous.catch(() => undefined).then(() => this.recordLocked(input, response, context, scope));
    this.writeQueues.set(scope.threadId, next);
    try {
      await next;
    } finally {
      if (this.writeQueues.get(scope.threadId) === next) this.writeQueues.delete(scope.threadId);
    }
  }

  private async recordLocked(
    input: ConversationTurnRequest,
    response: ConversationTurnResponse,
    context: PreparedConversationContext,
    scope: ConversationScope,
  ): Promise<void> {
    const existing = await this.memory.getThreadById({
      threadId: scope.threadId,
      resourceId: scope.resourceId,
    });
    const state = parseStoredState(existing, input, scope);
    const now = new Date();
    const incoming = input.memoryContext?.relevantMemories ?? [];
    const selectedSummary = newerSummary(state.durableSummary, incomingSummary(input));
    if (selectedSummary) state.durableSummary = selectedSummary;
    // A newly deployed service may join an existing H-IDE conversation whose
    // server summary already covers earlier turns. Continue numbering after
    // that audited range instead of pretending this is turn one.
    const turnNumber = Math.max(
      state.turnCount,
      selectedSummary?.coveredThroughTurn ?? 0,
    ) + 1;
    state.turnCount = turnNumber;
    state.contextRevision = Math.max(state.contextRevision, input.memoryContext?.revision ?? 0);
    state.memorySnippets = mergeStoredSnippets(state.memorySnippets, incoming);
    state.audit.push({
      eventId: randomUUID(),
      eventType: "CONTEXT_USED",
      recordedAt: now.toISOString(),
      turnNumber,
      ...(context.durableSummary ? { summaryRevision: context.durableSummary.revision } : {}),
      selectedMemoryIds: context.audit.selectedMemoryIds,
    });

    if (!existing) {
      await this.memory.saveThread({ thread: threadWithState(null, scope, state, now) });
    }
    await this.memory.saveMessages({
      messages: [
        dbMessage(scope, "user", input.text, turnNumber, now),
        dbMessage(scope, "assistant", response.assistantMessage, turnNumber, new Date(now.getTime() + 1)),
      ],
    });

    const previousCoverage = state.durableSummary?.coveredThroughTurn ?? 0;
    const targetCoverage = turnNumber - Math.floor(RECENT_MESSAGE_LIMIT / 2);
    const shouldCompact = turnNumber >= COMPACTION_START_TURNS
      && targetCoverage > previousCoverage
      && (targetCoverage - previousCoverage >= COMPACTION_INTERVAL_TURNS
        || previousCoverage === 0);
    if (shouldCompact) {
      const recalled = await this.memory.recall({
        threadId: scope.threadId,
        resourceId: scope.resourceId,
        perPage: RETRIEVAL_SCAN_LIMIT,
        page: 0,
        orderBy: { field: "createdAt", direction: "DESC" },
      });
      const messages = [...recalled.messages].reverse();
      const fromTurn = previousCoverage + 1;
      const compacted = summarizeTurns(
        state.durableSummary,
        messages,
        fromTurn,
        targetCoverage,
        input.conversationId,
      );
      if (compacted) {
        const selected = messages.filter((message) => {
          const turn = messageTurnNumber(message);
          return turn != null && turn >= fromTurn && turn <= targetCoverage;
        });
        state.durableSummary = compacted;
        state.audit.push({
          eventId: randomUUID(),
          eventType: "COMPACTION",
          recordedAt: new Date().toISOString(),
          turnNumber,
          summaryRevision: compacted.revision,
          coveredFromTurn: fromTurn,
          coveredThroughTurn: targetCoverage,
          sourceDigest: sha256(selected.map((message) => `${message.id}:${messageText(message)}`).join("\n")),
        });
      }
    }
    state.audit = state.audit.slice(-AUDIT_EVENT_LIMIT);
    await this.memory.saveThread({
      thread: threadWithState(existing, scope, state, new Date()),
    });
  }

  /** Test/diagnostic view. Raw messages remain in Mastra storage for audit. */
  async inspect(rawInput: unknown): Promise<StoredConversationMemoryState | undefined> {
    const input = conversationTurnRequestSchema.parse(rawInput);
    const scope = scopeFor(input);
    const thread = await this.memory.getThreadById({
      threadId: scope.threadId,
      resourceId: scope.resourceId,
    });
    return thread ? parseStoredState(thread, input, scope) : undefined;
  }
}

export const conversationMemoryInternals = {
  lexicalScore,
  mergeRecentHistory,
  scopeFor,
  summarizeTurns,
};

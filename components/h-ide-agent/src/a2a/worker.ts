import { randomUUID } from "node:crypto";
import { A2AAgent } from "@mastra/core/a2a";
import { planFragmentSchema, planningEnvelopeSchema, type PlanFragment } from "../contracts.js";

export interface RemoteWorkerConfig {
  id: string;
  name: string;
  description: string;
  url: string;
  token?: string | undefined;
  timeoutMs: number;
}

export function validateRemoteWorkerUrl(url: string, allowedOrigins: string[]): URL {
  const parsed = new URL(url);
  const allowlist = new Set(allowedOrigins.map((item) => new URL(item).origin));
  if (!allowlist.has(parsed.origin)) {
    throw new Error(`remote A2A origin is not in the allowlist: ${parsed.origin}`);
  }
  if (parsed.protocol !== "https:" && parsed.hostname !== "127.0.0.1" && parsed.hostname !== "localhost") {
    throw new Error("remote A2A workers must use HTTPS outside localhost");
  }
  return parsed;
}

export class MastraA2AWorker {
  readonly id: string;
  private readonly agent: A2AAgent;

  constructor(
    private readonly config: RemoteWorkerConfig,
    fetchImplementation: typeof fetch = fetch,
  ) {
    this.id = config.id;
    const configuredOrigin = new URL(config.url).origin;
    const guardedFetch: typeof fetch = async (input, init) => {
      const requestUrl = new URL(
        typeof input === "string" || input instanceof URL ? String(input) : input.url,
      );
      if (requestUrl.origin !== configuredOrigin) {
        throw new Error(
          `remote A2A request left its configured origin: ${requestUrl.origin}`,
        );
      }
      const response = await fetchImplementation(input, { ...init, redirect: "manual" });
      if (response.status >= 300 && response.status < 400) {
        throw new Error("remote A2A redirects are not allowed");
      }
      return response;
    };
    this.agent = new A2AAgent({
      id: config.id,
      name: config.name,
      description: config.description,
      url: config.url,
      headers: config.token ? { Authorization: `Bearer ${config.token}` } : {},
      timeoutMs: config.timeoutMs,
      retries: 0,
      fetch: guardedFetch,
    });
  }

  get subAgent(): A2AAgent {
    return this.agent;
  }

  async run(input: unknown, signal: AbortSignal): Promise<PlanFragment> {
    if (signal.aborted) throw new DOMException("remote worker aborted", "AbortError");
    const envelope = planningEnvelopeSchema.parse(input);
    const output = await this.agent.generate(
      [
        "Return exactly one JSON PlanFragment. Do not use markdown fences or prose.",
        `Your agentId must be ${this.config.id}.`,
        "You may only propose logical commands present in commandCatalog.",
        JSON.stringify(envelope),
      ].join("\n"),
      {
        runId: `a2a-${randomUUID()}`,
        abortSignal: signal,
        memory: {
          thread: envelope.taskRequest.conversationId,
          resource: envelope.taskRequest.requestedBy,
        },
      },
    );
    let parsed: unknown;
    try {
      parsed = JSON.parse(output.text);
    } catch (error) {
      throw new Error(`remote A2A worker returned invalid JSON: ${error}`);
    }
    const fragment = planFragmentSchema.parse(parsed);
    if (fragment.agentId !== this.config.id) {
      throw new Error(
        `remote A2A worker ${this.config.id} returned fragment for ${fragment.agentId}`,
      );
    }
    return fragment;
  }
}

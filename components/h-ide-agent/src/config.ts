import { z } from "zod";
import { validateRemoteWorkerUrl, type RemoteWorkerConfig } from "./a2a/worker.js";

const remoteWorkerSchema = z
  .object({
    id: z.string().trim().min(1).max(64).regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/),
    name: z.string().trim().min(1).max(120),
    description: z.string().trim().min(1).max(500),
    url: z.string().url(),
    token: z.string().min(1).optional(),
    timeoutMs: z.number().int().positive().max(120_000).default(45_000),
  })
  .strict();

export interface AppConfig {
  port: number;
  host: string;
  internalToken: string;
  internalTokenConfigured: boolean;
  jobDatabaseUrl: string;
  memoryDatabaseUrl: string;
  jobDatabaseAuthToken?: string;
  memoryDatabaseAuthToken?: string;
  supervisorModel: string;
  workerModel: string;
  structuringModel: string;
  modelsConfigured: boolean;
  modelCredentialReasons: string[];
  remoteWorkers: RemoteWorkerConfig[];
  allowedA2AOrigins: string[];
  visionUrl: string;
  visionAgentToken: string;
  visionSourceMode: "live" | "replay" | "synthetic";
  visionSensorIds: string[];
  visionVlmProvider: string;
  visionVlmModel: string;
  visionPromptVersion: string;
  visionAlgorithmVersion: string;
  visionApproachPolicyVersion: string;
  visionApproachPolicySha256: string;
}

function nonEmptyCsv(raw: string | undefined, fallback: string[]): string[] {
  const values = (raw ?? "").split(",").map((item) => item.trim()).filter(Boolean);
  return values.length ? values : fallback;
}

function parseRemoteWorkers(raw: string | undefined): RemoteWorkerConfig[] {
  if (!raw?.trim()) return [];
  const workers = z.array(remoteWorkerSchema).max(8).parse(JSON.parse(raw));
  const reserved = new Set([
    "motion-worker",
    "manipulation-worker",
    "vision-worker",
    "h-ide-supervisor",
    "plan-structuring-agent",
  ]);
  const seen = new Set<string>();
  for (const worker of workers) {
    if (reserved.has(worker.id)) {
      throw new Error(`external worker ID is reserved: ${worker.id}`);
    }
    if (seen.has(worker.id)) {
      throw new Error(`external worker IDs must be unique: ${worker.id}`);
    }
    seen.add(worker.id);
  }
  return workers;
}

export function loadConfig(environment: NodeJS.ProcessEnv = process.env): AppConfig {
  const suppliedInternalToken = environment.MASTRA_INTERNAL_TOKEN?.trim();
  const internalToken = suppliedInternalToken || "dev-internal-token";
  const internalTokenConfigured = Boolean(
    suppliedInternalToken &&
    suppliedInternalToken.length >= 32 &&
    suppliedInternalToken !== "dev-internal-token" &&
    !suppliedInternalToken.startsWith("replace-with-"),
  );
  const allowedA2AOrigins = (environment.A2A_ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const remoteWorkers = parseRemoteWorkers(environment.A2A_WORKERS_JSON);
  for (const worker of remoteWorkers) validateRemoteWorkerUrl(worker.url, allowedA2AOrigins);
  const supervisorModel = environment.SUPERVISOR_MODEL?.trim() || "openai/gpt-5-mini";
  const workerModel = environment.WORKER_MODEL?.trim() || supervisorModel;
  const structuringModel = environment.STRUCTURING_MODEL?.trim() || supervisorModel;
  const configuredModels = [supervisorModel, workerModel, structuringModel];
  const providerKeys: Record<string, string> = {
    openai: "OPENAI_API_KEY",
    deepseek: "DEEPSEEK_API_KEY",
    alibaba: "DASHSCOPE_API_KEY",
    "alibaba-cn": "DASHSCOPE_API_KEY",
    anthropic: "ANTHROPIC_API_KEY",
    google: "GOOGLE_GENERATIVE_AI_API_KEY",
    groq: "GROQ_API_KEY",
    mistral: "MISTRAL_API_KEY",
  };
  const modelCredentialReasons = [
    ...new Set(
      configuredModels
        .map((model) => model.split("/", 1)[0]!.toLowerCase())
        .map((provider) => providerKeys[provider])
        .filter((key): key is string => Boolean(key))
        .filter((key) => !environment[key]?.trim())
        .map((key) => `${key} is required by configured models`),
    ),
  ];
  const port = Number.parseInt(environment.PORT ?? "", 10);
  const config: AppConfig = {
    port: Number.isFinite(port) && port > 0 ? port : 4111,
    host: environment.HOST?.trim() || "127.0.0.1",
    internalToken,
    internalTokenConfigured,
    jobDatabaseUrl: environment.PLANNING_DATABASE_URL?.trim() || "file:./data/planning.db",
    memoryDatabaseUrl: environment.MASTRA_MEMORY_URL?.trim() || "file:./data/memory.db",
    supervisorModel,
    workerModel,
    structuringModel,
    modelsConfigured: Boolean(
      environment.SUPERVISOR_MODEL?.trim() && environment.STRUCTURING_MODEL?.trim(),
    ),
    modelCredentialReasons,
    remoteWorkers,
    allowedA2AOrigins,
    visionUrl: environment.HIDE_VISION_URL?.trim() || "http://127.0.0.1:8092",
    visionAgentToken: environment.HIDE_VISION_AGENT_TOKEN?.trim() || "dev-agent-token",
    visionSourceMode: z.enum(["live", "replay", "synthetic"]).parse(
      environment.HIDE_VISION_SOURCE_MODE?.trim() || "synthetic",
    ),
    visionSensorIds: nonEmptyCsv(environment.HIDE_VISION_SENSOR_IDS, ["head_rgbd"]),
    visionVlmProvider: environment.HIDE_VISION_VLM_PROVIDER?.trim() || "FakeVlmProvider",
    visionVlmModel: environment.HIDE_VISION_VLM_MODEL?.trim() || "fake-fixture-v1",
    visionPromptVersion: environment.HIDE_VISION_PROMPT_VERSION?.trim() || "visual-grounding-v1",
    visionAlgorithmVersion: environment.HIDE_VISION_ALGORITHM_VERSION?.trim() || "rgbd-median-v1",
    visionApproachPolicyVersion: environment.HIDE_VISION_APPROACH_POLICY_VERSION?.trim() || "v1",
    visionApproachPolicySha256:
      environment.HIDE_VISION_APPROACH_POLICY_SHA256?.trim() ||
      "a97bcf51b108b7c5c5241a82e2710eadb2619f12a1aba6e328d77b5474f35fc0",
  };
  if (!/^[0-9a-f]{64}$/.test(config.visionApproachPolicySha256)) {
    throw new Error("HIDE_VISION_APPROACH_POLICY_SHA256 must be a lowercase SHA-256 hash");
  }
  const jobAuth = environment.PLANNING_DATABASE_AUTH_TOKEN?.trim();
  const memoryAuth = environment.MASTRA_MEMORY_AUTH_TOKEN?.trim();
  if (jobAuth) config.jobDatabaseAuthToken = jobAuth;
  if (memoryAuth) config.memoryDatabaseAuthToken = memoryAuth;
  return config;
}

import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createClient } from "@libsql/client";
import { Mastra } from "@mastra/core/mastra";
import { loadConfig } from "../config.js";
import { createInternalAuth } from "../http/auth.js";
import { createHealthRoutes, createPlanningRoutes } from "../http/routes.js";
import { PlanningTurnRepository } from "../jobs/repository.js";
import { PlanningTurnService, type Planner } from "../jobs/service.js";
import { createAgentSystem } from "./agents.js";
import { HttpVisionResolutionClient } from "../vision/client.js";
import { VisionWorker } from "../vision/worker.js";
import { createMissionRoutes } from "../mission/routes.js";

function ensureLocalDatabaseDirectory(url: string) {
  if (!url.startsWith("file:") || url.startsWith("file::memory:")) return;
  const path = url.slice("file:".length).split("?", 1)[0];
  if (!path) return;
  mkdirSync(dirname(resolve(path)), { recursive: true });
}

export const config = loadConfig();
ensureLocalDatabaseDirectory(config.jobDatabaseUrl);
ensureLocalDatabaseDirectory(config.memoryDatabaseUrl);

const agentSystem = createAgentSystem(config);
const visualStageResolver = new VisionWorker(
  new HttpVisionResolutionClient(config.visionUrl, config.visionAgentToken),
);
const unavailablePlanner: Planner = {
  async plan() {
    throw new Error(
      "SUPERVISOR_MODEL and STRUCTURING_MODEL must be configured before planning is available",
    );
  },
};
const jobClient = createClient({
  url: config.jobDatabaseUrl,
  ...(config.jobDatabaseAuthToken ? { authToken: config.jobDatabaseAuthToken } : {}),
});
export const planningRepository = new PlanningTurnRepository(jobClient);
export const planningService = new PlanningTurnService(
  planningRepository,
  config.modelsConfigured ? agentSystem.planner : unavailablePlanner,
);

const readiness = () => {
  const reasons: string[] = [];
  if (!config.modelsConfigured) reasons.push("planning models are not configured");
  reasons.push(...config.modelCredentialReasons);
  if (!config.internalTokenConfigured) {
    reasons.push("MASTRA_INTERNAL_TOKEN must be a non-placeholder value of at least 32 characters");
  }
  return { ready: reasons.length === 0, reasons };
};

export const mastra = new Mastra({
  agents: agentSystem.agents,
  storage: agentSystem.memoryStore,
  server: {
    host: config.host,
    port: config.port,
    auth: createInternalAuth(config.internalToken),
    cors: false,
    apiRoutes: [
      ...createHealthRoutes(readiness),
      ...createPlanningRoutes(planningService, config.internalToken, visualStageResolver),
      ...createMissionRoutes(agentSystem.missionPlanner, config.internalToken),
    ],
  },
});

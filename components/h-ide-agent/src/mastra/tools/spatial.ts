import { randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import {
  navigationPayloadSchema,
  sceneContextSchema,
  spatialIntentSchema,
  type NavigationPayload,
  type SceneContext,
  type SpatialIntent,
} from "../../contracts.js";

export const SPATIAL_CONTEXT_KEY = "h-ide-scene-context";
export const NAVIGATION_COMMAND_KEY = "navigation.navigate_to_pose";
export const RESOLVER_VERSION = "scene-core/1" as const;

// DeepSeek requires every function's parameters schema to have an explicit
// object root. A discriminated union is valid JSON Schema, but Zod emits it as
// a root-level `anyOf` without `type`, which DeepSeek rejects before inference.
// Keep the precise intent union nested under one stable object property.
export const spatialResolverInputSchema = z.object({
  intent: spatialIntentSchema,
});

function normalizeYaw(angle: number) {
  const normalized = ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return Object.is(normalized, -0) ? 0 : normalized;
}

function contains(context: SceneContext, point: { x: number; y: number; z: number }) {
  return point.x >= context.bounds.min.x && point.x <= context.bounds.max.x
    && point.y >= context.bounds.min.y && point.y <= context.bounds.max.y
    && point.z >= context.bounds.min.z && point.z <= context.bounds.max.z;
}

export function resolveSpatialTarget(
  rawContext: SceneContext,
  rawIntent: SpatialIntent,
  resolutionId = `resolution-${randomUUID()}`,
  computedAt = new Date().toISOString(),
): NavigationPayload {
  const context = sceneContextSchema.parse(rawContext);
  const intent = spatialIntentSchema.parse(rawIntent);
  const source = context.currentPose ?? null;
  let goalPose: NavigationPayload["goalPose"];
  if (intent.kind === "relative") {
    if (!source) throw new Error("relative intent requires currentPose");
    const cosine = Math.cos(source.yaw);
    const sine = Math.sin(source.yaw);
    goalPose = {
      frameId: "map",
      x: source.x + intent.forwardMeters * cosine - intent.lateralMeters * sine,
      y: source.y + intent.forwardMeters * sine + intent.lateralMeters * cosine,
      z: source.z,
      yaw: normalizeYaw(source.yaw + intent.yawDeltaRad),
    };
  } else if (intent.kind === "semantic") {
    const region = context.annotations.find((item) => item.locationId === intent.locationId);
    if (!region) throw new Error(`unknown semantic locationId: ${intent.locationId}`);
    goalPose = region.entryPose;
  } else {
    if (!context.selectedPoint) throw new Error("selected_point intent requires SceneContext.selectedPoint");
    const selected = context.selectedPoint;
    if (
      Math.abs(selected.x - intent.pointMap.x) > 1e-9
      || Math.abs(selected.y - intent.pointMap.y) > 1e-9
      || Math.abs(selected.z - intent.pointMap.z) > 1e-9
    ) {
      throw new Error("selected_point pointMap must match SceneContext.selectedPoint");
    }
    if (intent.yaw == null && !source) {
      throw new Error("selected_point intent requires yaw when currentPose is unavailable");
    }
    goalPose = {
      frameId: "map",
      ...selected,
      yaw: intent.yaw == null
        ? normalizeYaw(Math.atan2(selected.y - source!.y, selected.x - source!.x))
        : normalizeYaw(intent.yaw),
    };
  }
  if (!contains(context, goalPose)) throw new Error("resolved goal is outside scene bounds");
  return navigationPayloadSchema.parse({
    goalPose,
    spatialResolution: {
      resolutionId,
      sceneRef: context.sceneRef,
      intent,
      sourcePose: source,
      goalPose,
      resolverVersion: RESOLVER_VERSION,
      warnings: [],
      computedAt,
    },
  });
}

export function validateNavigationPayload(
  rawContext: SceneContext,
  rawPayload: unknown,
): NavigationPayload {
  const context = sceneContextSchema.parse(rawContext);
  const payload = navigationPayloadSchema.parse(rawPayload);
  const resolution = payload.spatialResolution;
  if (!isDeepStrictEqual(resolution.sceneRef, context.sceneRef)) {
    throw new Error("navigation payload uses a different scene revision");
  }
  if (!isDeepStrictEqual(resolution.sourcePose ?? null, context.currentPose ?? null)) {
    throw new Error("navigation payload uses a different robot pose snapshot");
  }
  const snapshot = { ...context, currentPose: resolution.sourcePose ?? null };
  const expected = resolveSpatialTarget(
    snapshot,
    resolution.intent,
    resolution.resolutionId,
    resolution.computedAt,
  );
  if (!isDeepStrictEqual(expected, payload)) {
    throw new Error("navigation payload was not produced by the deterministic spatial resolver");
  }
  return payload;
}

export const spatialResolverTool = createTool({
  id: "resolve_spatial_target",
  description:
    "Resolve one relative, semantic-location, or selected-map-point intent into an immutable ROS map navigation payload. It never plans a path or controls a robot.",
  inputSchema: spatialResolverInputSchema,
  outputSchema: navigationPayloadSchema,
  execute: async ({ intent }, { requestContext }) => {
    const context = requestContext?.get(SPATIAL_CONTEXT_KEY);
    if (!context) throw new Error("no published H-IDE scene context is attached to this planning turn");
    return resolveSpatialTarget(sceneContextSchema.parse(context), intent);
  },
});

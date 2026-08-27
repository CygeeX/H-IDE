import { describe, expect, it } from "vitest";
import { z } from "zod";
import { sceneContextSchema } from "../src/contracts.js";
import {
  resolveSpatialTarget,
  spatialResolverInputSchema,
  validateNavigationPayload,
} from "../src/mastra/tools/spatial.js";

function context() {
  return sceneContextSchema.parse({
    sceneRef: { sceneId: "scene-demo", revision: 1 },
    bounds: { min: { x: -20, y: -20, z: -1 }, max: { x: 20, y: 20, z: 3 } },
    annotations: [{
      locationId: "kitchen",
      name: "厨房",
      aliases: ["kitchen"],
      polygonXY: [{ x: 2, y: 2 }, { x: 6, y: 2 }, { x: 6, y: 6 }],
      zMin: 0,
      zMax: 2.5,
      entryPose: { frameId: "map", x: 2, y: 3, z: 0, yaw: 0 },
    }],
    currentPose: {
      frameId: "map",
      x: 1,
      y: 2,
      z: 0,
      yaw: Math.PI / 2,
      observedAt: "2026-08-10T10:00:00+08:00",
    },
    selectedPoint: { x: 4, y: 5, z: 0 },
  });
}

describe("deterministic Mastra spatial tool", () => {
  it("publishes a DeepSeek-compatible object-root function schema", () => {
    const schema = z.toJSONSchema(spatialResolverInputSchema);
    expect(schema.type).toBe("object");
    expect(schema.properties).toHaveProperty("intent");
  });

  it("resolves relative and semantic intents in the ROS map frame", () => {
    const relative = resolveSpatialTarget(
      context(),
      { kind: "relative", forwardMeters: 5, lateralMeters: 0, yawDeltaRad: 0 },
      "resolution-relative",
      "2026-08-10T02:00:00Z",
    );
    expect(relative.goalPose.x).toBeCloseTo(1);
    expect(relative.goalPose.y).toBeCloseTo(7);
    expect(relative.goalPose.frameId).toBe("map");

    const semantic = resolveSpatialTarget(
      context(),
      { kind: "semantic", locationId: "kitchen" },
      "resolution-kitchen",
      "2026-08-10T02:00:00Z",
    );
    expect(semantic.goalPose).toEqual({ frameId: "map", x: 2, y: 3, z: 0, yaw: 0 });
  });

  it("rejects coordinates edited after deterministic resolution", () => {
    const payload = resolveSpatialTarget(
      context(),
      { kind: "selected_point", pointMap: { x: 4, y: 5, z: 0 } },
      "resolution-selected",
      "2026-08-10T02:00:00Z",
    );
    const tampered = structuredClone(payload);
    tampered.goalPose.x += 1;
    tampered.spatialResolution.goalPose.x += 1;
    expect(() => validateNavigationPayload(context(), tampered)).toThrow(/deterministic|resolver/i);
  });
});

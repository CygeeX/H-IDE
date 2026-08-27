import { describe, expect, it } from "vitest";
import {
  boundsFromPositionRange,
  mapPoseForBounds,
  perspectivePoseForBounds,
  yawForNorthVector,
} from "./potree-camera";

const bounds = {
  min: { x: -7.991, y: -27.285, z: -3.05 },
  max: { x: 30.466, y: 14.811, z: 2.861 },
};

function subtract(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number },
) {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function dot(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number },
) {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function normalize(vector: { x: number; y: number; z: number }) {
  const length = Math.hypot(vector.x, vector.y, vector.z);
  return { x: vector.x / length, y: vector.y / length, z: vector.z / length };
}

function cross(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number },
) {
  return normalize({
    x: left.y * right.z - left.z * right.y,
    y: left.z * right.x - left.x * right.z,
    z: left.x * right.y - left.y * right.x,
  });
}

describe("Potree camera framing", () => {
  it("reads the tight position range and rejects malformed ranges", () => {
    expect(boundsFromPositionRange([
      [bounds.min.x, bounds.min.y, bounds.min.z],
      [bounds.max.x, bounds.max.y, bounds.max.z],
    ])).toEqual(bounds);
    expect(boundsFromPositionRange([0, 1])).toBeNull();
    expect(boundsFromPositionRange([[1, 1, 1], [1, 1, 1]])).toBeNull();
  });

  it("frames every tight-bound corner in the initial 3D perspective view", () => {
    const aspect = 16 / 9;
    const pose = perspectivePoseForBounds(bounds, "isometric", aspect);
    const forward = normalize(subtract(pose.target, pose.position));
    const right = cross(forward, { x: 0, y: 0, z: 1 });
    const up = cross(right, forward);
    const tanVertical = Math.tan(30 * Math.PI / 180);
    const tanHorizontal = tanVertical * aspect;

    expect(pose.target.z).toBeCloseTo((bounds.min.z + bounds.max.z) / 2);
    expect(pose.target.z).toBeLessThan(1);
    for (const x of [bounds.min.x, bounds.max.x]) {
      for (const y of [bounds.min.y, bounds.max.y]) {
        for (const z of [bounds.min.z, bounds.max.z]) {
          const fromCamera = subtract({ x, y, z }, pose.position);
          const depth = dot(fromCamera, forward);
          expect(depth).toBeGreaterThan(0);
          expect(Math.abs(dot(fromCamera, right)) / depth).toBeLessThanOrEqual(tanHorizontal / 1.14);
          expect(Math.abs(dot(fromCamera, up)) / depth).toBeLessThanOrEqual(tanVertical / 1.14);
        }
      }
    }
  });

  it("fits the XY range in north-up orthographic mode", () => {
    const aspect = 1.5;
    const yaw = yawForNorthVector({ x: -1, y: 0, z: 0 });
    const pose = mapPoseForBounds(bounds, aspect, yaw);
    const radius = pose.position.z - pose.target.z;
    expect(radius).toBeGreaterThanOrEqual((bounds.max.x - bounds.min.x) / 2 * 1.15);
    expect(radius / aspect).toBeGreaterThanOrEqual((bounds.max.y - bounds.min.y) / 2 * 1.15);
    expect(pose.yaw).toBeCloseTo(Math.PI / 2);
  });
});

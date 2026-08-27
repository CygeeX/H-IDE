import type { Bounds3D, Point3D } from "./contracts";

export type PointCloudViewMode = "map" | "space";
export type PerspectivePreset = "isometric" | "front" | "side";

export interface CameraPose {
  position: Point3D;
  target: Point3D;
  yaw?: number;
}

const MIN_DISTANCE = 0.5;
const DEFAULT_MARGIN = 1.15;

function finitePoint(value: unknown): value is [number, number, number] {
  return Array.isArray(value)
    && value.length >= 3
    && value.slice(0, 3).every((coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate));
}

export function boundsFromPositionRange(range: unknown): Bounds3D | null {
  if (!Array.isArray(range) || range.length !== 2 || !finitePoint(range[0]) || !finitePoint(range[1])) {
    return null;
  }
  const [minimum, maximum] = range;
  if (maximum.some((coordinate, index) => coordinate < minimum[index])) return null;
  if (maximum.every((coordinate, index) => Math.abs(coordinate - minimum[index]) < 1e-9)) return null;
  return {
    min: { x: minimum[0], y: minimum[1], z: minimum[2] },
    max: { x: maximum[0], y: maximum[1], z: maximum[2] },
  };
}

export function boundsCenter(bounds: Bounds3D): Point3D {
  return {
    x: (bounds.min.x + bounds.max.x) / 2,
    y: (bounds.min.y + bounds.max.y) / 2,
    z: (bounds.min.z + bounds.max.z) / 2,
  };
}

export function boundsDiagonal(bounds: Bounds3D): number {
  return Math.hypot(
    bounds.max.x - bounds.min.x,
    bounds.max.y - bounds.min.y,
    bounds.max.z - bounds.min.z,
  );
}

function subtract(left: Point3D, right: Point3D): Point3D {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function scale(vector: Point3D, factor: number): Point3D {
  return { x: vector.x * factor, y: vector.y * factor, z: vector.z * factor };
}

function add(left: Point3D, right: Point3D): Point3D {
  return { x: left.x + right.x, y: left.y + right.y, z: left.z + right.z };
}

function dot(left: Point3D, right: Point3D): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function cross(left: Point3D, right: Point3D): Point3D {
  return {
    x: left.y * right.z - left.z * right.y,
    y: left.z * right.x - left.x * right.z,
    z: left.x * right.y - left.y * right.x,
  };
}

function normalize(vector: Point3D): Point3D | null {
  const length = Math.hypot(vector.x, vector.y, vector.z);
  if (!Number.isFinite(length) || length < 1e-9) return null;
  return scale(vector, 1 / length);
}

function corners(bounds: Bounds3D): Point3D[] {
  const values: Point3D[] = [];
  for (const x of [bounds.min.x, bounds.max.x]) {
    for (const y of [bounds.min.y, bounds.max.y]) {
      for (const z of [bounds.min.z, bounds.max.z]) values.push({ x, y, z });
    }
  }
  return values;
}

function fromTargetForPreset(preset: PerspectivePreset): Point3D {
  if (preset === "front") return { x: 0, y: -1, z: 0 };
  if (preset === "side") return { x: 1, y: 0, z: 0 };
  const azimuth = -Math.PI / 4;
  const elevation = 35 * Math.PI / 180;
  return {
    x: Math.cos(elevation) * Math.cos(azimuth),
    y: Math.cos(elevation) * Math.sin(azimuth),
    z: Math.sin(elevation),
  };
}

export function perspectivePoseForDirection(
  bounds: Bounds3D,
  cameraFromTarget: Point3D,
  aspect: number,
  verticalFovDegrees = 60,
  margin = DEFAULT_MARGIN,
): CameraPose {
  const fromTarget = normalize(cameraFromTarget) ?? fromTargetForPreset("isometric");
  const forward = scale(fromTarget, -1);
  let right = normalize(cross(forward, { x: 0, y: 0, z: 1 }));
  if (!right) right = { x: 1, y: 0, z: 0 };
  const up = normalize(cross(right, forward)) ?? { x: 0, y: 1, z: 0 };
  const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const verticalHalfFov = Math.max(1, Math.min(89, verticalFovDegrees / 2)) * Math.PI / 180;
  const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * safeAspect);
  const target = boundsCenter(bounds);
  let distance = MIN_DISTANCE;

  for (const corner of corners(bounds)) {
    const relative = subtract(corner, target);
    const forwardOffset = dot(relative, forward);
    const horizontalDistance = Math.abs(dot(relative, right)) * margin / Math.tan(horizontalHalfFov)
      - forwardOffset;
    const verticalDistance = Math.abs(dot(relative, up)) * margin / Math.tan(verticalHalfFov)
      - forwardOffset;
    distance = Math.max(distance, horizontalDistance, verticalDistance);
  }

  distance = Math.max(distance, boundsDiagonal(bounds) * 0.05, MIN_DISTANCE);
  return { position: add(target, scale(fromTarget, distance)), target };
}

export function perspectivePoseForBounds(
  bounds: Bounds3D,
  preset: PerspectivePreset,
  aspect: number,
  verticalFovDegrees = 60,
): CameraPose {
  return perspectivePoseForDirection(
    bounds,
    fromTargetForPreset(preset),
    aspect,
    verticalFovDegrees,
  );
}

export function mapPoseForBounds(
  bounds: Bounds3D,
  aspect: number,
  northYaw = 0,
  margin = DEFAULT_MARGIN,
): CameraPose {
  const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const width = bounds.max.x - bounds.min.x;
  const height = bounds.max.y - bounds.min.y;
  const radius = Math.max(width / 2, height * safeAspect / 2, MIN_DISTANCE) * margin;
  const target = boundsCenter(bounds);
  return {
    position: { x: target.x, y: target.y, z: target.z + radius },
    target,
    yaw: northYaw,
  };
}

export function focusPose(current: CameraPose, target: Point3D, distance: number): CameraPose {
  const fromTarget = normalize(subtract(current.position, current.target))
    ?? fromTargetForPreset("isometric");
  const safeDistance = Number.isFinite(distance) ? Math.max(distance, MIN_DISTANCE) : MIN_DISTANCE;
  return {
    position: add(target, scale(fromTarget, safeDistance)),
    target: { ...target },
    yaw: current.yaw,
  };
}

export function yawForNorthVector(vector: Point3D): number {
  if (!Number.isFinite(vector.x) || !Number.isFinite(vector.y) || Math.hypot(vector.x, vector.y) < 1e-9) {
    return 0;
  }
  return Math.atan2(-vector.x, vector.y);
}

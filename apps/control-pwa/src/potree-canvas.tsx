import { useCallback, useEffect, useRef, useState } from "react";
import type { Bounds3D, Point3D } from "./contracts";
import {
  boundsDiagonal,
  boundsFromPositionRange,
  focusPose,
  mapPoseForBounds,
  perspectivePoseForBounds,
  perspectivePoseForDirection,
  type CameraPose,
  type PointCloudViewMode,
  type PerspectivePreset,
} from "./potree-camera";
import {
  loadPotreeRuntime,
  type PotreeMeasurement,
  type PotreeViewerInstance,
} from "./potree-runtime";

export interface SceneOverlay {
  name: string;
  kind: "point" | "polygon" | "direction";
  points: Point3D[];
  color?: number;
}

export interface RobotSceneMarker {
  position: Point3D;
  heading: Point3D;
  stale: boolean;
  ageLabel: string;
}

type PointColorMode = "intensity" | "elevation";

const VIEW_MODE_STORAGE_PREFIX = "h-ide:potree-view-mode:v1:";
const CAMERA_TRANSITION_MS = 300;

function readStoredMode(sceneKey: string): PointCloudViewMode {
  try {
    return window.localStorage.getItem(`${VIEW_MODE_STORAGE_PREFIX}${sceneKey}`) === "map"
      ? "map"
      : "space";
  } catch {
    return "space";
  }
}

function storeMode(sceneKey: string, mode: PointCloudViewMode): void {
  try {
    window.localStorage.setItem(`${VIEW_MODE_STORAGE_PREFIX}${sceneKey}`, mode);
  } catch {
    // Private browsing and storage policies must not break scene navigation.
  }
}

function aspectOf(element: HTMLElement): number {
  return element.clientWidth > 0 && element.clientHeight > 0
    ? element.clientWidth / element.clientHeight
    : 16 / 9;
}

function readCameraPose(viewer: PotreeViewerInstance): CameraPose {
  const view = viewer.scene.view;
  const pivot = view.getPivot();
  return {
    position: { x: view.position.x, y: view.position.y, z: view.position.z },
    target: { x: pivot.x, y: pivot.y, z: pivot.z },
    yaw: view.yaw,
  };
}

function writeCameraPose(viewer: PotreeViewerInstance, pose: CameraPose): void {
  const view = viewer.scene.view;
  view.position.set(pose.position.x, pose.position.y, pose.position.z);
  view.lookAt(pose.target.x, pose.target.y, pose.target.z);
  if (typeof pose.yaw === "number" && Number.isFinite(pose.yaw)) view.yaw = pose.yaw;
  viewer.setMoveSpeed(Math.max(view.radius, 0.1));
}

function interpolatePoint(start: Point3D, end: Point3D, progress: number): Point3D {
  return {
    x: start.x + (end.x - start.x) * progress,
    y: start.y + (end.y - start.y) * progress,
    z: start.z + (end.z - start.z) * progress,
  };
}

function shortestAngle(start: number, end: number): number {
  return ((end - start + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

export function PotreeCanvas({
  assetBaseUrl,
  sceneKey,
  pickMode,
  overlays,
  mapNorthYaw = 0,
  robotMarker,
  robotFocusUnavailableReason,
  onPick,
}: {
  assetBaseUrl: string;
  sceneKey: string;
  pickMode: string | null;
  overlays: SceneOverlay[];
  mapNorthYaw?: number;
  robotMarker?: RobotSceneMarker;
  robotFocusUnavailableReason?: string;
  onPick: (point: Point3D) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<PotreeViewerInstance | null>(null);
  const measurementsRef = useRef<PotreeMeasurement[]>([]);
  const pickModeRef = useRef(pickMode);
  const onPickRef = useRef(onPick);
  const pointerStartRef = useRef<{ id: number; x: number; y: number } | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const tightBoundsRef = useRef<Bounds3D | null>(null);
  const viewModeRef = useRef<PointCloudViewMode>("space");
  const cameraPosesRef = useRef<Partial<Record<PointCloudViewMode, CameraPose>>>({});
  const mapNorthYawRef = useRef(mapNorthYaw);
  const maxPointBudgetRef = useRef(750_000);
  const [runtimeVersion, setRuntimeVersion] = useState(0);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [message, setMessage] = useState("正在加载分级点云…");
  const [viewMode, setViewMode] = useState<PointCloudViewMode>("space");
  const [colorMode, setColorMode] = useState<PointColorMode>("elevation");
  const [hasIntensity, setHasIntensity] = useState(false);
  const [displayDensity, setDisplayDensity] = useState(65);
  pickModeRef.current = pickMode;
  onPickRef.current = onPick;
  mapNorthYawRef.current = mapNorthYaw;

  const cancelCameraAnimation = useCallback(() => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const moveCamera = useCallback((
    viewer: PotreeViewerInstance,
    destination: CameraPose,
    animated = true,
  ) => {
    cancelCameraAnimation();
    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    const duration = animated && !reduceMotion ? CAMERA_TRANSITION_MS : 0;
    if (duration === 0) {
      writeCameraPose(viewer, destination);
      return;
    }

    const startedAt = performance.now();
    const start = readCameraPose(viewer);
    const yawDelta = typeof destination.yaw === "number"
      ? shortestAngle(start.yaw ?? 0, destination.yaw)
      : null;
    const tick = (now: number) => {
      const linear = Math.min(1, Math.max(0, (now - startedAt) / duration));
      const eased = 1 - (1 - linear) ** 4;
      writeCameraPose(viewer, {
        position: interpolatePoint(start.position, destination.position, eased),
        target: interpolatePoint(start.target, destination.target, eased),
        yaw: yawDelta === null ? undefined : (start.yaw ?? 0) + yawDelta * eased,
      });
      if (linear < 1) animationFrameRef.current = window.requestAnimationFrame(tick);
      else animationFrameRef.current = null;
    };
    animationFrameRef.current = window.requestAnimationFrame(tick);
  }, [cancelCameraAnimation]);

  const configureMode = useCallback((viewer: PotreeViewerInstance, mode: PointCloudViewMode) => {
    const Potree = window.Potree;
    if (!Potree) return;
    viewer.orbitControls.stop();
    viewer.orbitControls.navigationMode = mode === "map" ? "map" : "orbit";
    viewer.setControls(viewer.orbitControls);
    viewer.setCameraMode(
      mode === "map" ? Potree.CameraMode.ORTHOGRAPHIC : Potree.CameraMode.PERSPECTIVE,
    );
  }, []);

  const defaultPose = useCallback((
    mode: PointCloudViewMode,
    bounds: Bounds3D,
    container: HTMLElement,
  ): CameraPose => mode === "map"
    ? mapPoseForBounds(bounds, aspectOf(container), mapNorthYawRef.current)
    : perspectivePoseForBounds(bounds, "isometric", aspectOf(container)), []);

  useEffect(() => {
    let disposed = false;
    const container = containerRef.current;
    if (!container) return;
    const initialMode = readStoredMode(sceneKey);
    viewModeRef.current = initialMode;
    cameraPosesRef.current = {};
    tightBoundsRef.current = null;
    setViewMode(initialMode);
    setState("loading");
    setMessage("正在加载 Potree 运行时与场景切片…");
    cancelCameraAnimation();
    void (async () => {
      try {
        const Potree = await loadPotreeRuntime();
        if (disposed) return;
        const compact = window.matchMedia?.("(max-width: 720px)").matches ?? false;
        const viewer = new Potree.Viewer(container, { noDragAndDrop: true });
        viewerRef.current = viewer;
        // Eye-dome lighting makes walls, furniture and floor layers legible in
        // uncoloured LiDAR scans. Keep it off on small mobile GPUs.
        viewer.setEDLEnabled(!compact);
        viewer.setFOV(60);
        maxPointBudgetRef.current = compact ? 300_000 : 750_000;
        viewer.setPointBudget(Math.round(maxPointBudgetRef.current * 0.65));
        viewer.setBackground("gradient");
        const loaded = await Promise.race([
          Potree.loadPointCloud(`${assetBaseUrl}/metadata.json`, "H-IDE scene"),
          new Promise<never>((_, reject) => window.setTimeout(
            () => reject(new Error("点云加载超时，请检查转换产物或网络")),
            30_000,
          )),
        ]);
        if (disposed) return;
        loaded.pointcloud.name = "H-IDE scene";
        loaded.pointcloud.material.size = compact ? 0.8 : 1.0;
        loaded.pointcloud.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
        loaded.pointcloud.material.shape = Potree.PointShape.CIRCLE;
        const intensityRange = loaded.pointcloud.getAttribute("intensity")?.range;
        const usableIntensity = Array.isArray(intensityRange)
          && intensityRange.length === 2
          && intensityRange.every((value) => typeof value === "number" && Number.isFinite(value))
          && (intensityRange[1] as number) > (intensityRange[0] as number);
        setHasIntensity(usableIntensity);
        if (usableIntensity) {
          const numericRange = intensityRange as [number, number];
          loaded.pointcloud.material.activeAttributeName = "intensity";
          loaded.pointcloud.material.intensityRange = numericRange;
          setColorMode("intensity");
        } else {
          loaded.pointcloud.material.activeAttributeName = "elevation";
          setColorMode("elevation");
        }
        const tightBounds = boundsFromPositionRange(
          loaded.pointcloud.getAttribute("position")?.range,
        );
        tightBoundsRef.current = tightBounds;
        viewer.scene.addPointCloud(loaded.pointcloud);
        configureMode(viewer, initialMode);
        if (tightBounds) {
          const pose = defaultPose(initialMode, tightBounds, container);
          cameraPosesRef.current[initialMode] = pose;
          moveCamera(viewer, pose, false);
          setMessage(initialMode === "map" ? "场景已加载 · 2.5D 地图" : "场景已加载 · 3D 空间");
        } else {
          if (initialMode === "map") viewer.setTopView();
          else viewer.fitToScreen(0.8);
          setMessage("场景已加载 · 使用兼容取景");
        }
        setState("ready");
        setRuntimeVersion((value) => value + 1);
      } catch (error) {
        if (!disposed) {
          setState("error");
          setMessage(error instanceof Error ? error.message : "点云查看器初始化失败");
        }
      }
    })();

    const pointerDown = (event: PointerEvent) => {
      cancelCameraAnimation();
      pointerStartRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY };
    };
    const pointerUp = (event: PointerEvent) => {
      const start = pointerStartRef.current;
      pointerStartRef.current = null;
      const viewer = viewerRef.current;
      const mode = pickModeRef.current;
      if (!start || start.id !== event.pointerId || !viewer || !mode) return;
      if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > 9) return;
      const rectangle = viewer.renderer.domElement.getBoundingClientRect();
      const Potree = window.Potree;
      if (!Potree || rectangle.width <= 0 || rectangle.height <= 0) return;
      const intersection = Potree.Utils.getMousePointCloudIntersection(
        { x: event.clientX - rectangle.left, y: event.clientY - rectangle.top },
        viewer.scene.getActiveCamera(),
        viewer,
        viewer.scene.pointclouds,
        { pickClipped: true },
      );
      if (intersection) {
        onPickRef.current({
          x: intersection.location.x,
          y: intersection.location.y,
          z: intersection.location.z,
        });
      }
    };
    container.addEventListener("pointerdown", pointerDown);
    container.addEventListener("pointerup", pointerUp);
    return () => {
      disposed = true;
      cancelCameraAnimation();
      container.removeEventListener("pointerdown", pointerDown);
      container.removeEventListener("pointerup", pointerUp);
      const viewer = viewerRef.current;
      if (viewer) {
        viewer.renderer.setAnimationLoop(null);
        viewer.renderer.dispose();
      }
      viewerRef.current = null;
      measurementsRef.current = [];
      container.replaceChildren();
    };
  }, [assetBaseUrl, cancelCameraAnimation, configureMode, defaultPose, moveCamera, sceneKey]);

  useEffect(() => {
    const pointcloud = viewerRef.current?.scene.pointclouds[0];
    if (!pointcloud || state !== "ready") return;
    pointcloud.material.activeAttributeName = colorMode;
  }, [colorMode, state]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || state !== "ready") return;
    viewer.setPointBudget(Math.max(50_000, Math.round(maxPointBudgetRef.current * displayDensity / 100)));
  }, [displayDensity, state]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || state !== "ready" || viewModeRef.current !== "map") return;
    viewer.scene.view.yaw = mapNorthYaw;
  }, [mapNorthYaw, state]);

  useEffect(() => {
    const viewer = viewerRef.current;
    const Potree = window.Potree;
    if (!viewer || !Potree || state !== "ready") return;
    for (const measurement of measurementsRef.current) {
      viewer.scene.removeMeasurement(measurement);
    }
    const rendered: PotreeMeasurement[] = [];
    for (const overlay of overlays) {
      if (!overlay.points.length) continue;
      const measurement = new Potree.Measure();
      measurement.name = overlay.name;
      measurement.closed = overlay.kind === "polygon" && overlay.points.length >= 3;
      measurement.showArea = measurement.closed;
      measurement.showDistances = false;
      measurement.showCoordinates = overlay.kind === "point";
      measurement.showEdges = overlay.kind !== "point";
      measurement.maxMarkers = overlay.points.length;
      if (typeof overlay.color === "number") measurement.color.setHex(overlay.color);
      for (const point of overlay.points) measurement.addMarker([point.x, point.y, point.z]);
      viewer.scene.addMeasurement(measurement);
      rendered.push(measurement);
    }
    measurementsRef.current = rendered;
    return () => {
      for (const measurement of rendered) viewer.scene.removeMeasurement(measurement);
      if (measurementsRef.current === rendered) measurementsRef.current = [];
    };
  }, [overlays, runtimeVersion, state]);

  const selectMode = (nextMode: PointCloudViewMode) => {
    const viewer = viewerRef.current;
    const container = containerRef.current;
    if (!viewer || !container || state !== "ready" || nextMode === viewModeRef.current) return;
    cameraPosesRef.current[viewModeRef.current] = readCameraPose(viewer);
    viewModeRef.current = nextMode;
    setViewMode(nextMode);
    storeMode(sceneKey, nextMode);
    configureMode(viewer, nextMode);
    const bounds = tightBoundsRef.current;
    const saved = cameraPosesRef.current[nextMode];
    if (saved) {
      moveCamera(viewer, nextMode === "map" ? { ...saved, yaw: mapNorthYawRef.current } : saved);
    } else if (bounds) {
      moveCamera(viewer, defaultPose(nextMode, bounds, container));
    } else if (nextMode === "map") {
      viewer.setTopView();
    } else {
      viewer.fitToScreen(0.8);
    }
    setMessage(nextMode === "map" ? "2.5D 地图模式" : "3D 空间模式");
  };

  const applyPreset = (preset: PerspectivePreset | "all") => {
    const viewer = viewerRef.current;
    const container = containerRef.current;
    if (!viewer || !container || state !== "ready") return;
    let activeMode = viewModeRef.current;
    if (preset !== "all" && activeMode !== "space") {
      cameraPosesRef.current.map = readCameraPose(viewer);
      activeMode = "space";
      viewModeRef.current = activeMode;
      setViewMode(activeMode);
      storeMode(sceneKey, activeMode);
      configureMode(viewer, activeMode);
    }

    const bounds = tightBoundsRef.current;
    if (!bounds) {
      if (activeMode === "map") viewer.setTopView();
      else if (preset === "front") viewer.setFrontView();
      else if (preset === "side") viewer.setRightView();
      else viewer.fitToScreen(0.8);
      setMessage("使用 Potree 兼容取景");
      return;
    }

    let pose: CameraPose;
    if (activeMode === "map") {
      pose = mapPoseForBounds(bounds, aspectOf(container), mapNorthYawRef.current);
    } else if (preset === "all") {
      const current = readCameraPose(viewer);
      pose = perspectivePoseForDirection(bounds, {
        x: current.position.x - current.target.x,
        y: current.position.y - current.target.y,
        z: current.position.z - current.target.z,
      }, aspectOf(container));
    } else {
      pose = perspectivePoseForBounds(bounds, preset, aspectOf(container));
    }
    moveCamera(viewer, pose);
    const labels: Record<PerspectivePreset | "all", string> = {
      isometric: "等轴视角",
      front: "正面视角",
      side: "侧面视角",
      all: "已显示全部有效点云",
    };
    setMessage(labels[preset]);
  };

  const locateRobot = () => {
    const viewer = viewerRef.current;
    if (!viewer || !robotMarker || state !== "ready") return;
    const current = readCameraPose(viewer);
    const diagonal = tightBoundsRef.current ? boundsDiagonal(tightBoundsRef.current) : 20;
    const distance = Math.min(10, Math.max(3, diagonal * 0.12));
    const pose = viewModeRef.current === "map"
      ? {
        position: {
          x: robotMarker.position.x,
          y: robotMarker.position.y,
          z: robotMarker.position.z + distance,
        },
        target: { ...robotMarker.position },
        yaw: mapNorthYawRef.current,
      }
      : focusPose(current, robotMarker.position, distance);
    moveCamera(viewer, pose);
    setMessage(robotMarker.stale
      ? `已定位机器人 · 位姿${robotMarker.ageLabel}`
      : "已定位机器人");
  };

  return (
    <div className="potree-viewer-shell">
      <div className="potree-camera-toolbar" aria-label="点云预设视角">
        <div className="potree-preset-actions">
          <button className="mini-button" type="button" disabled={state !== "ready"} onClick={() => applyPreset("isometric")}>等轴</button>
          <button className="mini-button" type="button" disabled={state !== "ready"} onClick={() => applyPreset("front")}>正面</button>
          <button className="mini-button" type="button" disabled={state !== "ready"} onClick={() => applyPreset("side")}>侧面</button>
          <button className="mini-button" type="button" disabled={state !== "ready"} onClick={() => applyPreset("all")}>查看全部</button>
        </div>
        <div className="potree-robot-actions">
          <button
            className="mini-button"
            type="button"
            disabled={state !== "ready" || !robotMarker}
            title={robotMarker ? `机器人位姿${robotMarker.ageLabel}` : robotFocusUnavailableReason}
            onClick={locateRobot}
          >定位机器人</button>
          {robotMarker ? <span className={robotMarker.stale ? "pose-age stale" : "pose-age"}>位姿{robotMarker.ageLabel}</span>
            : robotFocusUnavailableReason ? <span className="potree-tool-hint">{robotFocusUnavailableReason}</span> : null}
        </div>
      </div>
      <div className={`potree-stage ${pickMode ? "picking" : ""}`}>
        <div className="potree-render-area" ref={containerRef} aria-label="PCD 点云场景" />
        <div className="potree-mode-controls" aria-label="点云观察模式">
          <button
            className={viewMode === "map" ? "active" : ""}
            type="button"
            aria-pressed={viewMode === "map"}
            disabled={state !== "ready"}
            onClick={() => selectMode("map")}
          >2.5D 地图</button>
          <button
            className={viewMode === "space" ? "active" : ""}
            type="button"
            aria-pressed={viewMode === "space"}
            disabled={state !== "ready"}
            onClick={() => selectMode("space")}
          >3D 空间</button>
        </div>
        <div className="potree-color-controls" aria-label="点云着色方式">
          <button
            className={colorMode === "intensity" ? "active" : ""}
            type="button"
            disabled={!hasIntensity}
            onClick={() => setColorMode("intensity")}
          >强度</button>
          <button
            className={colorMode === "elevation" ? "active" : ""}
            type="button"
            onClick={() => setColorMode("elevation")}
          >高度</button>
        </div>
        <label className="potree-density-control">显示密度
          <input
            type="range"
            min="10"
            max="100"
            step="5"
            value={displayDensity}
            disabled={state !== "ready"}
            onChange={(event) => setDisplayDensity(Number(event.target.value))}
          />
          <b>{displayDensity}%</b>
        </label>
        <div className={`potree-state potree-${state}`} role="status">
          <span className="connection-dot" /> {pickMode ? `点选模式：${pickMode}` : message}
        </div>
        {state === "error" ? <div className="potree-error">{message}</div> : null}
      </div>
    </div>
  );
}

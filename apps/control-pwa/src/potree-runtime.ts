const RUNTIME_ROOT = "/vendor/potree-runtime";

export interface PotreePoint {
  x: number;
  y: number;
  z: number;
}

export interface PotreeMeasurement {
  name: string;
  closed: boolean;
  showArea: boolean;
  showDistances: boolean;
  showCoordinates: boolean;
  showEdges: boolean;
  maxMarkers: number;
  color: { setHex(value: number): void };
  addMarker(point: [number, number, number]): void;
}

export interface PotreeVector3 {
  x: number;
  y: number;
  z: number;
  set(x: number, y: number, z: number): void;
}

export interface PotreeView {
  position: PotreeVector3;
  yaw: number;
  pitch: number;
  radius: number;
  getPivot(): PotreeVector3;
  lookAt(x: number, y: number, z: number): void;
}

export interface PotreeOrbitControls {
  navigationMode: "orbit" | "map";
  stop(): void;
}

export interface PotreePointCloud {
  name: string;
  material: {
    size: number;
    pointSizeType: unknown;
    shape: unknown;
    activeAttributeName: string;
    intensityRange: [number, number];
  };
  getAttribute(name: string): { range: [unknown, unknown] } | null;
}

export interface PotreeViewerInstance {
  renderer: {
    domElement: HTMLCanvasElement;
    setAnimationLoop(callback: null): void;
    dispose(): void;
  };
  scene: {
    pointclouds: PotreePointCloud[];
    measurements: PotreeMeasurement[];
    view: PotreeView;
    addPointCloud(pointcloud: unknown): void;
    addMeasurement(measurement: PotreeMeasurement): void;
    removeMeasurement(measurement: PotreeMeasurement): void;
    getActiveCamera(): unknown;
  };
  setEDLEnabled(value: boolean): void;
  setFOV(value: number): void;
  setPointBudget(value: number): void;
  setBackground(value: string): void;
  setCameraMode(value: number): void;
  setControls(controls: PotreeOrbitControls): void;
  setMoveSpeed(value: number): void;
  setFrontView(): void;
  setRightView(): void;
  setTopView(): void;
  orbitControls: PotreeOrbitControls;
  fitToScreen(factor?: number): void;
}

interface PotreeGlobal {
  Viewer: new (element: HTMLElement, options?: Record<string, unknown>) => PotreeViewerInstance;
  Measure: new () => PotreeMeasurement;
  PointSizeType: { ADAPTIVE: unknown };
  PointShape: { CIRCLE: unknown };
  CameraMode: { ORTHOGRAPHIC: number; PERSPECTIVE: number };
  Utils: {
    getMousePointCloudIntersection(
      mouse: { x: number; y: number },
      camera: unknown,
      viewer: PotreeViewerInstance,
      pointclouds: unknown[],
      params?: Record<string, unknown>,
    ): { location: PotreePoint } | null;
  };
  loadPointCloud(path: string, name: string): Promise<{ pointcloud: PotreePointCloud }>;
}

declare global {
  interface Window {
    Potree?: PotreeGlobal;
  }
}

let runtimePromise: Promise<PotreeGlobal> | null = null;

function loadScript(path: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[data-potree-src="${path}"]`);
    if (existing?.dataset.loaded === "true") {
      resolve();
      return;
    }
    if (existing) {
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error(`无法加载 ${path}`)), { once: true });
      return;
    }
    const script = document.createElement("script");
    // Keep this URL path query-free: Potree derives its worker and resource
    // roots from document.currentScript.src by appending `/..`.
    script.src = `${RUNTIME_ROOT}/${path}`;
    script.async = false;
    script.dataset.potreeSrc = path;
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolve();
    }, { once: true });
    script.addEventListener("error", () => reject(new Error(`无法加载点云运行时：${path}`)), { once: true });
    document.head.appendChild(script);
  });
}

export function loadPotreeRuntime(): Promise<PotreeGlobal> {
  if (window.Potree) return Promise.resolve(window.Potree);
  if (!runtimePromise) {
    runtimePromise = (async () => {
      for (const script of [
        "deps/jquery.js",
        "deps/jquery-ui.js",
        "deps/binary-heap.js",
        "deps/tween.js",
        "deps/d3.js",
        "deps/proj4.js",
        "deps/openlayers.js",
        "potree.js",
      ]) {
        await loadScript(script);
      }
      if (!window.Potree) throw new Error("Potree 运行时初始化失败");
      return window.Potree;
    })().catch((error) => {
      runtimePromise = null;
      throw error;
    });
  }
  return runtimePromise;
}

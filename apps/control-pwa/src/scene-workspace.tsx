import {
  type ChangeEvent,
  type DragEvent as ReactDragEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  Bounds3D,
  Point3D,
  PoseMap,
  SceneJob,
  SceneRevision,
  SceneSummary,
  SemanticRegion,
} from "./contracts";
import {
  formatBytes,
  formatPoint,
  IDENTITY_CALIBRATION,
  transformPoint,
  uploadPcd,
  type ApiFetcher,
} from "./scene-api";
import { yawForNorthVector } from "./potree-camera";
import {
  PotreeCanvas,
  type RobotSceneMarker,
  type SceneOverlay,
} from "./potree-canvas";
import { SceneMapCanvas, type MapOverlay } from "./scene-map-canvas";

type PickMode = "ground" | "origin" | "x-axis" | "polygon" | "entry" | "target" | null;
type SceneViewMode = "map" | "height" | "pointcloud";

export interface ActiveSceneContext {
  sceneId: string;
  revision: number;
  bounds: Bounds3D;
  annotations: SemanticRegion[];
  currentPose?: PoseMap;
  selectedPoint?: Point3D;
}

const PICK_LABELS: Record<Exclude<PickMode, null>, string> = {
  ground: "地面样本",
  origin: "场景原点",
  "x-axis": "X 轴参考点",
  polygon: "区域边界",
  entry: "安全入口",
  target: "Agent 目标点",
};

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function shortId(value: string): string {
  return value.slice(0, 8);
}

function poseAgeLabel(observedAt: string | null | undefined): { label: string; stale: boolean } {
  if (!observedAt) return { label: "时间未知", stale: true };
  const observedAtMs = Date.parse(observedAt);
  if (!Number.isFinite(observedAtMs)) return { label: "时间未知", stale: true };
  const ageSeconds = Math.max(0, Math.floor((Date.now() - observedAtMs) / 1000));
  if (ageSeconds < 5) return { label: "刚刚更新", stale: false };
  if (ageSeconds < 60) return { label: `${ageSeconds} 秒前`, stale: ageSeconds > 30 };
  const ageMinutes = Math.floor(ageSeconds / 60);
  if (ageMinutes < 60) return { label: `${ageMinutes} 分钟前`, stale: true };
  return { label: `${Math.floor(ageMinutes / 60)} 小时前`, stale: true };
}

function revisionKey(revision: SceneRevision): string {
  return `${revision.sceneId}:${revision.revision}`;
}

function SceneStatus({ value }: { value: string }) {
  const tone = ["succeeded", "published"].includes(value)
    ? "success"
    : ["failed", "cancelled"].includes(value) ? "danger" : "working";
  const labels: Record<string, string> = {
    queued: "排队转换",
    running: "转换中",
    succeeded: "转换完成",
    failed: "转换失败",
    cancelled: "已取消",
    superseded: "已替代",
    draft: "可编辑",
    published: "已发布",
  };
  return <span className={`scene-status ${tone}`}>{labels[value] ?? value}</span>;
}

export function SceneWorkspace({
  apiFetch,
  selectedRobotId,
  currentPose,
  onContextChange,
  onClose,
  initialFile,
  onInitialFileConsumed,
}: {
  apiFetch: ApiFetcher;
  selectedRobotId: string;
  currentPose?: PoseMap;
  onContextChange: (context: ActiveSceneContext | null) => void;
  onClose: () => void;
  initialFile?: File | null;
  onInitialFileConsumed?: () => void;
}) {
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [selectedRevisionNumber, setSelectedRevisionNumber] = useState<number | null>(null);
  const [assetBaseUrl, setAssetBaseUrl] = useState("");
  const [binding, setBinding] = useState<{ sceneId: string; revision: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [dragging, setDragging] = useState(false);
  const [uploadName, setUploadName] = useState("");
  const [uploadProgress, setUploadProgress] = useState<{ done: number; total: number } | null>(null);
  const [conversionJobId, setConversionJobId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<SceneViewMode>("pointcloud");
  const [pickMode, setPickMode] = useState<PickMode>(null);
  const [groundPoints, setGroundPoints] = useState<Point3D[]>([]);
  const [originPoint, setOriginPoint] = useState<Point3D | null>(null);
  const [xAxisPoint, setXAxisPoint] = useState<Point3D | null>(null);
  const [knownDistance, setKnownDistance] = useState("1");
  const [mapOrigin, setMapOrigin] = useState({ x: "0", y: "0", z: "0" });
  const [polygonPoints, setPolygonPoints] = useState<Point3D[]>([]);
  const [entryPoint, setEntryPoint] = useState<Point3D | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<Point3D | null>(null);
  const [locationId, setLocationId] = useState("");
  const [locationName, setLocationName] = useState("");
  const [aliases, setAliases] = useState("");
  const [zMin, setZMin] = useState("0");
  const [zMax, setZMax] = useState("2.5");
  const [entryYaw, setEntryYaw] = useState("0");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const aliveRef = useRef(true);
  const consumedInitialFileRef = useRef<File | null>(null);

  const selectedScene = scenes.find((scene) => scene.sceneId === selectedSceneId) ?? null;
  const selectedRevision = selectedScene?.revisions.find(
    (revision) => revision.revision === selectedRevisionNumber,
  ) ?? null;
  const formalMapReady = Boolean(
    selectedRevision?.mapJob?.status === "succeeded"
    && selectedRevision.mapSummary?.verified,
  );
  const previewMapReady = Boolean(selectedRevision?.previewAvailable);
  const mapAvailable = formalMapReady || previewMapReady;
  const mapVariant: "preview" | "map" = formalMapReady ? "map" : "preview";

  const replaceRevision = useCallback((replacement: SceneRevision) => {
    setScenes((current) => current.map((scene) => scene.sceneId !== replacement.sceneId ? scene : {
      ...scene,
      revisions: scene.revisions.some((revision) => revision.revision === replacement.revision)
        ? scene.revisions.map((revision) => revision.revision === replacement.revision ? replacement : revision)
        : [...scene.revisions, replacement].sort((left, right) => left.revision - right.revision),
    }));
    setSelectedSceneId(replacement.sceneId);
    setSelectedRevisionNumber(replacement.revision);
  }, []);

  const loadScenes = useCallback(async () => {
    const loaded = await apiFetch<SceneSummary[]>("/api/v1/scenes");
    if (!aliveRef.current) return;
    setScenes(loaded);
    setSelectedSceneId((current) => current && loaded.some((scene) => scene.sceneId === current)
      ? current
      : loaded[0]?.sceneId ?? null);
  }, [apiFetch]);

  useEffect(() => {
    aliveRef.current = true;
    setLoading(true);
    void Promise.all([
      loadScenes(),
      apiFetch<{ sceneId: string; revision: number }>(
        `/api/v1/robots/${encodeURIComponent(selectedRobotId)}/scene-binding`,
      ).then((value) => setBinding(value)).catch(() => setBinding(null)),
    ]).catch((reason) => setError(reason instanceof Error ? reason.message : "场景库加载失败"))
      .finally(() => setLoading(false));
    return () => {
      aliveRef.current = false;
    };
  }, [apiFetch, loadScenes, selectedRobotId]);

  useEffect(() => {
    if (!selectedScene) {
      setSelectedRevisionNumber(null);
      return;
    }
    if (!selectedScene.revisions.some((revision) => revision.revision === selectedRevisionNumber)) {
      setSelectedRevisionNumber(selectedScene.revisions.at(-1)?.revision ?? null);
    }
  }, [selectedRevisionNumber, selectedScene]);

  useEffect(() => {
    setViewMode(previewMapReady || formalMapReady ? "map" : "pointcloud");
  }, [selectedSceneId, selectedRevisionNumber]);

  useEffect(() => {
    const job = selectedRevision?.mapJob;
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(() => void loadScenes().catch(() => undefined), 1200);
    return () => window.clearInterval(timer);
  }, [loadScenes, selectedRevision?.mapJob?.jobId, selectedRevision?.mapJob?.status]);

  useEffect(() => {
    // Every picked point is expressed in the currently displayed scene frame.
    // Never carry it into a different robot, cloud, or immutable revision.
    setPickMode(null);
    setGroundPoints([]);
    setOriginPoint(null);
    setXAxisPoint(null);
    setPolygonPoints([]);
    setEntryPoint(null);
    setSelectedTarget(null);
  }, [selectedRobotId, selectedRevisionNumber, selectedSceneId]);

  useEffect(() => {
    setAssetBaseUrl("");
    if (!selectedScene || !selectedRevision || !selectedRevision.artifactsAvailable) return;
    let disposed = false;
    const refreshSession = async () => {
      try {
        const session = await apiFetch<{ assetBaseUrl: string }>(
          `/api/v1/scenes/${encodeURIComponent(selectedScene.sceneId)}/revisions/${selectedRevision.revision}/asset-session`,
          { method: "POST" },
        );
        if (!disposed) setAssetBaseUrl(session.assetBaseUrl);
      } catch (reason) {
        if (!disposed) setError(reason instanceof Error ? reason.message : "无法创建点云读取会话");
      }
    };
    void refreshSession();
    const timer = window.setInterval(() => void refreshSession(), 10 * 60 * 1000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [apiFetch, selectedRevision, selectedScene]);

  useEffect(() => {
    const bounds = selectedRevision?.boundsMap;
    if (!bounds) return;
    setZMin(String(bounds.min.z));
    setZMax(String(Math.min(bounds.max.z, bounds.min.z + 2.5)));
  }, [selectedRevision?.boundsMap]);

  const selectedPointMap = useMemo(() => {
    if (!selectedTarget || !selectedRevision?.calibration) return undefined;
    try {
      return transformPoint(selectedRevision.calibration.TMapFromScene, selectedTarget);
    } catch {
      return undefined;
    }
  }, [selectedRevision?.calibration, selectedTarget]);

  const mapNorthYaw = useMemo(() => {
    const calibration = selectedRevision?.calibration;
    if (!calibration) return 0;
    try {
      const origin = transformPoint(calibration.TSceneFromMap, { x: 0, y: 0, z: 0 });
      const north = transformPoint(calibration.TSceneFromMap, { x: 0, y: 1, z: 0 });
      return yawForNorthVector({
        x: north.x - origin.x,
        y: north.y - origin.y,
        z: north.z - origin.z,
      });
    } catch {
      return 0;
    }
  }, [selectedRevision?.calibration]);

  const robotVisualization = useMemo<{
    marker?: RobotSceneMarker;
    unavailableReason?: string;
  }>(() => {
    const calibration = selectedRevision?.calibration;
    if (!calibration) return { unavailableReason: "完成坐标标定后可定位机器人" };
    if (!currentPose) return { unavailableReason: "当前机器人没有 map 位姿" };
    try {
      const position = transformPoint(calibration.TSceneFromMap, currentPose);
      const heading = transformPoint(calibration.TSceneFromMap, {
        x: currentPose.x + Math.cos(currentPose.yaw),
        y: currentPose.y + Math.sin(currentPose.yaw),
        z: currentPose.z,
      });
      const age = poseAgeLabel(currentPose.observedAt);
      return {
        marker: {
          position,
          heading,
          stale: age.stale,
          ageLabel: age.label,
        },
      };
    } catch {
      return { unavailableReason: "机器人位姿无法转换到当前场景" };
    }
  }, [currentPose, selectedRevision?.calibration]);

  useEffect(() => {
    if (
      !selectedScene || !selectedRevision?.boundsMap || selectedRevision.status !== "published" ||
      binding?.sceneId !== selectedScene.sceneId || binding.revision !== selectedRevision.revision
    ) {
      onContextChange(null);
      return;
    }
    onContextChange({
      sceneId: selectedScene.sceneId,
      revision: selectedRevision.revision,
      bounds: selectedRevision.boundsMap,
      annotations: selectedRevision.annotations,
      currentPose,
      selectedPoint: selectedPointMap,
    });
  }, [binding, currentPose, onContextChange, selectedPointMap, selectedRevision, selectedScene]);

  const overlays = useMemo<SceneOverlay[]>(() => {
    const output: SceneOverlay[] = [];
    if (robotVisualization.marker) {
      output.push({
        name: `${selectedRobotId} 位姿`,
        kind: "direction",
        points: [robotVisualization.marker.position, robotVisualization.marker.heading],
        color: robotVisualization.marker.stale ? 0xf59e0b : 0x22c55e,
      });
    }
    const calibration = selectedRevision?.calibration;
    if (calibration) {
      for (const annotation of selectedRevision.annotations) {
        output.push({
          name: annotation.name,
          kind: "polygon",
          points: annotation.polygonXY.map((point) => transformPoint(
            calibration.TSceneFromMap,
            { x: point.x, y: point.y, z: annotation.zMin },
          )),
        });
        output.push({
          name: `${annotation.name}入口`,
          kind: "point",
          points: [transformPoint(calibration.TSceneFromMap, annotation.entryPose)],
        });
      }
    }
    if (groundPoints.length) output.push({ name: "地面样本", kind: "point", points: groundPoints });
    if (originPoint) output.push({ name: "场景原点", kind: "point", points: [originPoint] });
    if (xAxisPoint) output.push({ name: "X轴参考", kind: "point", points: [xAxisPoint] });
    if (polygonPoints.length) output.push({ name: "待保存区域", kind: "polygon", points: polygonPoints });
    if (entryPoint) output.push({ name: "待保存入口", kind: "point", points: [entryPoint] });
    if (selectedTarget) output.push({ name: "Agent目标", kind: "point", points: [selectedTarget] });
    return output;
  }, [
    entryPoint,
    groundPoints,
    originPoint,
    polygonPoints,
    robotVisualization.marker,
    selectedRevision,
    selectedRobotId,
    selectedTarget,
    xAxisPoint,
  ]);

  const formalMapOverlays = useMemo<MapOverlay[]>(() => {
    const output: MapOverlay[] = [];
    if (currentPose) {
      const age = poseAgeLabel(currentPose.observedAt);
      output.push({
        name: `${selectedRobotId} 位姿`,
        kind: "direction",
        points: [
          currentPose,
          {
            x: currentPose.x + Math.cos(currentPose.yaw),
            y: currentPose.y + Math.sin(currentPose.yaw),
            z: currentPose.z,
          },
        ],
        color: age.stale ? 0xf59e0b : 0x22c55e,
      });
    }
    for (const annotation of selectedRevision?.annotations ?? []) {
      output.push({
        name: annotation.name,
        kind: "polygon",
        points: annotation.polygonXY.map((point) => ({ ...point, z: annotation.zMin })),
      });
      output.push({ name: `${annotation.name}入口`, kind: "point", points: [annotation.entryPose] });
    }
    const calibration = selectedRevision?.calibration;
    const addScenePoints = (name: string, kind: MapOverlay["kind"], points: Point3D[]) => {
      if (!calibration || !points.length) return;
      output.push({
        name,
        kind,
        points: points.map((point) => transformPoint(calibration.TMapFromScene, point)),
      });
    };
    addScenePoints("地面样本", "point", groundPoints);
    addScenePoints("场景原点", "point", originPoint ? [originPoint] : []);
    addScenePoints("X轴参考", "point", xAxisPoint ? [xAxisPoint] : []);
    addScenePoints("待保存区域", "polygon", polygonPoints);
    addScenePoints("待保存入口", "point", entryPoint ? [entryPoint] : []);
    addScenePoints("Agent目标", "point", selectedTarget ? [selectedTarget] : []);
    return output;
  }, [
    currentPose,
    entryPoint,
    groundPoints,
    originPoint,
    polygonPoints,
    selectedRevision,
    selectedRobotId,
    selectedTarget,
    xAxisPoint,
  ]);

  const onPick = useCallback((point: Point3D) => {
    if (pickMode === "ground") setGroundPoints((current) => [...current, point].slice(-12));
    if (pickMode === "origin") {
      setOriginPoint(point);
      setPickMode(null);
    }
    if (pickMode === "x-axis") {
      setXAxisPoint(point);
      setPickMode(null);
    }
    if (pickMode === "polygon") setPolygonPoints((current) => [...current, point]);
    if (pickMode === "entry") {
      setEntryPoint(point);
      setPickMode(null);
    }
    if (pickMode === "target") {
      setSelectedTarget(point);
      setPickMode(null);
    }
  }, [pickMode]);

  const onMapPick = useCallback((point: Point3D, frameId: "scene" | "map") => {
    if (frameId === "scene") {
      onPick(point);
      return;
    }
    const calibration = selectedRevision?.calibration;
    if (!calibration) {
      setError("正式地图缺少坐标标定，无法把点选结果转换回场景坐标");
      return;
    }
    try {
      onPick(transformPoint(calibration.TSceneFromMap, point));
    } catch {
      setError("地图点选坐标转换失败");
    }
  }, [onPick, selectedRevision?.calibration]);

  const mapOverlays = mapVariant === "map" ? formalMapOverlays : overlays;

  const pollConversion = useCallback(async (jobId: string) => {
    setConversionJobId(jobId);
    for (let attempt = 0; attempt < 3600 && aliveRef.current; attempt += 1) {
      const job = await apiFetch<SceneJob>(`/api/v1/scene-jobs/${encodeURIComponent(jobId)}`);
      if (["succeeded", "failed", "cancelled"].includes(job.status)) {
        await loadScenes();
        setConversionJobId(null);
        if (job.status === "succeeded") {
          if (job.revision) setSelectedRevisionNumber(job.revision);
          setNotice(`PCD 转换完成${job.revision ? `，已创建场景草稿 r${job.revision}` : ""}`);
        }
        else setError(job.message || `场景转换${job.status === "failed" ? "失败" : "已取消"}`);
        return;
      }
      await sleep(1000);
    }
  }, [apiFetch, loadScenes]);

  const importFile = useCallback(async (file: File) => {
    setBusy(true);
    setError("");
    setNotice("");
    setUploadName(file.name);
    try {
      const completed = await uploadPcd(file, apiFetch, (done, total) => {
        if (aliveRef.current) setUploadProgress({ done, total });
      });
      setUploadProgress(null);
      setSelectedSceneId(completed.sceneId);
      setNotice("上传完成，工作站正在后台生成浏览器点云");
      await loadScenes();
      await pollConversion(completed.jobId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "PCD 导入失败");
      setUploadProgress(null);
    } finally {
      setBusy(false);
    }
  }, [apiFetch, loadScenes, pollConversion]);

  useEffect(() => {
    if (!initialFile || consumedInitialFileRef.current === initialFile) return;
    consumedInitialFileRef.current = initialFile;
    onInitialFileConsumed?.();
    void importFile(initialFile);
  }, [importFile, initialFile, onInitialFileConsumed]);

  const chooseFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) void importFile(file);
  };

  const dropFile = (event: ReactDragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file && !busy) void importFile(file);
  };

  const mutate = useCallback(async (operation: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await operation();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "场景操作失败");
    } finally {
      setBusy(false);
    }
  }, []);

  const saveIdentity = () => {
    if (!selectedScene || !selectedRevision) return;
    void mutate(async () => {
      const revision = await apiFetch<SceneRevision>(
        `/api/v1/scenes/${encodeURIComponent(selectedScene.sceneId)}/revisions/${selectedRevision.revision}/calibration`,
        { method: "PUT", body: JSON.stringify({ calibration: IDENTITY_CALIBRATION }) },
      );
      replaceRevision(revision);
      setSelectedTarget(null);
      setNotice("已按 ROS map 同坐标系保存；原有区域标注已清空");
    });
  };

  const fitInteractive = () => {
    if (!selectedScene || !selectedRevision || groundPoints.length < 3 || !originPoint || !xAxisPoint) return;
    void mutate(async () => {
      const distance = Number(knownDistance);
      const origin = { x: Number(mapOrigin.x), y: Number(mapOrigin.y), z: Number(mapOrigin.z) };
      if (!Number.isFinite(distance) || distance <= 0 || !Object.values(origin).every(Number.isFinite)) {
        throw new Error("已知距离和 map 原点必须是有效数字");
      }
      const revision = await apiFetch<SceneRevision>(
        `/api/v1/scenes/${encodeURIComponent(selectedScene.sceneId)}/revisions/${selectedRevision.revision}/calibration/fit`,
        {
          method: "POST",
          body: JSON.stringify({
            groundPoints,
            originScene: originPoint,
            xAxisScene: xAxisPoint,
            knownDistanceM: distance,
            mapOrigin: origin,
          }),
        },
      );
      replaceRevision(revision);
      setPickMode(null);
      setSelectedTarget(null);
      setNotice(`交互标定已保存，平面残差 ${revision.calibration?.residualMeters.toFixed(3)} m`);
    });
  };

  const saveAnnotation = () => {
    if (!selectedScene || !selectedRevision?.calibration || polygonPoints.length < 3 || !entryPoint) return;
    const calibration = selectedRevision.calibration;
    const safeEntryPoint = entryPoint;
    void mutate(async () => {
      const minimum = Number(zMin);
      const maximum = Number(zMax);
      const yaw = Number(entryYaw);
      if (!locationId.trim() || !locationName.trim()) throw new Error("请填写区域 ID 和名称");
      if (![minimum, maximum, yaw].every(Number.isFinite) || minimum > maximum) {
        throw new Error("高度范围或入口朝向无效");
      }
      const mapPolygon = polygonPoints.map((point) => transformPoint(
        calibration.TMapFromScene,
        point,
      ));
      const mapEntry = transformPoint(calibration.TMapFromScene, safeEntryPoint);
      const annotation: SemanticRegion = {
        locationId: locationId.trim(),
        name: locationName.trim(),
        aliases: aliases.split(",").map((value) => value.trim()).filter(Boolean),
        polygonXY: mapPolygon.map(({ x, y }) => ({ x, y })),
        zMin: minimum,
        zMax: maximum,
        entryPose: { frameId: "map", ...mapEntry, yaw },
      };
      const revision = await apiFetch<SceneRevision>(
        `/api/v1/scenes/${encodeURIComponent(selectedScene.sceneId)}/revisions/${selectedRevision.revision}/annotations`,
        {
          method: "PUT",
          body: JSON.stringify({ annotations: [...selectedRevision.annotations, annotation] }),
        },
      );
      replaceRevision(revision);
      setPolygonPoints([]);
      setEntryPoint(null);
      setLocationId("");
      setLocationName("");
      setAliases("");
      setNotice(`区域“${annotation.name}”已保存到草稿版本`);
    });
  };

  const deleteAnnotation = (location: string) => {
    if (!selectedScene || !selectedRevision) return;
    void mutate(async () => {
      const revision = await apiFetch<SceneRevision>(
        `/api/v1/scenes/${encodeURIComponent(selectedScene.sceneId)}/revisions/${selectedRevision.revision}/annotations`,
        {
          method: "PUT",
          body: JSON.stringify({
            annotations: selectedRevision.annotations.filter((item) => item.locationId !== location),
          }),
        },
      );
      replaceRevision(revision);
    });
  };

  const publish = () => {
    if (!selectedScene || !selectedRevision) return;
    void mutate(async () => {
      const revision = await apiFetch<SceneRevision>(
        `/api/v1/scenes/${encodeURIComponent(selectedScene.sceneId)}/revisions/${selectedRevision.revision}/publish`,
        { method: "POST" },
      );
      replaceRevision(revision);
      setNotice(`场景版本 r${revision.revision} 已发布；绑定机器人后 Agent 才会使用`);
    });
  };

  const bindRobot = () => {
    if (!selectedScene || !selectedRevision || selectedRevision.status !== "published") return;
    void mutate(async () => {
      const result = await apiFetch<{ sceneId: string; revision: number }>(
        `/api/v1/robots/${encodeURIComponent(selectedRobotId)}/scene-binding`,
        {
          method: "PUT",
          body: JSON.stringify({ sceneId: selectedScene.sceneId, revision: selectedRevision.revision }),
        },
      );
      setBinding(result);
      setNotice(`已将 ${selectedRobotId} 绑定到场景 r${selectedRevision.revision}`);
    });
  };

  const cloneRevision = () => {
    if (!selectedScene || !selectedRevision) return;
    void mutate(async () => {
      const revision = await apiFetch<SceneRevision>(
        `/api/v1/scenes/${encodeURIComponent(selectedScene.sceneId)}/revisions`,
        { method: "POST", body: JSON.stringify({ baseRevision: selectedRevision.revision }) },
      );
      replaceRevision(revision);
      setNotice(`已从 r${selectedRevision.revision} 创建可编辑版本 r${revision.revision}`);
    });
  };

  const reprocessScene = () => {
    if (!selectedScene?.sourceAvailable) return;
    void mutate(async () => {
      const job = await apiFetch<SceneJob>(
        `/api/v1/scenes/${encodeURIComponent(selectedScene.sceneId)}/reprocess`,
        { method: "POST" },
      );
      setNotice("已保留原始 PCD，并开始生成新的场景产物版本");
      await loadScenes();
      await pollConversion(job.jobId);
    });
  };

  return (
    <section className="panel scene-workspace" aria-label="PCD 场景工作区">
      <div className="section-heading scene-heading">
        <div><p className="eyebrow">PCD · MAP CONTEXT</p><h2>场景地图与语义标注</h2></div>
        <div className="scene-heading-actions"><span className="muted">点云只在浏览器渲染，不发送给 LLM</span><button className="button button-ghost" type="button" onClick={onClose}>返回会话</button></div>
      </div>
      {error ? <div className="alert alert-error" role="alert"><span>{error}</span><button type="button" onClick={() => setError("")}>×</button></div> : null}
      {notice ? <div className="alert alert-success" role="status"><span>{notice}</span><button type="button" onClick={() => setNotice("")}>×</button></div> : null}
      <div className="scene-layout">
        <aside className="scene-library">
          <input ref={fileInputRef} type="file" accept=".pcd,application/octet-stream" hidden onChange={chooseFile} />
          <div
            className={`scene-dropzone ${dragging ? "dragging" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={dropFile}
          >
            <span className="scene-drop-icon">＋</span>
            <strong>导入 PCD</strong>
            <small>拖到这里，或从文件系统选择</small>
            <button className="button button-primary" type="button" disabled={busy} onClick={() => fileInputRef.current?.click()}>选择 PCD 文件</button>
          </div>
          {uploadProgress ? <div className="upload-progress">
            <div><span>{uploadName}</span><b>{Math.round(uploadProgress.done / uploadProgress.total * 100)}%</b></div>
            <progress value={uploadProgress.done} max={uploadProgress.total} />
            <small>{formatBytes(uploadProgress.done)} / {formatBytes(uploadProgress.total)}</small>
          </div> : null}
          {conversionJobId ? <div className="conversion-note"><span className="scene-spinner" />后台转换中 <code>{shortId(conversionJobId)}</code></div> : null}
          <div className="scene-library-title"><strong>场景库</strong><span>{scenes.length}</span></div>
          <div className="scene-list">
            {loading ? <p className="muted">正在读取场景…</p> : null}
            {!loading && scenes.length === 0 ? <p className="muted">尚未导入场景</p> : null}
            {scenes.map((scene) => <button
              key={scene.sceneId}
              type="button"
              className={`scene-card ${scene.sceneId === selectedSceneId ? "selected" : ""}`}
              onClick={() => setSelectedSceneId(scene.sceneId)}
            >
              <span><strong>{scene.sourceFileName ?? `场景 ${shortId(scene.sceneId)}`}</strong><SceneStatus value={scene.job?.status ?? "queued"} /></span>
              <small>{new Date(scene.createdAt).toLocaleString("zh-CN")} · {scene.revisions.length} 个版本</small>
            </button>)}
          </div>
        </aside>

        <div className="scene-editor">
          {!selectedScene ? <div className="scene-empty"><span>◫</span><h3>导入一个 PCD 开始</h3><p>转换完成后可在这里旋转、缩放、标定 map 坐标并标注厨房等语义区域。</p></div> : <>
            <div className="scene-revision-bar">
              <div><strong>{selectedScene.sourceFileName ?? `场景 ${shortId(selectedScene.sceneId)}`}</strong><code>{selectedScene.sceneId}</code></div>
              {selectedScene.revisions.length ? <label>版本
                <select value={selectedRevisionNumber ?? ""} onChange={(event) => setSelectedRevisionNumber(Number(event.target.value))}>
                  {selectedScene.revisions.map((revision) => <option key={revisionKey(revision)} value={revision.revision}>r{revision.revision} · {revision.status === "draft" ? "草稿" : "已发布"}</option>)}
                </select>
              </label> : <SceneStatus value={selectedScene.job?.status ?? "queued"} />}
            </div>
            {selectedScene.job?.status === "failed" ? <div className="scene-job-error"><strong>转换失败</strong><span>{selectedScene.job.message ?? selectedScene.job.errorCode}</span><button className="button" type="button" disabled={busy} onClick={() => void mutate(async () => {
              await apiFetch(`/api/v1/scene-jobs/${encodeURIComponent(selectedScene.job!.jobId)}/retry`, { method: "POST" });
              await pollConversion(selectedScene.job!.jobId);
            })}>重试转换</button></div> : null}
            {selectedScene.job && ["queued", "running"].includes(selectedScene.job.status) ? <div className="scene-converting"><span className="scene-spinner" /><div><strong>{selectedScene.job.status === "queued" ? "等待转换 Worker" : "正在生成 Potree LOD"}</strong><small>可以离开此页面，任务状态已持久化</small></div></div> : null}
            {selectedRevision?.processingSummary ? <div className="scene-processing-summary">
              <span>原始点 <b>{selectedRevision.processingSummary.sourcePointCount?.toLocaleString() ?? "—"}</b></span>
              <span>浏览点 <b>{selectedRevision.processingSummary.visualPointCount?.toLocaleString() ?? "—"}</b></span>
              <span>体素 <b>{selectedRevision.processingSummary.voxelSize?.toFixed(4) ?? "—"}</b></span>
              <button className="mini-button" type="button" disabled={busy || !selectedScene.sourceAvailable || Boolean(conversionJobId)} onClick={reprocessScene}>重新分析原始 PCD</button>
            </div> : null}
            {selectedRevision?.mapJob && ["queued", "running"].includes(selectedRevision.mapJob.status) ? <div className="scene-map-job"><span className="scene-spinner" /><div><strong>正在派生 r{selectedRevision.revision} 正式地图</strong><small>标定已保存；完成前继续显示未验证预览</small></div></div> : null}
            {selectedRevision?.mapJob?.status === "failed" ? <div className="scene-job-error"><strong>正式地图生成失败</strong><span>{selectedRevision.mapJob.message ?? selectedRevision.mapJob.errorCode}</span><button className="button" type="button" disabled={busy} onClick={() => void mutate(async () => {
              await apiFetch(`/api/v1/scene-jobs/${encodeURIComponent(selectedRevision.mapJob!.jobId)}/retry`, { method: "POST" });
              await loadScenes();
            })}>重试地图</button></div> : null}
            {selectedRevision && assetBaseUrl ? <>
              <div className="scene-view-toolbar">
                <div className="scene-view-tabs" role="tablist" aria-label="场景视图">
                  <button className={viewMode === "map" ? "active" : ""} type="button" disabled={!mapAvailable} onClick={() => setViewMode("map")}>二维地图</button>
                  <button className={viewMode === "height" ? "active" : ""} type="button" disabled={!mapAvailable} onClick={() => setViewMode("height")}>高度图</button>
                  <button className={viewMode === "pointcloud" ? "active" : ""} type="button" onClick={() => setViewMode("pointcloud")}>三维点云</button>
                </div>
                <div className="scene-pick-toolbar"><span>{pickMode ? `点击场景选择：${PICK_LABELS[pickMode]}` : viewMode === "pointcloud" ? "拖动旋转 · 滚轮/双指缩放" : "拖动平移 · 滚轮缩放"}</span><button className={`mini-button ${pickMode === "target" ? "active" : ""}`} type="button" onClick={() => setPickMode(pickMode === "target" ? null : "target")}>选 Agent 目标点</button>{pickMode ? <button className="mini-button" type="button" onClick={() => setPickMode(null)}>退出点选</button> : null}</div>
              </div>
              {viewMode !== "pointcloud" && mapAvailable ? <>
                {!formalMapReady ? <div className="scene-preview-banner">未验证预览 · 假设源坐标 XY 为水平面，仅用于理解场景和辅助标定</div> : <div className={`scene-formal-map-banner ${selectedRevision.mapSummary?.status === "degraded" ? "degraded" : ""}`}>正式 map 地图 · {selectedRevision.mapSummary?.confidence === "high" ? "高" : selectedRevision.mapSummary?.confidence === "medium" ? "中" : "低"}置信度 · 覆盖 {Math.round((selectedRevision.mapSummary?.observedCoverageRatio ?? 0) * 100)}%</div>}
                <SceneMapCanvas
                  assetBaseUrl={assetBaseUrl}
                  variant={mapVariant}
                  layer={viewMode === "height" ? "height" : "occupancy"}
                  pickMode={pickMode ? PICK_LABELS[pickMode] : null}
                  overlays={mapOverlays}
                  onPick={onMapPick}
                />
              </> : <PotreeCanvas
                  assetBaseUrl={assetBaseUrl}
                  sceneKey={selectedScene.sceneId}
                  pickMode={pickMode ? PICK_LABELS[pickMode] : null}
                  overlays={overlays}
                  mapNorthYaw={mapNorthYaw}
                  robotMarker={robotVisualization.marker}
                  robotFocusUnavailableReason={robotVisualization.unavailableReason}
                  onPick={onPick}
                />}
              {selectedPointMap ? <div className="selected-target"><span>Agent 目标点（map）</span><strong>{formatPoint(selectedPointMap)}</strong><button type="button" onClick={() => setSelectedTarget(null)}>清除</button></div> : null}
            </> : null}

            {selectedRevision ? <div className="scene-tools">
              <div className="scene-tool-heading"><div><p className="eyebrow">CALIBRATION</p><h3>坐标标定</h3></div><SceneStatus value={selectedRevision.status} /></div>
              {selectedRevision.status === "published" ? <div className="published-lock"><span>此版本不可变</span><button className="button" type="button" onClick={cloneRevision} disabled={busy}>从该版本创建草稿</button></div> : <div className="calibration-modes">
                <article className={`calibration-card ${selectedRevision.calibration?.mode === "ros_map_identity" ? "selected" : ""}`}>
                  <span className="mode-number">A</span><div><strong>ROS map 同坐标</strong><p>PCD 已经是 Z-up、米制、map 原点，无需缩放或旋转。</p></div>
                  <button className="button" type="button" onClick={saveIdentity} disabled={busy}>使用 identity 标定</button>
                </article>
                <article className={`calibration-card interactive ${selectedRevision.calibration?.mode === "interactive_similarity" ? "selected" : ""}`}>
                  <span className="mode-number">B</span><div><strong>交互相似变换</strong><p>在场景中选择地面、原点与 X 轴参考，拟合 Z-up 的尺度与方向。</p></div>
                  <div className="pick-actions">
                    <button className={`mini-button ${pickMode === "ground" ? "active" : ""}`} type="button" onClick={() => setPickMode(pickMode === "ground" ? null : "ground")}>地面点 {groundPoints.length}/3+</button>
                    <button className={`mini-button ${pickMode === "origin" ? "active" : ""}`} type="button" onClick={() => setPickMode("origin")}>原点 {originPoint ? "✓" : ""}</button>
                    <button className={`mini-button ${pickMode === "x-axis" ? "active" : ""}`} type="button" onClick={() => setPickMode("x-axis")}>X轴点 {xAxisPoint ? "✓" : ""}</button>
                    <button className="mini-button" type="button" onClick={() => setGroundPoints((points) => points.slice(0, -1))} disabled={!groundPoints.length}>撤销地面点</button>
                  </div>
                  <div className="calibration-fields">
                    <label>参考段真实长度（米）<input type="number" min="0.001" step="0.01" value={knownDistance} onChange={(event) => setKnownDistance(event.target.value)} /></label>
                    <label>map 原点 X<input type="number" step="0.01" value={mapOrigin.x} onChange={(event) => setMapOrigin((value) => ({ ...value, x: event.target.value }))} /></label>
                    <label>Y<input type="number" step="0.01" value={mapOrigin.y} onChange={(event) => setMapOrigin((value) => ({ ...value, y: event.target.value }))} /></label>
                    <label>Z<input type="number" step="0.01" value={mapOrigin.z} onChange={(event) => setMapOrigin((value) => ({ ...value, z: event.target.value }))} /></label>
                  </div>
                  <button className="button" type="button" onClick={fitInteractive} disabled={busy || groundPoints.length < 3 || !originPoint || !xAxisPoint}>计算并保存标定</button>
                </article>
              </div>}
              {selectedRevision.calibration ? <div className="calibration-result"><span>模式 <b>{selectedRevision.calibration.mode}</b></span><span>比例 <b>{selectedRevision.calibration.scaleMetersPerUnit.toFixed(4)} m/unit</b></span><span>残差 <b>{selectedRevision.calibration.residualMeters.toFixed(4)} m</b></span></div> : <p className="inline-warning">先完成一种标定，才能保存 map 坐标区域和发布版本。</p>}

              {selectedRevision.calibration ? <>
                <div className="scene-tool-heading annotation-heading"><div><p className="eyebrow">SEMANTIC MAP</p><h3>区域与安全入口</h3></div><span className="muted">{selectedRevision.annotations.length} 个位置</span></div>
                {selectedRevision.status === "draft" ? <div className="annotation-editor">
                  <div className="annotation-fields">
                    <label>区域 ID<input value={locationId} onChange={(event) => setLocationId(event.target.value)} placeholder="kitchen" /></label>
                    <label>显示名称<input value={locationName} onChange={(event) => setLocationName(event.target.value)} placeholder="厨房" /></label>
                    <label>别名（逗号分隔）<input value={aliases} onChange={(event) => setAliases(event.target.value)} placeholder="kitchen, 烹饪区" /></label>
                    <label>最低 Z<input type="number" step="0.1" value={zMin} onChange={(event) => setZMin(event.target.value)} /></label>
                    <label>最高 Z<input type="number" step="0.1" value={zMax} onChange={(event) => setZMax(event.target.value)} /></label>
                    <label>入口 yaw（弧度）<input type="number" step="0.1" value={entryYaw} onChange={(event) => setEntryYaw(event.target.value)} /></label>
                  </div>
                  <div className="pick-actions annotation-picks">
                    <button className={`mini-button ${pickMode === "polygon" ? "active" : ""}`} type="button" onClick={() => setPickMode(pickMode === "polygon" ? null : "polygon")}>点选边界 {polygonPoints.length}/3+</button>
                    <button className={`mini-button ${pickMode === "entry" ? "active" : ""}`} type="button" onClick={() => setPickMode("entry")}>点选入口 {entryPoint ? "✓" : ""}</button>
                    <button className="mini-button" type="button" onClick={() => setPolygonPoints((points) => points.slice(0, -1))} disabled={!polygonPoints.length}>撤销顶点</button>
                    <button className="mini-button" type="button" onClick={() => { setPolygonPoints([]); setEntryPoint(null); }}>清空</button>
                  </div>
                  <button className="button button-primary" type="button" disabled={busy || polygonPoints.length < 3 || !entryPoint || !locationId.trim() || !locationName.trim()} onClick={saveAnnotation}>保存语义区域</button>
                </div> : null}
                <div className="annotation-list">
                  {selectedRevision.annotations.map((annotation) => <div key={annotation.locationId}><span><strong>{annotation.name}</strong><small>{annotation.locationId} · 入口 ({annotation.entryPose.x.toFixed(2)}, {annotation.entryPose.y.toFixed(2)})</small></span>{selectedRevision.status === "draft" ? <button className="mini-button" type="button" onClick={() => deleteAnnotation(annotation.locationId)} disabled={busy}>删除</button> : null}</div>)}
                  {!selectedRevision.annotations.length ? <p className="muted">还没有语义区域；也可仅标定后发布。</p> : null}
                </div>
              </> : null}

              <div className="scene-publish-row">
                {selectedRevision.status === "draft" ? <button className="button button-confirm" type="button" disabled={busy || !selectedRevision.calibration || (previewMapReady && !formalMapReady)} onClick={publish}>发布不可变版本</button> : <button className="button button-confirm" type="button" disabled={busy || (binding?.sceneId === selectedScene.sceneId && binding.revision === selectedRevision.revision)} onClick={bindRobot}>{binding?.sceneId === selectedScene.sceneId && binding.revision === selectedRevision.revision ? `已绑定 ${selectedRobotId}` : `绑定到 ${selectedRobotId}`}</button>}
                <p>H-IDE 只把已发布且绑定的场景上下文交给 Agent；路径规划仍由真机 Robot Agent / Nav2 完成。</p>
              </div>
              {selectedRevision.status === "draft" && selectedRevision.calibration && previewMapReady && !formalMapReady ? <p className="inline-warning">等待正式地图派生完成后再发布，避免 Agent 审核引用未验证预览。</p> : null}
            </div> : null}
          </>}
        </div>
      </div>
    </section>
  );
}

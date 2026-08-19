import {
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { Point3D, SceneMapMetadata } from "./contracts";

export interface MapOverlay {
  name: string;
  kind: "point" | "polygon" | "direction";
  points: Point3D[];
  color?: number;
}

interface ViewState {
  centerX: number;
  centerY: number;
  zoom: number;
}

function color(value: number | undefined, fallback: string): string {
  return value == null ? fallback : `#${value.toString(16).padStart(6, "0")}`;
}

function validMetadata(value: unknown): value is SceneMapMetadata {
  if (!value || typeof value !== "object") return false;
  const item = value as Partial<SceneMapMetadata>;
  return item.schemaVersion === "h-ide.scene-map/1"
    && (item.frameId === "scene" || item.frameId === "map")
    && typeof item.width === "number" && Number.isInteger(item.width) && item.width > 0
    && typeof item.height === "number" && Number.isInteger(item.height) && item.height > 0
    && typeof item.resolution === "number" && Number.isFinite(item.resolution) && item.resolution > 0
    && typeof item.groundZ === "number" && Number.isFinite(item.groundZ)
    && Boolean(item.origin && Number.isFinite(item.origin.x) && Number.isFinite(item.origin.y));
}

export function SceneMapCanvas({
  assetBaseUrl,
  variant,
  layer,
  pickMode,
  overlays,
  onPick,
}: {
  assetBaseUrl: string;
  variant: "preview" | "map";
  layer: "occupancy" | "height";
  pickMode: string | null;
  overlays: MapOverlay[];
  onPick: (point: Point3D, frameId: "scene" | "map") => void;
}) {
  const [metadata, setMetadata] = useState<SceneMapMetadata | null>(null);
  const [view, setView] = useState<ViewState>({ centerX: 0, centerY: 0, zoom: 1 });
  const [error, setError] = useState("");
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<{
    pointerId: number;
    clientX: number;
    clientY: number;
    centerX: number;
    centerY: number;
    moved: boolean;
  } | null>(null);

  const mapUrl = `${assetBaseUrl}/${variant}/map.json`;
  const imageUrl = `${assetBaseUrl}/${variant}/${layer}.png`;

  useEffect(() => {
    const controller = new AbortController();
    setMetadata(null);
    setError("");
    void fetch(mapUrl, {
      credentials: "include",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    }).then(async (response) => {
      if (!response.ok) throw new Error(`地图元数据读取失败（HTTP ${response.status}）`);
      const value: unknown = await response.json();
      if (!validMetadata(value)) throw new Error("地图元数据格式无效");
      setMetadata(value);
      setView({ centerX: value.width / 2, centerY: value.height / 2, zoom: 1 });
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "地图读取失败");
      }
    });
    return () => controller.abort();
  }, [mapUrl]);

  const viewport = useMemo(() => {
    if (!metadata) return { x: 0, y: 0, width: 1, height: 1 };
    const width = metadata.width / view.zoom;
    const height = metadata.height / view.zoom;
    return {
      x: view.centerX - width / 2,
      y: view.centerY - height / 2,
      width,
      height,
    };
  }, [metadata, view]);

  const toPixel = (point: Point3D) => {
    if (!metadata) return { x: 0, y: 0 };
    return {
      x: (point.x - metadata.origin.x) / metadata.resolution,
      y: metadata.height - (point.y - metadata.origin.y) / metadata.resolution,
    };
  };

  const svgPoint = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return null;
    const matrix = svg.getScreenCTM();
    if (!matrix) return null;
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    return point.matrixTransform(matrix.inverse());
  };

  const onWheel = (event: ReactWheelEvent<SVGSVGElement>) => {
    if (!metadata) return;
    event.preventDefault();
    const anchor = svgPoint(event.clientX, event.clientY);
    if (!anchor) return;
    const nextZoom = Math.max(1, Math.min(24, view.zoom * (event.deltaY < 0 ? 1.25 : 0.8)));
    if (nextZoom === view.zoom) return;
    const fractionX = (anchor.x - viewport.x) / viewport.width;
    const fractionY = (anchor.y - viewport.y) / viewport.height;
    const nextWidth = metadata.width / nextZoom;
    const nextHeight = metadata.height / nextZoom;
    setView({
      zoom: nextZoom,
      centerX: anchor.x + (0.5 - fractionX) * nextWidth,
      centerY: anchor.y + (0.5 - fractionY) * nextHeight,
    });
  };

  const onPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      centerX: view.centerX,
      centerY: view.centerY,
      moved: false,
    };
  };

  const onPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current;
    const svg = svgRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !svg) return;
    const rectangle = svg.getBoundingClientRect();
    const deltaX = event.clientX - drag.clientX;
    const deltaY = event.clientY - drag.clientY;
    if (Math.hypot(deltaX, deltaY) > 5) drag.moved = true;
    setView((current) => ({
      ...current,
      centerX: drag.centerX - deltaX * viewport.width / Math.max(rectangle.width, 1),
      centerY: drag.centerY - deltaY * viewport.height / Math.max(rectangle.height, 1),
    }));
  };

  const onPointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag || drag.pointerId !== event.pointerId || drag.moved || !pickMode || !metadata) return;
    const point = svgPoint(event.clientX, event.clientY);
    if (!point || point.x < 0 || point.y < 0 || point.x > metadata.width || point.y > metadata.height) return;
    onPick({
      x: metadata.origin.x + point.x * metadata.resolution,
      y: metadata.origin.y + (metadata.height - point.y) * metadata.resolution,
      z: metadata.groundZ,
    }, metadata.frameId);
  };

  const zoomBy = (factor: number) => setView((current) => ({
    ...current,
    zoom: Math.max(1, Math.min(24, current.zoom * factor)),
  }));
  const reset = () => metadata && setView({
    centerX: metadata.width / 2,
    centerY: metadata.height / 2,
    zoom: 1,
  });

  return <div className={`scene-map-shell ${pickMode ? "picking" : ""}`}>
    <div className="scene-map-controls">
      <button type="button" onClick={() => zoomBy(1.25)} disabled={!metadata}>＋</button>
      <button type="button" onClick={() => zoomBy(0.8)} disabled={!metadata || view.zoom <= 1}>−</button>
      <button type="button" onClick={reset} disabled={!metadata}>适配</button>
    </div>
    {metadata ? <>
      <svg
        ref={svgRef}
        className="scene-map-svg"
        viewBox={`${viewport.x} ${viewport.y} ${viewport.width} ${viewport.height}`}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => { dragRef.current = null; }}
        role="img"
        aria-label={`${metadata.verified ? "正式" : "预览"}${layer === "height" ? "高度" : "占用"}地图`}
      >
        <image href={imageUrl} x="0" y="0" width={metadata.width} height={metadata.height} />
        {overlays.map((overlay, index) => {
          const points = overlay.points.map(toPixel);
          const stroke = color(overlay.color, overlay.kind === "direction" ? "#22c55e" : "#68d9e8");
          if (overlay.kind === "polygon" && points.length > 1) {
            return <polygon key={`${overlay.name}-${index}`} points={points.map((point) => `${point.x},${point.y}`).join(" ")} fill={`${stroke}2b`} stroke={stroke} strokeWidth={Math.max(1, 2 / view.zoom)} vectorEffect="non-scaling-stroke"><title>{overlay.name}</title></polygon>;
          }
          if (overlay.kind === "direction" && points.length > 1) {
            return <g key={`${overlay.name}-${index}`}><line x1={points[0].x} y1={points[0].y} x2={points[1].x} y2={points[1].y} stroke={stroke} strokeWidth="3" vectorEffect="non-scaling-stroke" /><circle cx={points[0].x} cy={points[0].y} r={Math.max(2, 6 / view.zoom)} fill={stroke} vectorEffect="non-scaling-stroke"><title>{overlay.name}</title></circle></g>;
          }
          return points.map((point, pointIndex) => <circle key={`${overlay.name}-${index}-${pointIndex}`} cx={point.x} cy={point.y} r={Math.max(2, 6 / view.zoom)} fill={stroke} stroke="#07121e" strokeWidth="2" vectorEffect="non-scaling-stroke"><title>{overlay.name}</title></circle>);
        })}
      </svg>
      <div className="scene-map-legend">
        <span><i className="legend-free" />{layer === "height" ? "低处" : "观测地面"}</span>
        <span><i className="legend-occupied" />{layer === "height" ? "高处" : "障碍证据"}</span>
        <span><i className="legend-unknown" />未知</span>
        <b>{metadata.resolution.toFixed(3)} m/cell · {metadata.frameId}</b>
      </div>
      <div className="scene-map-north" aria-label="地图正北">N<span>↑</span></div>
    </> : <div className="scene-map-loading"><span className="scene-spinner" />正在读取地图…</div>}
    {error ? <div className="potree-error">{error}</div> : null}
  </div>;
}

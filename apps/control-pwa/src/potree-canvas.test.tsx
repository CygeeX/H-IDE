import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { PotreeCanvas } from "./potree-canvas";

interface FakePoint {
  x: number;
  y: number;
  z: number;
  set(x: number, y: number, z: number): void;
}

function point(x: number, y: number, z: number): FakePoint {
  return { x, y, z, set(nextX, nextY, nextZ) { this.x = nextX; this.y = nextY; this.z = nextZ; } };
}

describe("Potree view controls", () => {
  let viewer: any;

  beforeEach(() => {
    localStorage.clear();
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query.includes("prefers-reduced-motion"),
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }));

    class FakeViewer {
      renderer: any;
      orbitControls = { navigationMode: "orbit", stop: vi.fn() };
      scene: any;
      setCameraMode = vi.fn();
      setControls = vi.fn();
      setMoveSpeed = vi.fn();
      setEDLEnabled = vi.fn();
      setFOV = vi.fn();
      setPointBudget = vi.fn();
      setBackground = vi.fn();
      fitToScreen = vi.fn();
      setTopView = vi.fn();
      setFrontView = vi.fn();
      setRightView = vi.fn();

      constructor(element: HTMLElement) {
        const canvas = document.createElement("canvas");
        canvas.getBoundingClientRect = () => ({
          x: 0, y: 0, width: 800, height: 450,
          top: 0, right: 800, bottom: 450, left: 0,
          toJSON: () => ({}),
        });
        element.appendChild(canvas);
        this.renderer = {
          domElement: canvas,
          setAnimationLoop: vi.fn(),
          dispose: vi.fn(),
        };
        const position = point(0, 0, 1);
        const target = point(0, 0, 0);
        this.scene = {
          pointclouds: [] as any[],
          measurements: [] as any[],
          view: {
            position,
            target,
            yaw: 0,
            pitch: -Math.PI / 4,
            radius: 1,
            getPivot: () => point(target.x, target.y, target.z),
            lookAt: (x: number, y: number, z: number) => {
              target.set(x, y, z);
              this.scene.view.radius = Math.hypot(position.x - x, position.y - y, position.z - z);
            },
          },
          addPointCloud: (cloud: any) => this.scene.pointclouds.push(cloud),
          addMeasurement: (measurement: any) => this.scene.measurements.push(measurement),
          removeMeasurement: (measurement: any) => {
            this.scene.measurements = this.scene.measurements.filter((item: any) => item !== measurement);
          },
          getActiveCamera: () => ({}),
        };
        viewer = this;
      }
    }

    class FakeMeasure {
      name = "";
      closed = false;
      showArea = false;
      showDistances = false;
      showCoordinates = false;
      showEdges = false;
      maxMarkers = 0;
      color = { setHex: vi.fn() };
      addMarker = vi.fn();
    }

    window.Potree = {
      Viewer: FakeViewer,
      Measure: FakeMeasure,
      PointSizeType: { ADAPTIVE: 1 },
      PointShape: { CIRCLE: 1 },
      CameraMode: { ORTHOGRAPHIC: 0, PERSPECTIVE: 1 },
      Utils: { getMousePointCloudIntersection: () => null },
      loadPointCloud: async () => ({
        pointcloud: {
          name: "",
          material: {},
          getAttribute: (name: string) => name === "position"
            ? { range: [[-8, -27, -3], [30, 15, 3]] }
            : name === "intensity" ? { range: [0, 156] } : null,
        },
      }),
    } as any;
  });

  afterEach(() => {
    cleanup();
    delete window.Potree;
    vi.unstubAllGlobals();
  });

  it("starts near the tight bounds and restores each mode camera", async () => {
    render(<PotreeCanvas
      assetBaseUrl="/scene-assets/scene-1/1"
      sceneKey="scene-1"
      pickMode={null}
      overlays={[]}
      onPick={() => undefined}
    />);

    await screen.findByText("场景已加载 · 3D 空间");
    expect(screen.getByRole("button", { name: "3D 空间" })).toHaveAttribute("aria-pressed", "true");
    expect(viewer.scene.view.target.z).toBe(0);

    fireEvent.click(screen.getByRole("button", { name: "2.5D 地图" }));
    expect(viewer.setCameraMode).toHaveBeenLastCalledWith(0);
    expect(viewer.orbitControls.navigationMode).toBe("map");
    expect(localStorage.getItem("h-ide:potree-view-mode:v1:scene-1")).toBe("map");

    viewer.scene.view.position.x += 4;
    viewer.scene.view.target.x += 4;
    const rememberedMapX = viewer.scene.view.position.x;
    fireEvent.click(screen.getByRole("button", { name: "3D 空间" }));
    expect(viewer.setCameraMode).toHaveBeenLastCalledWith(1);
    expect(viewer.orbitControls.navigationMode).toBe("orbit");
    fireEvent.click(screen.getByRole("button", { name: "2.5D 地图" }));
    expect(viewer.scene.view.position.x).toBeCloseTo(rememberedMapX);
  });
});

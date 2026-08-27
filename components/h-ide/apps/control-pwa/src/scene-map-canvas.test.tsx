import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SceneMapCanvas } from "./scene-map-canvas";

afterEach(() => vi.restoreAllMocks());

describe("SceneMapCanvas", () => {
  it("loads the selected versioned map metadata and raster", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
      schemaVersion: "h-ide.scene-map/1",
      mapVersion: "scene-r1-hash",
      frameId: "map",
      verified: true,
      status: "ready",
      resolution: 0.05,
      origin: { x: -2, y: -1 },
      width: 100,
      height: 80,
      imageRowOrder: "negative_y",
      groundZ: 0,
      summary: {
        schemaVersion: "h-ide.scene-map/1",
        mapVersion: "scene-r1-hash",
        frameId: "map",
        verified: true,
        status: "ready",
        resolution: 0.05,
        groundZ: 0,
        observedCoverageRatio: 0.8,
        confidence: "high",
        warnings: [],
      },
    }), { status: 200 }));

    const { container } = render(<SceneMapCanvas
      assetBaseUrl="/scene-assets/scene/1"
      variant="map"
      layer="height"
      pickMode={null}
      overlays={[]}
      onPick={vi.fn()}
    />);

    await screen.findByRole("img", { name: "正式高度地图" });
    expect(fetchMock).toHaveBeenCalledWith(
      "/scene-assets/scene/1/map/map.json",
      expect.objectContaining({ credentials: "include" }),
    );
    await waitFor(() => expect(container.querySelector("image")?.getAttribute("href"))
      .toBe("/scene-assets/scene/1/map/height.png"));
    expect(screen.getByText(/0.050 m\/cell/)).toBeInTheDocument();
  });
});

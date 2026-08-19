import { describe, expect, it } from "vitest";
import { isSceneApiPath } from "./vite-proxy-routing";

describe("Vite development API routing", () => {
  it.each([
    "/api/v1/scenes",
    "/api/v1/scenes/scene-1/revisions/1/publish",
    "/api/v1/scene-uploads",
    "/api/v1/scene-jobs/job-1?after=0",
    "/api/v1/robots/g1-dev/scene-binding",
  ])("routes %s to Scene API", (path) => {
    expect(isSceneApiPath(path)).toBe(true);
  });

  it.each([
    "/api/v1/robots",
    "/api/v1/robots/g1-dev/capabilities",
    "/api/v1/assistant/conversations",
    "/api/v1/scenery",
  ])("keeps %s on Control API", (path) => {
    expect(isSceneApiPath(path)).toBe(false);
  });
});

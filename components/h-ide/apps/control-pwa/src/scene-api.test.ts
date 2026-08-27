import { afterEach, describe, expect, it, vi } from "vitest";
import { formatBytes, transformPoint, uploadPcd } from "./scene-api";

afterEach(() => {
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe("scene API helpers", () => {
  it("transforms a scene point into ROS map coordinates", () => {
    expect(transformPoint([
      [0, -2, 0, 10],
      [2, 0, 0, -3],
      [0, 0, 2, 1],
      [0, 0, 0, 1],
    ], { x: 2, y: 4, z: 1 })).toEqual({ x: 2, y: 1, z: 3 });
    expect(formatBytes(1024 ** 2)).toBe("1.0 MiB");
  });

  it("rejects non-PCD files before creating an upload session", async () => {
    const apiFetch = async <T>(): Promise<T> => {
      throw new Error("API should not be called");
    };
    await expect(uploadPcd(
      new File(["not a point cloud"], "map.txt"),
      apiFetch,
      () => undefined,
    )).rejects.toThrow("请选择 .pcd 点云文件");
  });

  it("resumes the same server upload after a lost response and a new user attempt", async () => {
    vi.stubGlobal("crypto", {
      subtle: { digest: async () => new Uint8Array(32).buffer },
    });
    const file = new File(["abcdef"], "room.pcd", { lastModified: 123 });
    let serverOffset = 0;
    let creates = 0;
    let statusReads = 0;
    let firstPatch = true;
    const apiFetch = async <T>(path: string, init: RequestInit = {}): Promise<T> => {
      if (path === "/api/v1/scene-uploads" && init.method === "POST") {
        creates += 1;
        return {
          uploadId: "upload-1", status: "active", offset: 0,
          sizeBytes: 6, chunkSize: 3, expiresAt: "2099-01-01T00:00:00Z",
        } as T;
      }
      if (path === "/api/v1/scene-uploads/upload-1" && !init.method) {
        statusReads += 1;
        if (statusReads === 1) {
          const error = new Error("temporary outage") as Error & { status: number };
          error.status = 503;
          throw error;
        }
        return {
          uploadId: "upload-1", status: "active", offset: serverOffset,
          sizeBytes: 6, chunkSize: 3, expiresAt: "2099-01-01T00:00:00Z",
        } as T;
      }
      if (path === "/api/v1/scene-uploads/upload-1" && init.method === "PATCH") {
        serverOffset += 3;
        if (firstPatch) {
          firstPatch = false;
          throw new Error("response lost after server write");
        }
        return { offset: serverOffset } as T;
      }
      if (path === "/api/v1/scene-uploads/upload-1/complete" && init.method === "POST") {
        return {
          sceneId: "scene-1", assetId: "asset-1", jobId: "job-1",
          sha256: "digest", status: "queued", dataEncoding: "ascii",
        } as T;
      }
      throw new Error(`unexpected request ${init.method ?? "GET"} ${path}`);
    };

    await expect(uploadPcd(file, apiFetch, () => undefined)).rejects.toThrow("temporary outage");
    await expect(uploadPcd(file, apiFetch, () => undefined)).resolves.toMatchObject({
      sceneId: "scene-1",
      jobId: "job-1",
    });
    expect(creates).toBe(1);
    expect(serverOffset).toBe(6);
    expect(localStorage.length).toBe(0);
  });
});

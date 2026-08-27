import type {
  Calibration,
  Point3D,
  SceneUploadCompleted,
  SceneUploadSession,
} from "./contracts";

export type ApiFetcher = <T>(path: string, init?: RequestInit) => Promise<T>;

export const IDENTITY_CALIBRATION: Calibration = {
  mode: "ros_map_identity",
  TMapFromScene: [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
  TSceneFromMap: [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
  scaleMetersPerUnit: 1,
  residualMeters: 0,
};

const UPLOAD_CHECKPOINT_PREFIX = "h-ide:scene-upload:v1:";

interface UploadCheckpoint {
  uploadId: string;
  fileName: string;
  sizeBytes: number;
  lastModified: number;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

async function checksum(blob: Blob): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error("当前浏览器不支持安全的分片校验，请通过 HTTPS 或本机地址打开 H-IDE");
  }
  const bytes = typeof blob.arrayBuffer === "function"
    ? await blob.arrayBuffer()
    : await new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(reader.error ?? new Error("无法读取 PCD 分片"));
      reader.onload = () => reader.result instanceof ArrayBuffer
        ? resolve(reader.result)
        : reject(new Error("无法读取 PCD 分片"));
      reader.readAsArrayBuffer(blob);
    });
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return bytesToBase64(new Uint8Array(digest));
}

async function fileCheckpointKey(file: File): Promise<string> {
  const sampleSize = 64 * 1024;
  const fingerprint = await checksum(new Blob([
    file.slice(0, Math.min(sampleSize, file.size)),
    file.slice(Math.max(0, file.size - sampleSize)),
  ]));
  return `${UPLOAD_CHECKPOINT_PREFIX}${encodeURIComponent(file.name)}:${file.size}:${file.lastModified}:${fingerprint}`;
}

function browserStorage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

function readCheckpoint(storage: Storage | null, key: string, file: File): UploadCheckpoint | null {
  if (!storage) return null;
  try {
    const value = JSON.parse(storage.getItem(key) ?? "null") as Partial<UploadCheckpoint> | null;
    if (
      value?.uploadId && value.fileName === file.name && value.sizeBytes === file.size &&
      value.lastModified === file.lastModified
    ) {
      return value as UploadCheckpoint;
    }
    storage.removeItem(key);
  } catch {
    try { storage.removeItem(key); } catch { /* Storage can be disabled by browser policy. */ }
  }
  return null;
}

function writeCheckpoint(storage: Storage | null, key: string, value: UploadCheckpoint | null): void {
  if (!storage) return;
  try {
    if (value) storage.setItem(key, JSON.stringify(value));
    else storage.removeItem(key);
  } catch {
    // Upload still works when private browsing or policy disables localStorage.
  }
}

function validSession(session: SceneUploadSession, file: File): boolean {
  return session.sizeBytes === file.size && Number.isInteger(session.offset) &&
    session.offset >= 0 && session.offset <= file.size && session.chunkSize > 0;
}

async function uploadSession(
  file: File,
  apiFetch: ApiFetcher,
  storage: Storage | null,
  key: string,
): Promise<SceneUploadSession> {
  const checkpoint = readCheckpoint(storage, key, file);
  if (checkpoint) {
    try {
      const resumed = await apiFetch<SceneUploadSession>(
        `/api/v1/scene-uploads/${encodeURIComponent(checkpoint.uploadId)}`,
      );
      if (validSession(resumed, file) && ["active", "completed"].includes(resumed.status)) {
        return resumed;
      }
    } catch (error) {
      const status = typeof error === "object" && error !== null && "status" in error
        ? (error as { status?: unknown }).status
        : undefined;
      if (status !== 404) throw error;
      // A removed server-side session is the only case that starts a new upload.
    }
    writeCheckpoint(storage, key, null);
  }
  const created = await apiFetch<SceneUploadSession>("/api/v1/scene-uploads", {
    method: "POST",
    body: JSON.stringify({
      fileName: file.name,
      sizeBytes: file.size,
      lastModified: file.lastModified,
    }),
  });
  if (!validSession(created, file) || created.status !== "active") {
    throw new Error("工作站返回了无效的上传会话");
  }
  writeCheckpoint(storage, key, {
    uploadId: created.uploadId,
    fileName: file.name,
    sizeBytes: file.size,
    lastModified: file.lastModified,
  });
  return created;
}

export async function uploadPcd(
  file: File,
  apiFetch: ApiFetcher,
  onProgress: (uploadedBytes: number, totalBytes: number) => void,
): Promise<SceneUploadCompleted> {
  if (!file.name.toLowerCase().endsWith(".pcd")) {
    throw new Error("请选择 .pcd 点云文件");
  }
  if (file.size <= 0) throw new Error("PCD 文件为空");
  const checkpointKey = await fileCheckpointKey(file);
  const storage = browserStorage();
  const session = await uploadSession(file, apiFetch, storage, checkpointKey);
  if (session.status === "completed") {
    const completed = await apiFetch<SceneUploadCompleted>(
      `/api/v1/scene-uploads/${encodeURIComponent(session.uploadId)}/complete`,
      { method: "POST" },
    );
    writeCheckpoint(storage, checkpointKey, null);
    return completed;
  }
  let offset = session.offset;
  let recoveryRetries = 0;
  onProgress(offset, file.size);
  while (offset < file.size) {
    const chunk = file.slice(offset, Math.min(offset + session.chunkSize, file.size));
    let result: { offset: number };
    try {
      result = await apiFetch<{ offset: number }>(
        `/api/v1/scene-uploads/${encodeURIComponent(session.uploadId)}`,
        {
          method: "PATCH",
          headers: {
            "content-type": "application/octet-stream",
            "upload-offset": String(offset),
            "upload-checksum": `sha256 ${await checksum(chunk)}`,
          },
          body: chunk,
        },
      );
      recoveryRetries = 0;
    } catch (error) {
      if (recoveryRetries >= 1) throw error;
      recoveryRetries += 1;
      const recovered = await apiFetch<SceneUploadSession>(
        `/api/v1/scene-uploads/${encodeURIComponent(session.uploadId)}`,
      );
      if (!validSession(recovered, file) || recovered.status !== "active") throw error;
      offset = recovered.offset;
      onProgress(offset, file.size);
      continue;
    }
    if (!Number.isInteger(result.offset) || result.offset <= offset || result.offset > file.size) {
      throw new Error("工作站返回了无效的上传偏移量");
    }
    offset = result.offset;
    onProgress(offset, file.size);
  }
  const completed = await apiFetch<SceneUploadCompleted>(
    `/api/v1/scene-uploads/${encodeURIComponent(session.uploadId)}/complete`,
    { method: "POST" },
  );
  writeCheckpoint(storage, checkpointKey, null);
  return completed;
}

export function transformPoint(matrix: number[][], point: Point3D): Point3D {
  if (matrix.length !== 4 || matrix.some((row) => row.length !== 4)) {
    throw new Error("场景标定矩阵无效");
  }
  const input = [point.x, point.y, point.z, 1];
  const output = matrix.map((row) => row.reduce((sum, value, index) => sum + value * input[index], 0));
  if (!output.every(Number.isFinite) || Math.abs(output[3] - 1) > 1e-6) {
    throw new Error("场景坐标无法转换到 map 坐标");
  }
  return { x: output[0], y: output[1], z: output[2] };
}

export function formatPoint(point: Point3D | null | undefined): string {
  if (!point) return "未选择";
  return `${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

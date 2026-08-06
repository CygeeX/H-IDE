export type RobotStatus = "online" | "offline" | "degraded" | "unknown" | "estop";
export type SafetyMode = "normal" | "degraded" | "stopped" | "estop" | "unknown";
export type CommandStatus =
  | "pending"
  | "accepted"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "expired"
  | "rejected"
  | "unknown";

export interface RobotState {
  robotId: string;
  name?: string;
  status: RobotStatus;
  connected: boolean;
  safetyMode: SafetyMode;
  battery?: number;
  taskState?: string;
  details?: Record<string, unknown>;
}

export interface CapabilitySnapshot {
  robotId: string;
  agent: string;
  protocol: string;
  capabilities: string[];
  safetyMode: SafetyMode;
  schema?: Record<string, unknown>;
}

export interface CommandReceipt {
  commandId: string;
  robotId?: string;
  status: CommandStatus;
  acceptedAt?: string;
  startedAt?: string | null;
  completedAt?: string | null;
  result?: unknown;
  error?: string | null;
  safetyStop?: boolean;
  safetyWarning?: string | null;
}

export interface MappingSnapshot {
  mapping: Record<string, unknown>;
  mappingId: string;
  version: string;
  commandKey: string;
  status: "draft" | "validated" | "probed" | "active" | "superseded";
  targetId?: string | null;
  probe?: { status?: string; result?: unknown; error?: string | null } | null;
  updatedAt?: string;
}

export interface PlanStep {
  stepId: string;
  commandKey: string;
  payload: Record<string, unknown>;
  dependsOn?: string[];
}

export interface PlanDraft {
  planId?: string;
  draftId?: string;
  robotId: string;
  steps: PlanStep[];
  expiresAt?: string;
}

export interface ControlEvent {
  eventId?: number;
  commandId?: string;
  robotId?: string;
  status?: string;
  mappingVersion?: string;
  traceId?: string;
  at?: string;
  createdAt?: string;
  error?: string | null;
  [key: string]: unknown;
}

export interface ApiError {
  detail?: string;
}

export function statusLabel(status: string | undefined): string {
  const labels: Record<string, string> = {
    online: "在线",
    offline: "离线",
    degraded: "降级",
    unknown: "未知",
    estop: "急停",
    normal: "正常",
    stopped: "已停止",
    pending: "排队",
    accepted: "已接收",
    running: "执行中",
    succeeded: "成功",
    failed: "失败",
    cancelled: "已取消",
    expired: "已过期",
    rejected: "已拒绝",
    draft: "草稿",
    validated: "已校验",
    probed: "已探测",
    active: "启用",
    superseded: "已替代",
  };
  return labels[status ?? ""] ?? status ?? "未知";
}

export function isTerminal(status: string | undefined): boolean {
  return ["succeeded", "failed", "cancelled", "expired", "rejected"].includes(status ?? "");
}

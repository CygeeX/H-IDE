import type { MissionRecordV2 } from "./contracts";

export interface MissionUserProgress {
  sequence: number;
  text: string;
  state: "active" | "done" | "blocked" | "info";
}
function payloadString(payload: Record<string, unknown>, key: string): string | null {
  const value = payload[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function missionUserProgress(mission: MissionRecordV2): MissionUserProgress[] {
  const nodes = new Map(
    mission.activeRevision.nodes.map((node) => [node.nodeId, node]),
  );
  const items: MissionUserProgress[] = [];
  for (const event of [...mission.events].sort((left, right) => left.sequence - right.sequence)) {
    const node = event.nodeId ? nodes.get(event.nodeId) : null;
    const purpose = node?.purpose ?? "当前步骤";
    const skillId = payloadString(event.payload, "skillId") ?? node?.skillId ?? "";
    let text: string | null = null;
    let state: MissionUserProgress["state"] = "info";
    if (event.eventType === "mission.started") {
      text = "行动任务已开始。";
      state = "active";
    } else if (event.eventType === "skill.requested") {
      state = "active";
      text = skillId === "navigation.navigate_to_pose"
        ? `已向机器人下发导航：${purpose}`
        : skillId === "navigation.rotate"
          ? `机器人正在转向：${purpose}`
          : skillId === "vision.observe"
            ? `正在获取并分析新图像：${purpose}`
            : skillId === "scene.verify_inside_region"
              ? `正在验证机器人位置：${purpose}`
              : skillId === "inspection.scan_station"
                ? `正在多方向观察：${purpose}`
                : skillId === "mission.report"
                  ? `正在汇总任务结果：${purpose}`
                  : `正在执行：${purpose}`;
    } else if (event.eventType === "skill.progress") {
      const phase = payloadString(event.payload, "phase");
      const detail = payloadString(event.payload, "detail");
      text = phase === "captured_heading"
        ? `已采集一个观察方向：${detail ?? purpose}`
        : detail ?? `机器人正在执行：${purpose}`;
      state = "active";
    } else if (event.eventType === "skill.succeeded") {
      text = `已完成：${purpose}`;
      state = "done";
    } else if (event.eventType === "skill.failed") {
      text = `未完成：${purpose}`;
      state = "blocked";
    } else if (event.eventType === "skill.retry_scheduled") {
      text = `正在重试审核过的目标：${purpose}`;
      state = "active";
    } else if (event.eventType === "mission.nodes_skipped") {
      text = "已找到目标，跳过剩余搜索方向。";
      state = "done";
    } else if (event.eventType === "mission.needs_input") {
      text = payloadString(event.payload, "detail") ?? "任务需要补充目标信息。";
      state = "blocked";
    }
    if (!text) continue;
    const previous = items.at(-1);
    if (previous?.text === text && previous.state === state) {
      items[items.length - 1] = { sequence: event.sequence, text, state };
    } else {
      items.push({ sequence: event.sequence, text, state });
    }
  }
  return items.slice(-8);
}

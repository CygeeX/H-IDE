import { describe, expect, it } from "vitest";

import type { MissionRecordV2 } from "./contracts";
import { missionUserProgress } from "./mission-progress";

describe("Mission user progress", () => {
  it("maps durable events to ordered user-readable activity", () => {
    const mission = {
      activeRevision: {
        nodes: [{
          nodeId: "navigate-entry",
          skillId: "navigation.navigate_to_pose",
          purpose: "前往开放办公室入口",
        }],
      },
      events: [
        { sequence: 3, eventType: "skill.succeeded", nodeId: "navigate-entry", payload: {} },
        { sequence: 1, eventType: "mission.started", nodeId: null, payload: {} },
        { sequence: 2, eventType: "skill.requested", nodeId: "navigate-entry", payload: { skillId: "navigation.navigate_to_pose" } },
      ],
    } as unknown as MissionRecordV2;

    expect(missionUserProgress(mission).map((item) => item.text)).toEqual([
      "行动任务已开始。",
      "已向机器人下发导航：前往开放办公室入口",
      "已完成：前往开放办公室入口",
    ]);
  });
});

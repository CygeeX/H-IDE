import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { MissionRecordV2 } from "./contracts";
import {
  currentMissionOfficeReport,
  MissionOfficeReportCard,
  parseMissionOfficeReport,
} from "./mission-office-report";

afterEach(cleanup);

function rawReport(overrides: Record<string, unknown> = {}) {
  return {
    schemaVersion: "h-ide.office-report/1",
    missionId: "mission-office-1",
    status: "completed",
    fullChainAccepted: true,
    regions: [
      {
        regionId: "meeting_room",
        name: "会议室",
        kind: "room",
        status: "observed",
        summaries: ["看到会议室门牌、会议桌和显示屏"],
        stations: [],
      },
      {
        regionId: "corridor",
        name: "走廊",
        kind: "corridor",
        status: "observed",
        summaries: ["走廊可通行，沿途可见立柱和绿植"],
        stations: [],
      },
    ],
    missingStations: [],
    evidence: [
      {
        stationId: "meeting_main",
        analysisId: "analysis-office-1",
        sha256: "a".repeat(64),
        mediaType: "image/jpeg",
      },
    ],
    narrativeZhCN: "Office V2 巡检报告\n- 会议室：看到会议桌和显示屏\n- 走廊：可通行",
    ...overrides,
  };
}

function missionFixture(): MissionRecordV2 {
  const requestId = "skill-mission-office-report-1";
  const reportNode = {
    nodeId: "office-final-report",
    skillId: "mission.report",
    skillVersion: "1.0.0",
    purpose: "汇总巡检报告",
    intent: {},
    dependsOn: [],
    preconditionIds: [],
    postconditionIds: [],
    riskLevel: "low",
  };
  return {
    missionId: "mission-office-1",
    robotId: "g1-sim",
    status: "completed",
    plan: { nodes: [reportNode] },
    authorization: {},
    activeRevision: {
      revisionId: "revision-office-1",
      sequence: 1,
      objective: "巡检全部办公室",
      nodes: [reportNode],
      disposition: "human_approved",
      changeSummary: [],
      createdAt: "2026-08-24T00:00:00Z",
    },
    revisions: [],
    snapshot: {
      missionId: "mission-office-1",
      robotId: "g1-sim",
      status: "completed",
      lastSequence: 90,
      activeRevisionId: "revision-office-1",
      activeNodeId: null,
      activeSkillRequestId: null,
      autoRevisionCount: 0,
      updatedAt: "2026-08-24T00:10:00Z",
    },
    events: [],
    worldRecords: [],
    nodeStates: [{
      missionId: "mission-office-1",
      revisionId: "revision-office-1",
      nodeId: "office-final-report",
      status: "succeeded",
      attempt: 1,
      state: { skillRequestId: requestId, resultStatus: "succeeded" },
      updatedAt: "2026-08-24T00:10:00Z",
    }],
    skillExecutions: [{
      requestId,
      missionId: "mission-office-1",
      revisionId: "revision-office-1",
      nodeId: "office-final-report",
      status: "succeeded",
      request: {
        requestId,
        missionId: "mission-office-1",
        revisionId: "revision-office-1",
        nodeId: "office-final-report",
        skillId: "mission.report",
        skillVersion: "1.0.0",
      },
      result: {
        requestId,
        status: "succeeded",
        result: { missionReport: rawReport() },
      },
    }],
  } as unknown as MissionRecordV2;
}

describe("Mission Office report", () => {
  it("binds the final report to the exact revision, node, request, and terminal", () => {
    const report = currentMissionOfficeReport(missionFixture());

    expect(report?.fullChainAccepted).toBe(true);
    expect(report?.regions.map((region) => region.name)).toEqual(["会议室", "走廊"]);
    expect(report?.evidenceCount).toBe(1);
  });

  it("fails closed for duplicate executions or forged report identity", () => {
    const duplicate = missionFixture();
    duplicate.skillExecutions.push(structuredClone(duplicate.skillExecutions[0]));
    expect(currentMissionOfficeReport(duplicate)).toBeNull();

    const forged = missionFixture();
    const execution = forged.skillExecutions[0] as Record<string, unknown>;
    const terminal = execution.result as Record<string, unknown>;
    const result = terminal.result as Record<string, unknown>;
    result.missionReport = rawReport({ missionId: "mission-other" });
    expect(currentMissionOfficeReport(forged)).toBeNull();
  });

  it("rejects a completed claim when missing stations remain", () => {
    expect(parseMissionOfficeReport(rawReport({
      missingStations: ["storage_main"],
    }), "mission-office-1")).toBeNull();

    const partial = parseMissionOfficeReport(rawReport({
      status: "completed_with_gaps",
      fullChainAccepted: false,
      missingStations: ["storage_main"],
    }), "mission-office-1");
    expect(partial?.fullChainAccepted).toBe(false);
  });

  it("renders the evidence-bound Chinese regional report", () => {
    const report = currentMissionOfficeReport(missionFixture());
    expect(report).not.toBeNull();
    render(<MissionOfficeReportCard report={report!} />);

    expect(screen.getByRole("region", { name: "办公室巡检报告" })).toBeTruthy();
    expect(screen.getByText("全部区域已完成")).toBeTruthy();
    expect(screen.getByText("1 份证据")).toBeTruthy();
    expect(screen.getByText("会议室")).toBeTruthy();
    expect(screen.getByText(/Office V2 巡检报告/)).toBeTruthy();
  });

  it("parses and labels a current-location report separately from a full tour", () => {
    const report = parseMissionOfficeReport(rawReport({
      reportScope: "current_location",
      locationId: "meeting_room",
      stationId: "meeting_south",
      regions: [rawReport().regions[0]],
      narrativeZhCN: "会议室观察报告\n- 看到会议桌和显示屏",
    }), "mission-office-1");

    expect(report?.reportScope).toBe("current_location");
    expect(report?.locationId).toBe("meeting_room");
    render(<MissionOfficeReportCard report={report!} />);
    expect(screen.getByText("目标区域观察完成")).toBeTruthy();
    expect(screen.queryByText("全部区域已完成")).toBeNull();
  });
});

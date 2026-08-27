import type { MissionRecordV2 } from "./contracts";

export interface MissionOfficeReportRegion {
  regionId: string;
  name: string;
  kind: "room" | "corridor";
  status: "observed" | "partial";
  summaries: string[];
}
export interface MissionOfficeReport {
  schemaVersion: "h-ide.office-report/1";
  missionId: string;
  reportScope: "all_office_regions_and_corridor" | "current_location";
  locationId?: string;
  stationId?: string;
  status: "completed" | "completed_with_gaps";
  fullChainAccepted: boolean;
  regions: MissionOfficeReportRegion[];
  missingStations: string[];
  evidenceCount: number;
  narrativeZhCN: string;
}

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function strings(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  const result: string[] = [];
  for (const item of value) {
    if (typeof item !== "string" || !item.trim()) return null;
    result.push(item.trim());
  }
  return result;
}

export function parseMissionOfficeReport(
  value: unknown,
  missionId: string,
): MissionOfficeReport | null {
  const report = object(value);
  if (!report || report.schemaVersion !== "h-ide.office-report/1") return null;
  if (report.missionId !== missionId) return null;
  if (report.status !== "completed" && report.status !== "completed_with_gaps") return null;
  if (typeof report.fullChainAccepted !== "boolean") return null;
  if (report.fullChainAccepted !== (report.status === "completed")) return null;
  const missingStations = strings(report.missingStations);
  if (!missingStations || new Set(missingStations).size !== missingStations.length) return null;
  if ((report.status === "completed") !== (missingStations.length === 0)) return null;
  if (typeof report.narrativeZhCN !== "string" || !report.narrativeZhCN.trim()) return null;
  const reportScope = report.reportScope === "current_location"
    ? "current_location"
    : report.reportScope == null || report.reportScope === "all_office_regions_and_corridor"
      ? "all_office_regions_and_corridor"
      : null;
  if (!reportScope) return null;
  if (!Array.isArray(report.regions) || report.regions.length === 0) return null;
  const regions: MissionOfficeReportRegion[] = [];
  const regionIds = new Set<string>();
  for (const raw of report.regions) {
    const region = object(raw);
    if (!region) return null;
    if (typeof region.regionId !== "string" || !region.regionId.trim()) return null;
    if (regionIds.has(region.regionId.trim())) return null;
    if (typeof region.name !== "string" || !region.name.trim()) return null;
    if (region.kind !== "room" && region.kind !== "corridor") return null;
    if (region.status !== "observed" && region.status !== "partial") return null;
    const summaries = strings(region.summaries);
    if (!summaries) return null;
    regionIds.add(region.regionId.trim());
    regions.push({
      regionId: region.regionId.trim(),
      name: region.name.trim(),
      kind: region.kind,
      status: region.status,
      summaries,
    });
  }
  if (!Array.isArray(report.evidence)) return null;
  const locationId = typeof report.locationId === "string" && report.locationId.trim()
    ? report.locationId.trim()
    : undefined;
  const stationId = typeof report.stationId === "string" && report.stationId.trim()
    ? report.stationId.trim()
    : undefined;
  if (reportScope === "current_location" && (
    !locationId
    || !stationId
    || regions.length !== 1
    || regions[0].regionId !== locationId
    || missingStations.length !== 0
  )) return null;
  const evidenceIds = new Set<string>();
  for (const raw of report.evidence) {
    const evidence = object(raw);
    if (!evidence || typeof evidence.sha256 !== "string") return null;
    const sha256 = evidence.sha256.trim().toLowerCase();
    if (!/^(?:sha256:)?[0-9a-f]{64}$/.test(sha256)) return null;
    evidenceIds.add(sha256);
  }
  return {
    schemaVersion: "h-ide.office-report/1",
    missionId,
    reportScope,
    ...(locationId ? { locationId } : {}),
    ...(stationId ? { stationId } : {}),
    status: report.status,
    fullChainAccepted: report.fullChainAccepted,
    regions,
    missingStations,
    evidenceCount: evidenceIds.size,
    narrativeZhCN: report.narrativeZhCN.trim(),
  };
}

export function currentMissionOfficeReport(
  mission: MissionRecordV2,
): MissionOfficeReport | null {
  const revisionId = mission.snapshot.activeRevisionId;
  if (typeof revisionId !== "string" || !revisionId) return null;
  const nodes = mission.activeRevision.nodes.filter(
    (node) => node.skillId === "mission.report" && node.skillVersion === "1.0.0",
  );
  if (nodes.length !== 1) return null;
  const node = nodes[0];
  const states = mission.nodeStates.filter(
    (state) => state.revisionId === revisionId && state.nodeId === node.nodeId,
  );
  if (states.length !== 1 || states[0].status !== "succeeded") return null;
  const requestId = object(states[0].state)?.skillRequestId;
  if (typeof requestId !== "string" || !requestId) return null;
  const executions = mission.skillExecutions.filter((raw) => {
    const execution = object(raw);
    return execution?.requestId === requestId;
  });
  if (executions.length !== 1) return null;
  const execution = object(executions[0]);
  const request = object(execution?.request);
  const terminal = object(execution?.result);
  if (!execution || !request || !terminal) return null;
  if (
    execution.missionId !== mission.missionId
    || execution.revisionId !== revisionId
    || execution.nodeId !== node.nodeId
    || execution.status !== "succeeded"
    || request.requestId !== requestId
    || request.missionId !== mission.missionId
    || request.revisionId !== revisionId
    || request.nodeId !== node.nodeId
    || request.skillId !== "mission.report"
    || request.skillVersion !== "1.0.0"
    || terminal.requestId !== requestId
    || terminal.status !== "succeeded"
  ) return null;
  const result = object(terminal.result);
  return parseMissionOfficeReport(result?.missionReport, mission.missionId);
}

export function MissionOfficeReportCard({ report }: { report: MissionOfficeReport }) {
  return <section className="mission-office-report" aria-label="办公室巡检报告">
    <header>
      <div><small>OFFICE V2 · 证据汇总</small><strong>{report.reportScope === "current_location" ? "目标区域观察完成" : report.fullChainAccepted ? "全部区域已完成" : "巡检完成但仍有信息缺口"}</strong></div>
      <span>{report.evidenceCount} 份证据</span>
    </header>
    <pre>{report.narrativeZhCN}</pre>
    <div className="mission-office-region-grid">
      {report.regions.map((region) => <article key={region.regionId}>
        <div><strong>{region.name}</strong><span>{region.status === "observed" ? "已观察" : "部分观察"}</span></div>
        {region.summaries.length
          ? <ul>{region.summaries.map((summary, index) => <li key={`${region.regionId}-${index}`}>{summary}</li>)}</ul>
          : <p>未取得可靠视觉信息</p>}
      </article>)}
    </div>
    {report.missingStations.length
      ? <p className="mission-office-report-gap">缺失站点：{report.missingStations.join("、")}</p>
      : null}
  </section>;
}

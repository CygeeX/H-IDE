import { describe, expect, it } from "vitest";
import type { MissionBrief } from "../src/mission/contracts.js";
import { SkillRetriever, tokenizeSkillText } from "../src/mission/skill-retriever.js";

type MissionCapability = MissionBrief["capabilities"][number];

function capability(skillId: string, description: string): MissionCapability {
  return {
    skillId,
    skillVersion: "1",
    description,
    kind: "non_motion",
    riskLevel: "low",
    executor: "local",
    executorProfile: `${skillId}.default`,
    requiredSensorIds: [],
  };
}

function catalog() {
  return [
    capability("inspection.scan_station", "在已审核的 Office 巡检站点旋转检查并采集 RGB-D 场景描述"),
    capability("assistant.voice", "语音助手回答问题并朗读回复"),
    capability("navigation.navigate_to_pose", "通过 Nav2 导航或移动到指定地点或房间，例如办公室或会议室"),
    capability("vision.observe", "观察或查看当前 RGB-D 画面并识别电脑等目标物体"),
    capability("mission.report", "汇总站点观察并生成 Office 巡检报告"),
  ];
}

function digests(workstation = "sha256:workstation") {
  return {
    capabilityManifestDigest: "sha256:robot",
    workstationCapabilityDigest: workstation,
  };
}

describe("SkillRetriever", () => {
  it("uses precise Chinese segmentation and preserves structured skill IDs", () => {
    const chinese = tokenizeSkillText("办公室巡检");
    expect(chinese).toEqual(expect.arrayContaining(["办公室", "巡检"]));
    expect(chinese).not.toContain("办公");
    expect(chinese).not.toContain("公室");

    const identifier = tokenizeSkillText("inspection.office");
    expect(identifier).toEqual(expect.arrayContaining([
      "inspection.office",
      "inspection",
      "office",
    ]));
  });

  it("returns ranked BM25 candidates without changing the source catalog", () => {
    const capabilities = catalog();
    const retriever = new SkillRetriever({ topK: 2 });
    const selected = retriever.select("检查办公室并给我巡检报告", capabilities, digests());

    expect(selected.capabilities.map((item) => item.skillId)).toContain("inspection.scan_station");
    expect(selected.capabilities.length).toBeLessThanOrEqual(2);
    expect(selected.fallbackToFullCatalog).toBe(false);
    expect(capabilities).toHaveLength(5);
  });

  it.each([
    ["去会议室", "navigation.navigate_to_pose"],
    ["看看电脑", "vision.observe"],
  ])("retrieves the expected skill for %s", (query, expectedSkillId) => {
    const selected = new SkillRetriever({ topK: 2 }).select(query, catalog(), digests());

    expect(selected.capabilities[0]?.skillId).toBe(expectedSkillId);
  });

  it("falls back to the full catalog when BM25 has no lexical match", () => {
    const capabilities = catalog();
    const selected = new SkillRetriever({ topK: 2 }).select(
      "量子纠缠实验",
      capabilities,
      digests(),
    );

    expect(selected.fallbackToFullCatalog).toBe(true);
    expect(selected.capabilities).toEqual(capabilities);
  });

  it("retains active Mission skills and adds newly retrieved candidates", () => {
    const selected = new SkillRetriever({ topK: 2 }).select(
      "在巡检站点旋转检查",
      catalog(),
      digests(),
      ["navigation.navigate_to_pose"],
    );

    expect(selected.capabilities.map((item) => item.skillId)).toEqual(expect.arrayContaining([
      "navigation.navigate_to_pose",
      "inspection.scan_station",
    ]));
  });

  it("reuses indexes by both capability digests and bounds the cache", () => {
    const retriever = new SkillRetriever({ topK: 2, maximumCachedIndexes: 2 });
    const capabilities = catalog();

    retriever.select("办公室巡检", capabilities, digests("sha256:workstation-1"));
    retriever.select("语音助手", capabilities, digests("sha256:workstation-1"));
    expect(retriever.cachedIndexCount).toBe(1);

    retriever.select("导航", capabilities, digests("sha256:workstation-2"));
    retriever.select("观察", capabilities, digests("sha256:workstation-3"));
    expect(retriever.cachedIndexCount).toBe(2);
  });
});

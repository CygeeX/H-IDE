import { Jieba } from "@node-rs/jieba";
import { dict } from "@node-rs/jieba/dict.js";
import MiniSearch from "minisearch";
import type { MissionBrief, MissionEnvelope } from "./contracts.js";

type MissionCapability = MissionBrief["capabilities"][number];

interface SkillDocument {
  id: string;
  text: string;
}

interface CachedIndex {
  search: MiniSearch<SkillDocument>;
  capabilitiesById: ReadonlyMap<string, MissionCapability>;
}

export interface SkillMatch {
  skillId: string;
  score: number;
}

export interface SkillSelection {
  capabilities: MissionCapability[];
  matches: SkillMatch[];
  fallbackToFullCatalog: boolean;
}

export interface SkillRetrieverOptions {
  topK?: number;
  maximumCachedIndexes?: number;
}

const jieba = Jieba.withDict(dict);
const structuredIdentifier = /[a-z0-9]+(?:[._-][a-z0-9]+)+/giu;
const searchableToken = /[\p{L}\p{N}]/u;
const ignoredTokens = new Set([
  "一个", "一份", "一下", "了", "和", "并", "我", "把", "的", "给", "请", "让",
]);

export function tokenizeSkillText(text: string): string[] {
  const normalized = text.normalize("NFKC").toLowerCase();
  const tokens = jieba
    .cut(normalized, false)
    .map((token) => token.trim())
    .filter((token) => searchableToken.test(token) && !ignoredTokens.has(token));

  for (const match of normalized.matchAll(structuredIdentifier)) {
    const identifier = match[0];
    tokens.push(identifier, ...identifier.split(/[._-]+/u).filter(Boolean));
  }
  return [...new Set(tokens)];
}

function catalogKey(envelope: Pick<
  MissionEnvelope,
  "capabilityManifestDigest" | "workstationCapabilityDigest"
>): string {
  return `${envelope.capabilityManifestDigest}|${envelope.workstationCapabilityDigest}`;
}

function searchableCapabilityText(capability: MissionCapability): string {
  return [
    capability.skillId,
    capability.description,
    capability.kind,
    capability.executorProfile,
    ...capability.requiredSensorIds,
  ].join(" ");
}

export class SkillRetriever {
  private readonly topK: number;
  private readonly maximumCachedIndexes: number;
  private readonly indexes = new Map<string, CachedIndex>();

  constructor(options: SkillRetrieverOptions = {}) {
    this.topK = options.topK ?? 8;
    this.maximumCachedIndexes = options.maximumCachedIndexes ?? 16;
    if (
      !Number.isInteger(this.topK)
      || !Number.isInteger(this.maximumCachedIndexes)
      || this.topK < 1
      || this.maximumCachedIndexes < 1
    ) {
      throw new Error("SkillRetriever limits must be positive integers");
    }
  }

  get cachedIndexCount(): number {
    return this.indexes.size;
  }

  select(
    query: string,
    capabilities: MissionCapability[],
    envelope: Pick<MissionEnvelope, "capabilityManifestDigest" | "workstationCapabilityDigest">,
    requiredSkillIds: Iterable<string> = [],
  ): SkillSelection {
    if (capabilities.length <= this.topK) {
      return { capabilities: [...capabilities], matches: [], fallbackToFullCatalog: false };
    }

    const index = this.getOrCreateIndex(capabilities, envelope);
    const results = index.search.search(query).slice(0, this.topK);
    if (results.length === 0) {
      return { capabilities: [...capabilities], matches: [], fallbackToFullCatalog: true };
    }

    const selectedIds = new Set<string>();
    const selected: MissionCapability[] = [];
    const append = (skillId: string) => {
      if (selectedIds.has(skillId)) return;
      const capability = index.capabilitiesById.get(skillId);
      if (!capability) return;
      selectedIds.add(skillId);
      selected.push(capability);
    };

    for (const skillId of requiredSkillIds) append(skillId);
    for (const result of results) append(String(result.id));

    return {
      capabilities: selected,
      matches: results.map((result) => ({ skillId: String(result.id), score: result.score })),
      fallbackToFullCatalog: false,
    };
  }

  private getOrCreateIndex(
    capabilities: MissionCapability[],
    envelope: Pick<MissionEnvelope, "capabilityManifestDigest" | "workstationCapabilityDigest">,
  ): CachedIndex {
    const key = catalogKey(envelope);
    const cached = this.indexes.get(key);
    if (cached) {
      this.indexes.delete(key);
      this.indexes.set(key, cached);
      return cached;
    }

    const search = new MiniSearch<SkillDocument>({
      fields: ["text"],
      tokenize: tokenizeSkillText,
    });
    search.addAll(capabilities.map((capability) => ({
      id: capability.skillId,
      text: searchableCapabilityText(capability),
    })));
    const created = {
      search,
      capabilitiesById: new Map(capabilities.map((capability) => [capability.skillId, capability])),
    };
    this.indexes.set(key, created);

    while (this.indexes.size > this.maximumCachedIndexes) {
      const oldestKey = this.indexes.keys().next().value;
      if (oldestKey === undefined) break;
      this.indexes.delete(oldestKey);
    }
    return created;
  }
}

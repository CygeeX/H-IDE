import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  planFragmentSchema,
  planProposalSchema,
  planningEnvelopeSchema,
  planningTurnResultSchema,
} from "../src/contracts.js";
import { envelope, planProposal } from "./fixtures.js";

describe("planning contracts", () => {
  it("accepts the versioned H-IDE envelope and rejects transport details", () => {
    expect(planningEnvelopeSchema.parse(envelope()).taskRequest.schemaVersion).toBe("1.0");
    expect(() => planningEnvelopeSchema.parse(envelope({ mapping: { topic: "/cmd_vel" } }))).toThrow();
  });

  it("requires strict worker contributions and a valid merged DAG", () => {
    expect(planProposalSchema.parse(planProposal()).contributions).toHaveLength(2);
    expect(() => planProposalSchema.parse({ ...planProposal(), contributions: [] })).toThrow();
    expect(() => planProposalSchema.parse({ ...planProposal(), steps: planProposal().steps.slice(0, 1) }))
      .toThrow(/omit|fragment/i);
    const cyclic = planProposal();
    cyclic.steps[0]!.dependsOn = ["open"];
    expect(() => planProposalSchema.parse(cyclic)).toThrow(/cycle/i);
  });

  it("does not accept a malformed or empty worker result", () => {
    expect(() => planFragmentSchema.parse({ agentId: "motion-worker", steps: [] })).toThrow();
    expect(
      planningTurnResultSchema.parse({ status: "needs_input", assistantMessage: "目标在哪里？" }).status,
    ).toBe("needs_input");
  });

  it("keeps opaque command payloads open without DashScope-incompatible schema keywords", () => {
    const arbitraryPayload = {
      nested: { enabled: true, values: [1, "two", null] },
      vendorExtension: "preserved",
    };
    const parsed = planFragmentSchema.parse({
      agentId: "external-worker",
      contribution: "vendor command",
      summary: "Preserve an opaque JSON payload.",
      steps: [{
        stepId: "vendor-step",
        commandKey: "vendor.execute",
        payload: arbitraryPayload,
        dependsOn: [],
        rationale: "Exercise the open payload boundary.",
      }],
      assumptions: [],
      warnings: [],
    });
    expect(parsed.steps[0]?.payload).toEqual(arbitraryPayload);
    expect(() => planFragmentSchema.parse({
      ...parsed,
      steps: [{ ...parsed.steps[0]!, payload: [] }],
    })).toThrow();

    const wireSchema = z.toJSONSchema(planFragmentSchema);
    const forbiddenPaths: string[] = [];
    const visit = (value: unknown, path = "$") => {
      if (value == null || typeof value !== "object") return;
      const record = value as Record<string, unknown>;
      if ("propertyNames" in record || "patternProperties" in record) {
        forbiddenPaths.push(path);
      }
      for (const [key, child] of Object.entries(record)) visit(child, `${path}.${key}`);
    };
    visit(wireSchema);
    expect(forbiddenPaths).toEqual([]);
  });
});

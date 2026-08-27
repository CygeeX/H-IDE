import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SimulationWorkspace } from "./simulation-workspace";

describe("SimulationWorkspace", () => {
  it("loads the observatory without presenting a physical actuator", async () => {
    const fakeApiFetch = async <T,>(path: string): Promise<T> => {
      if (path === "/api/v2/simulations") return [] as T;
      if (path === "/api/v2/simulations/backends") {
        return [
          { backend: "deterministic", available: true },
          { backend: "gazebo", available: false, installed: false },
        ] as T;
      }
      return [] as T;
    };
    const apiFetch = fakeApiFetch;

    render(<SimulationWorkspace apiFetch={apiFetch} apiFetchBlob={vi.fn()} onBack={vi.fn()} />);

    expect(await screen.findByText("连续任务仿真联调")).toBeInTheDocument();
    expect(screen.getByText("SIMULATION · NO PHYSICAL ACTUATOR")).toBeInTheDocument();
    expect(screen.getByText("还没有仿真会话")).toBeInTheDocument();
  });
});

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  LEADER_SKILLS,
  ROBOT_DISPLAY_OPTIONS,
  RobotPicker,
  SkillSelection,
  browserSpeakChinese,
} from "./leader-demo";

describe("leader demo entry points", () => {
  it("exposes inspection and voice skills with the agreed greeting prompts", () => {
    const onSelect = vi.fn();
    render(<SkillSelection selected={null} onSelect={onSelect} />);
    expect(screen.getByRole("button", { name: /巡检 Skill/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /语音助手 Skill/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /巡检 Skill/ }));
    expect(onSelect).toHaveBeenCalledWith(LEADER_SKILLS[0]);
    expect(LEADER_SKILLS[0].prompt).toContain("巡检开放办公室");
  });

  it("keeps display-only robot cards on the bound Unitree backend", () => {
    const onRobotChange = vi.fn();
    const onDisplayChange = vi.fn();
    render(
      <RobotPicker
        robots={[{ robotId: "g1-sim", name: "G1 仿真", connected: true } as never]}
        selectedRobotId="g1-sim"
        selectedDisplayId="unitree"
        onRobotChange={onRobotChange}
        onDisplayChange={onDisplayChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "选择机器人" }));
    expect(screen.getByText(/智元 AgiBot/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /智元 AgiBot/ }));
    expect(onDisplayChange).toHaveBeenCalledWith("agibot");
    expect(onRobotChange).not.toHaveBeenCalled();
    expect(ROBOT_DISPLAY_OPTIONS.map((item) => item.id)).toEqual([
      "unitree",
      "agibot",
      "deeprobotics",
    ]);
  });

  it("does not throw when browser speech synthesis is unavailable", () => {
    expect(browserSpeakChinese("你好")).toBe(false);
  });
});

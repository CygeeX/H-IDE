import { useEffect, useState } from "react";
import type { RobotState } from "./contracts";

/** The two operator-facing entry points used by the leader demonstration. */
export type LeaderSkillId = "inspection.office" | "assistant.voice";

export interface LeaderSkillDefinition {
  id: LeaderSkillId;
  title: string;
  eyebrow: string;
  description: string;
  prompt: string;
  accent: "violet" | "cyan";
}

export const LEADER_SKILLS: readonly LeaderSkillDefinition[] = [
  {
    id: "inspection.office",
    title: "巡检 Skill",
    eyebrow: "OFFICE INSPECTION",
    description: "进入指定办公室，多视角观察并汇总房间与沿途信息。",
    prompt: "今天还是巡检开放办公室吗？",
    accent: "violet",
  },
  {
    id: "assistant.voice",
    title: "语音助手 Skill",
    eyebrow: "VOICE ASSISTANT",
    description: "用自然语言问我问题，我会用中文语音回答并向你挥手。",
    prompt: "语音助手已就绪，你可以问我任何问题。",
    accent: "cyan",
  },
];

export interface RobotDisplayOption {
  id: "unitree" | "agibot" | "deeprobotics";
  company: string;
  model: string;
  image: string;
  control: "active" | "display-only";
}

export const ROBOT_DISPLAY_OPTIONS: readonly RobotDisplayOption[] = [
  {
    id: "unitree",
    company: "宇树 Unitree",
    model: "G1 · 当前控制机器人",
    image: "/assets/leader-demo/robots/unitree-g1.png",
    control: "active",
  },
  {
    id: "agibot",
    company: "智元 AgiBot",
    model: "展示模型 · 适配预览",
    image: "/assets/leader-demo/robots/agibot.png",
    control: "display-only",
  },
  {
    id: "deeprobotics",
    company: "云深处 DEEP Robotics",
    model: "展示模型 · 适配预览",
    image: "/assets/leader-demo/robots/deeprobotics.png",
    control: "display-only",
  },
];

export function SkillSelection({
  selected,
  onSelect,
}: {
  selected: LeaderSkillId | null;
  onSelect: (skill: LeaderSkillDefinition) => void;
}) {
  return (
    <section className="leader-skill-selection" aria-label="选择 Skill">
      <div className="leader-skill-heading">
        <strong>先选择今天要使用的能力</strong>
        <small>选择后，下面的对话会沿着对应能力继续；机器人行动仍由当前 Mission 审核链负责。</small>
      </div>
      <div className="leader-skill-grid">
        {LEADER_SKILLS.map((skill) => (
          <button
            className={`leader-skill-card accent-${skill.accent} ${selected === skill.id ? "selected" : ""}`}
            type="button"
            key={skill.id}
            onClick={() => onSelect(skill)}
            aria-pressed={selected === skill.id}
          >
            <span className="leader-skill-icon" aria-hidden="true">
              {skill.id === "inspection.office" ? "⌂" : "◌"}
            </span>
            <span className="leader-skill-copy">
              <small>{skill.eyebrow}</small>
              <strong>{skill.title}</strong>
              <span>{skill.description}</span>
            </span>
            <span className="leader-skill-arrow" aria-hidden="true">→</span>
          </button>
        ))}
      </div>
    </section>
  );
}

export function SkillModeBanner({
  skill,
  onClear,
}: {
  skill: LeaderSkillDefinition;
  onClear: () => void;
}) {
  return (
    <div className={`leader-skill-banner accent-${skill.accent}`}>
      <span className="leader-skill-banner-dot" />
      <span><b>{skill.title}</b><small>{skill.id}</small></span>
      <button type="button" onClick={onClear}>切换 Skill</button>
    </div>
  );
}

/**
 * A card-based picker.  Display-only cards deliberately retain the currently
 * bound backend robot; they are visual choices for the recording, not hidden
 * transport switches.
 */
export function RobotPicker({
  robots,
  selectedRobotId,
  selectedDisplayId,
  onRobotChange,
  onDisplayChange,
  compact = false,
}: {
  robots: RobotState[];
  selectedRobotId: string;
  selectedDisplayId: RobotDisplayOption["id"];
  onRobotChange: (robotId: string) => void;
  onDisplayChange: (displayId: RobotDisplayOption["id"]) => void;
  compact?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const selected = ROBOT_DISPLAY_OPTIONS.find((item) => item.id === selectedDisplayId) ?? ROBOT_DISPLAY_OPTIONS[0];
  const boundRobot = robots.find((item) => item.robotId === selectedRobotId);
  const readyRobot = robots.find(
    (item) => item.connected && (item.status === "online" || item.status === "degraded"),
  );
  const simulationRobot = robots.find(
    (item) => item.robotId === "g1-sim"
      && item.connected
      && (item.status === "online" || item.status === "degraded"),
  );
  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);
  const choose = (option: RobotDisplayOption) => {
    onDisplayChange(option.id);
    // Only Unitree is wired to the live/simulation backend in this demo.
    if (option.control === "active") {
      const current = robots.find((item) => item.robotId === selectedRobotId);
      const preferred = (
        current?.connected
        && (current.status === "online" || current.status === "degraded")
          ? current
          : simulationRobot
            ?? readyRobot
      )
        ?? current
        ?? robots[0];
      if (preferred) onRobotChange(preferred.robotId);
    }
    setOpen(false);
  };
  return (
    <div className={`leader-robot-picker ${compact ? "compact" : ""}`}>
      <select
        className="sr-only leader-robot-native-select"
        aria-label={compact ? "当前机器人" : "对话机器人"}
        value={selectedRobotId}
        onChange={(event) => onRobotChange(event.target.value)}
      >
        {robots.length ? robots.map((item, index) => (
          <option key={item.robotId} value={item.robotId}>{item.name ?? `机器人 ${index + 1}`}{compact ? ` (${item.robotId})` : ""}</option>
        )) : <option value={selectedRobotId}>{selectedRobotId}</option>}
      </select>
      <button
        className="leader-robot-trigger"
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-controls="leader-robot-options"
        aria-label="选择机器人"
      >
        <img src={selected.image} alt="" />
        <span><b>{selected.company}</b><small>{selected.control === "active" ? (boundRobot?.name ?? "G1 仿真") : "展示预览"}</small></span>
        {compact ? <span className="sr-only">{selectedRobotId}</span> : null}
        <i aria-hidden="true">⌄</i>
      </button>
      {open ? (
        <div className="leader-picker-modal-layer">
          <button className="leader-picker-scrim" type="button" onClick={() => setOpen(false)} aria-label="关闭机器人选择" />
          <div className="leader-robot-options" id="leader-robot-options" role="dialog" aria-modal="true" aria-labelledby="leader-robot-options-title">
            <header><span><strong id="leader-robot-options-title">选择机器人</strong><small>当前演示只调用宇树 G1 控制链路</small></span><button className="leader-robot-options-close" type="button" onClick={() => setOpen(false)} aria-label="关闭机器人选择">×</button></header>
            <div className="leader-robot-option-grid">
              {ROBOT_DISPLAY_OPTIONS.map((option) => (
                <button
                  className={`leader-robot-option ${selectedDisplayId === option.id ? "selected" : ""}`}
                  type="button"
                  key={option.id}
                  onClick={() => choose(option)}
                >
                  <img src={option.image} alt={`${option.company} ${option.model}`} />
                  <span><b>{option.company}</b><small>{option.model}</small></span>
                  <em>{option.control === "active" ? "已接入" : "展示"}</em>
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function browserSpeakChinese(text: string): boolean {
  if (typeof window === "undefined" || !("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) return false;
  const value = text.trim();
  if (!value) return false;
  const synth = window.speechSynthesis;
  synth.cancel();
  const utterance = new SpeechSynthesisUtterance(value);
  utterance.lang = "zh-CN";
  utterance.rate = 0.96;
  utterance.pitch = 1.04;
  synth.speak(utterance);
  return true;
}

interface RobotAvatarProps {
  speech?: string;
  compact?: boolean;
  className?: string;
}

/** CSS-only assistant avatar. It avoids a heavyweight/low-quality GLB while
 * preserving the themed motion and speech bubble used by the demo shell. */
export function RobotAvatar({ speech, compact = false, className = "" }: RobotAvatarProps) {
  return (
    <div className={`robot-avatar ${compact ? "compact" : ""} ${className}`}>
      <div className="robot-avatar-orbit" aria-hidden="true">
        <span className="robot-avatar-ring ring-a" />
        <span className="robot-avatar-ring ring-b" />
        <div className="robot-avatar-face" aria-label="H-IDE 机器人头像">
          <div className="robot-avatar-visor"><i /><i /><b /></div>
          <span className="robot-avatar-mark"><img src="/assets/leader-demo/brand/h-ide-logo.png" alt="" /></span>
        </div>
      </div>
      {speech ? <div className="robot-speech-bubble" role="status">{speech}</div> : null}
    </div>
  );
}

// Kept as a source-compatible alias for integrations that imported the old
// name; it no longer loads or renders a GLB.
export const RobotAvatar3D = RobotAvatar;

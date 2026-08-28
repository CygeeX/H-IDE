import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
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
        <span className="leader-skill-kicker">LEADER DEMO · SKILLS</span>
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
  const choose = (option: RobotDisplayOption) => {
    onDisplayChange(option.id);
    // Only Unitree is wired to the live/simulation backend in this demo.
    if (option.control === "active") {
      const preferred = robots.find((item) => item.robotId === selectedRobotId)
        ?? robots.find((item) => item.connected)
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
        aria-label="选择机器人"
      >
        <img src={selected.image} alt="" />
        <span><b>{selected.company}</b><small>{selected.control === "active" ? (boundRobot?.name ?? "G1 仿真") : "展示预览"}</small></span>
        {compact ? <span className="sr-only">{selectedRobotId}</span> : null}
        <i aria-hidden="true">⌄</i>
      </button>
      {open ? (
        <>
          <button className="leader-picker-scrim" type="button" onClick={() => setOpen(false)} aria-label="关闭机器人选择" />
          <div className="leader-robot-options" role="dialog" aria-label="机器人型号选择">
            <header><strong>选择机器人</strong><small>当前演示只调用宇树 G1 控制链路</small></header>
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
        </>
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

interface RobotAvatar3DProps {
  speech?: string;
  compact?: boolean;
  className?: string;
}

/** Lightweight Three.js wrapper used by both login and conversation shells. */
export function RobotAvatar3D({ speech, compact = false, className = "" }: RobotAvatar3DProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const modelRef = useRef<THREE.Object3D | null>(null);
  const pointerRef = useRef({ x: 0, y: 0 });
  const targetRef = useRef({ x: 0, y: 0 });
  const [loadError, setLoadError] = useState(false);
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;
    let disposed = false;
    if (typeof navigator !== "undefined" && /jsdom/i.test(navigator.userAgent)) {
      setLoadError(true);
      return undefined;
    }
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 100);
    camera.position.set(0, 0.05, compact ? 3.25 : 3.1);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "high-performance" });
    } catch {
      // Headless test runners and browsers without WebGL still receive the
      // semantic avatar/speech shell; a real browser uses the full GLB path.
      setLoadError(true);
      return undefined;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    host.appendChild(renderer.domElement);
    const ambient = new THREE.HemisphereLight(0xf5ecff, 0x3c2b56, 2.4);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffffff, 3.1);
    key.position.set(2.4, 3.2, 4.5);
    scene.add(key);
    const rim = new THREE.PointLight(0x8bdcff, 2.2, 8);
    rim.position.set(-2, 1, 2);
    scene.add(rim);
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(compact ? 0.7 : 0.82, 0.012, 8, 96),
      new THREE.MeshBasicMaterial({ color: 0xa394ff, transparent: true, opacity: 0.45 }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.y = -0.54;
    scene.add(ring);
    const loader = new GLTFLoader();
    loader.load(
      "/assets/leader-demo/models/h-ide-head.glb?v=1ea6fee4",
      (gltf) => {
        if (disposed) return;
        const model = gltf.scene;
        const bounds = new THREE.Box3().setFromObject(model);
        const size = bounds.getSize(new THREE.Vector3());
        const center = bounds.getCenter(new THREE.Vector3());
        const scale = 1.35 / Math.max(size.x, size.y, size.z, 0.01);
        model.scale.setScalar(scale);
        model.position.set(-center.x * scale, -center.y * scale - 0.28, -center.z * scale);
        model.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            object.castShadow = false;
            object.receiveShadow = false;
            const material = object.material;
            if (Array.isArray(material)) material.forEach((item) => { item.needsUpdate = true; });
            else if (material) material.needsUpdate = true;
          }
        });
        modelRef.current = model;
        scene.add(model);
      },
      undefined,
      () => { if (!disposed) setLoadError(true); },
    );
    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const onPointerMove = (event: PointerEvent) => {
      const bounds = host.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      pointerRef.current.x = Math.max(-1, Math.min(1, ((event.clientX - bounds.left) / bounds.width) * 2 - 1));
      pointerRef.current.y = Math.max(-1, Math.min(1, ((event.clientY - bounds.top) / bounds.height) * 2 - 1));
      targetRef.current.x = pointerRef.current.y * 0.18;
      targetRef.current.y = pointerRef.current.x * 0.35;
    };
    const onPointerLeave = () => { targetRef.current.x = 0; targetRef.current.y = 0; };
    host.addEventListener("pointermove", onPointerMove);
    host.addEventListener("pointerleave", onPointerLeave);
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(resize);
    observer?.observe(host);
    resize();
    let frame = 0;
    const animate = () => {
      if (disposed) return;
      frame = window.requestAnimationFrame(animate);
      const model = modelRef.current;
      if (model) {
        model.rotation.x += (targetRef.current.x - model.rotation.x) * 0.08;
        model.rotation.y += (targetRef.current.y - model.rotation.y) * 0.08;
      }
      ring.rotation.z += 0.004;
      renderer.render(scene, camera);
    };
    animate();
    return () => {
      disposed = true;
      window.cancelAnimationFrame(frame);
      observer?.disconnect();
      host.removeEventListener("pointermove", onPointerMove);
      host.removeEventListener("pointerleave", onPointerLeave);
      modelRef.current = null;
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const material = object.material;
          if (Array.isArray(material)) material.forEach((item) => item.dispose());
          else material?.dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [compact]);
  return (
    <div className={`robot-avatar-3d ${compact ? "compact" : ""} ${loadError ? "load-error" : ""} ${className}`}>
      <div className="robot-avatar-3d-canvas" ref={hostRef} aria-label="H-IDE 机器人 3D 头像" />
      <div className="robot-avatar-3d-glow" aria-hidden="true" />
      {speech ? <div className="robot-speech-bubble" role="status">{speech}</div> : null}
      {loadError ? <span className="robot-avatar-3d-fallback" aria-hidden="true">H</span> : null}
    </div>
  );
}

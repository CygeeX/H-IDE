"use strict";

const terminalStates = new Set([
  "SUCCEEDED",
  "FAILED",
  "CANCELLED",
  "REJECTED",
  "TIMED_OUT",
  "ABORTED_AGENT_RESTART",
]);

const elements = Object.fromEntries(
  [
    "uiStatus",
    "agentStatus",
    "modeBadge",
    "modeToggle",
    "autoDemoBtn",
    "stageState",
    "stageReason",
    "controlLed",
    "controlState",
    "controlReason",
    "acquireBtn",
    "powerOnBtn",
    "powerOffBtn",
    "taskOrder",
    "taskBtn",
    "navX",
    "navY",
    "navYaw",
    "vlnBtn",
    "cancelBtn",
    "leaseBtn",
    "freshness",
    "robotMode",
    "robotModeHint",
    "leaseProgress",
    "leaseCountdown",
    "rxMetric",
    "txMetric",
    "leaseSequence",
    "commandMetric",
    "robotId",
    "bootId",
    "capabilityId",
    "commandList",
    "eventList",
    "clearEventsBtn",
    "footerClock",
    "toast",
    "observationStatus",
    "observationSource",
    "observationFreshness",
    "observationLed",
    "rgbImage",
    "rgbPlaceholder",
    "depthImage",
    "depthPlaceholder",
    "observationSequence",
    "rgbFrameNumber",
    "depthFrameNumber",
    "frameBytes",
    "frameTf",
    "captureAgeBar",
    "captureAgeMetric",
    "encodeBar",
    "encodeMetric",
    "queueBar",
    "queueMetric",
    "ackBar",
    "ackMetric",
    "networkBar",
    "networkMetric",
    "sendFpsMetric",
    "bandwidthMetric",
    "acceptedMetric",
    "dropMetric",
    "gapMetric",
    "rejectedMetric",
    "observationSession",
    "observationScene",
    "observationCalibration",
    "observationReason",
    "ackFaultBtn",
    "rejectFrameBtn",
    "sessionFaultBtn",
    "cameraFaultBtn",
  ].map((id) => [id, document.getElementById(id)]),
);

let socket = null;
let reconnectTimer = null;
let demoMode = false;
let autoDemoRunning = false;
let toastTimer = null;
let lastSnapshotAt = 0;

const emptySnapshot = () => ({
  type: "snapshot",
  generated_at: new Date().toISOString(),
  robot_id: "g1-dev",
  agent: { connected: false },
  control: { state: "SAFE_HOLD", reason: "等待 Robot Agent 连接" },
  lease: {
    enabled: false,
    sequence: null,
    expires_at: null,
    interval_ms: 500,
    timeout_ms: 2000,
  },
  robot_state: {},
  commands: [],
  events: [],
  metrics: {
    messages_from_agent: 0,
    messages_to_agent: 0,
    last_agent_message_at: null,
    ui_clients: 0,
  },
  observation: { enabled: false, health: { status: "OFFLINE", reason: "NOT_STARTED" } },
});

let state = emptySnapshot();
let lastObservationSequence = null;
let lastObservationFrameAt = 0;

const demoState = {
  sequence: 11,
  commandCounter: 0,
  interval: null,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function shortId(value, length = 12) {
  if (!value) return "—";
  return String(value).length > length ? `${String(value).slice(0, length)}…` : String(value);
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", isError);
  elements.toast.classList.add("is-visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("is-visible"), 3200);
}

function sendAction(action) {
  if (demoMode) {
    handleDemoAction(action);
    return;
  }
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    showToast("测试 BFF 尚未连接。可进入浏览器演示模式查看效果。", true);
    return;
  }
  socket.send(JSON.stringify(action));
}

function connect() {
  if (demoMode) return;
  window.clearTimeout(reconnectTimer);
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${window.location.host}/ui`, "robot-agent-demo.v1");

  socket.addEventListener("open", () => {
    elements.uiStatus.classList.add("is-online");
    elements.uiStatus.innerHTML = "<i></i>BFF UI 已连接";
  });

  socket.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      showToast("收到无法解析的 UI 消息", true);
      return;
    }
    if (message.type === "snapshot") {
      state = message;
      lastSnapshotAt = Date.now();
      render();
      return;
    }
    if (message.type === "error") {
      showToast(`${message.code}: ${message.message}`, true);
    }
  });

  socket.addEventListener("close", () => {
    elements.uiStatus.classList.remove("is-online");
    elements.uiStatus.innerHTML = "<i></i>BFF UI 重连中";
    if (!demoMode) reconnectTimer = window.setTimeout(connect, 1200);
  });

  socket.addEventListener("error", () => {
    elements.uiStatus.classList.remove("is-online");
  });
}

function render() {
  const connected = Boolean(state.agent?.connected);
  const active = state.control?.state === "ACTIVE";
  const leaseEnabled = Boolean(state.lease?.enabled);
  const mode = state.robot_state?.["robot.mode"] ?? state.robot_state?.robot?.mode ?? "—";

  document.body.classList.toggle("state-active", active);
  elements.agentStatus.classList.toggle("is-online", connected);
  elements.agentStatus.innerHTML = connected ? "<i></i>Agent 在线" : "<i></i>等待 Agent";
  elements.stageState.querySelector("strong").textContent = state.control?.state ?? "SAFE_HOLD";
  elements.stageReason.textContent = state.control?.reason ?? "—";
  elements.controlState.textContent = state.control?.state ?? "SAFE_HOLD";
  elements.controlReason.textContent = state.control?.reason ?? "—";
  elements.controlLed.classList.toggle("is-active", active);
  elements.controlLed.innerHTML = active ? "<i></i>ACTIVE" : "<i></i>LOCKED";

  elements.acquireBtn.disabled = !demoMode && !connected;
  elements.acquireBtn.textContent = active ? "已获取控制" : "获取控制权";
  for (const button of [elements.powerOnBtn, elements.powerOffBtn, elements.taskBtn]) {
    button.disabled = !active;
  }
  elements.vlnBtn.disabled = !active || state.observation?.health?.status !== "USABLE";
  elements.cancelBtn.disabled = !active || !latestCancelableCommand();
  elements.leaseBtn.disabled = !active;
  elements.leaseBtn.textContent = leaseEnabled ? "暂停续租" : "恢复续租";

  elements.robotMode.textContent = mode;
  elements.robotModeHint.textContent = mode === "—" ? "等待 /robot/mode" : "std_msgs/String · normalized";
  elements.rxMetric.textContent = state.metrics?.messages_from_agent ?? 0;
  elements.txMetric.textContent = state.metrics?.messages_to_agent ?? 0;
  elements.leaseSequence.textContent = state.lease?.sequence ?? "—";
  elements.commandMetric.textContent = state.commands?.length ?? 0;
  elements.robotId.textContent = state.robot_id ?? "—";
  elements.bootId.textContent = shortId(state.agent?.boot_id, 18);
  elements.capabilityId.textContent = shortId(state.agent?.capabilities_digest, 20);
  elements.freshness.textContent = connected ? `LIVE · ${formatTime(state.generated_at)}` : "NO AGENT DATA";

  renderLease();
  renderArchitecture(connected);
  renderCommands();
  renderEvents();
  renderObservation();
}

function metric(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function byteSize(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(2)} MiB`
    : `${(bytes / 1024).toFixed(0)} KiB`;
}

function setLatency(element, value, budget) {
  const number = Number(value);
  element.style.width = `${Number.isFinite(number) ? Math.min(100, Math.max(2, (number / budget) * 100)) : 0}%`;
  element.classList.toggle("is-danger", Number.isFinite(number) && number > budget);
}

function renderObservation() {
  const observation = state.observation ?? {};
  const health = observation.health ?? {};
  const frame = observation.frame ?? {};
  const telemetry = observation.telemetry ?? {};
  const enabled = Boolean(observation.enabled);
  const usable = health.status === "USABLE";
  const streaming = Boolean(observation.observationSessionId);
  const sequence = observation.lastSequence;

  if (sequence != null && sequence !== lastObservationSequence) {
    lastObservationSequence = sequence;
    lastObservationFrameAt = Date.now();
    const cacheKey = `${sequence}-${Date.now()}`;
    if (observation.hasRgb) {
      elements.rgbImage.src = `/api/observation/rgb.jpg?v=${cacheKey}`;
      elements.rgbImage.hidden = false;
      elements.rgbPlaceholder.hidden = true;
    }
    if (observation.hasDepth) {
      elements.depthImage.src = `/api/observation/depth.png?v=${cacheKey}`;
      elements.depthImage.hidden = false;
      elements.depthPlaceholder.hidden = true;
    }
  }

  const statusText = enabled ? health.status ?? "OFFLINE" : "DISABLED";
  elements.observationStatus.classList.toggle("is-usable", usable);
  elements.observationStatus.classList.toggle("is-degraded", enabled && !usable);
  elements.observationStatus.innerHTML = `<i></i>${escapeHtml(statusText)}`;
  elements.observationSource.textContent = `SOURCE · ${(observation.source ?? "—").toUpperCase()}`;
  elements.observationLed.classList.toggle("is-active", usable);
  elements.observationLed.classList.toggle("is-danger", enabled && !usable);
  elements.observationLed.innerHTML = `<i></i>${escapeHtml(statusText)}`;
  elements.observationFreshness.textContent = sequence == null
    ? "NO FRAME"
    : `${Date.now() - lastObservationFrameAt < 1200 ? "LIVE" : "STALE"} · #${sequence}`;

  elements.observationSequence.textContent = sequence ?? "—";
  elements.rgbFrameNumber.textContent = frame.rgbFrameNumber ?? "—";
  elements.depthFrameNumber.textContent = frame.depthFrameNumber ?? "—";
  elements.frameBytes.textContent = byteSize(frame.serializedBytes);
  elements.frameTf.textContent = frame.mapFrame && frame.cameraFrame
    ? `${frame.mapFrame}→${frame.cameraFrame}`
    : "—";

  elements.captureAgeMetric.textContent = `${metric(frame.captureAgeMs)} ms`;
  elements.encodeMetric.textContent = `${metric(frame.encodeMs)} ms`;
  elements.queueMetric.textContent = `${metric(frame.queueMs)} ms`;
  elements.ackMetric.textContent = `${metric(telemetry.ackRttMs)} ms`;
  elements.networkMetric.textContent = `${metric(telemetry.networkMs)} ms`;
  setLatency(elements.captureAgeBar, frame.captureAgeMs, 500);
  setLatency(elements.encodeBar, frame.encodeMs, 80);
  setLatency(elements.queueBar, frame.queueMs, 100);
  setLatency(elements.ackBar, telemetry.ackRttMs, 750);
  setLatency(elements.networkBar, telemetry.networkMs, 750);

  elements.sendFpsMetric.textContent = metric(telemetry.actualSendFps, 2);
  elements.bandwidthMetric.textContent = metric(telemetry.bandwidthMbps, 2);
  elements.acceptedMetric.textContent = observation.acceptedFrames ?? 0;
  elements.dropMetric.textContent = telemetry.dropCount ?? health.dropCount ?? 0;
  elements.gapMetric.textContent = observation.sequenceGap ?? 0;
  elements.rejectedMetric.textContent = observation.rejectedFrames ?? 0;
  elements.observationSession.textContent = shortId(observation.observationSessionId, 22);
  elements.observationScene.textContent = observation.sceneVersion || "—";
  elements.observationCalibration.textContent = shortId(frame.calibrationSha256, 22);
  elements.observationReason.textContent = health.reason ?? "—";

  const activePipeline = enabled && observation.connected;
  for (const node of document.querySelectorAll("[data-observation-node]")) {
    node.classList.toggle("is-active", activePipeline);
  }
  document.querySelector('[data-observation-node="ack"]')?.classList.toggle("is-usable", usable);
  document.querySelector('[data-observation-node="camera"]')?.classList.toggle(
    "is-danger",
    Boolean(observation.faults?.cameraPaused),
  );
  document.querySelector('[data-observation-node="ack"]')?.classList.toggle(
    "is-danger",
    Boolean(observation.faults?.acksPaused) || health.reason === "ACK_STALE",
  );

  elements.ackFaultBtn.disabled = !enabled;
  elements.ackFaultBtn.textContent = observation.faults?.acksPaused ? "恢复 ACK" : "暂停 ACK";
  elements.rejectFrameBtn.disabled = !enabled || !streaming;
  elements.sessionFaultBtn.disabled = !enabled || !observation.connected;
  elements.sessionFaultBtn.textContent = streaming ? "停止观测会话" : "启动观测会话";
  elements.cameraFaultBtn.disabled = !enabled || observation.source !== "synthetic";
  elements.cameraFaultBtn.textContent = observation.faults?.cameraPaused ? "恢复模拟相机" : "暂停模拟相机";
}

function renderLease() {
  const timeout = Number(state.lease?.timeout_ms ?? 2000);
  let remaining = 0;
  if (state.lease?.expires_at) {
    remaining = Math.max(0, new Date(state.lease.expires_at).valueOf() - Date.now());
  } else if (demoMode && state.lease?.enabled) {
    remaining = timeout;
  }
  const ratio = timeout > 0 ? Math.min(1, remaining / timeout) : 0;
  elements.leaseProgress.max = timeout;
  elements.leaseProgress.value = remaining;
  elements.leaseCountdown.textContent = remaining > 0 ? `${(remaining / 1000).toFixed(1)}s` : "—";
  const ring = elements.leaseProgress.closest(".gauge-ring");
  ring.classList.remove("lease-100", "lease-75", "lease-50", "lease-25", "lease-0");
  ring.classList.add(
    ratio > 0.87 ? "lease-100" : ratio > 0.62 ? "lease-75" : ratio > 0.37 ? "lease-50" : ratio > 0 ? "lease-25" : "lease-0",
  );
}

function renderArchitecture(connected) {
  for (const name of ["browser", "bff"]) {
    document.querySelector(`[data-node="${name}"]`)?.classList.add("is-active");
  }
  for (const name of ["agent", "ros", "robot"]) {
    document.querySelector(`[data-node="${name}"]`)?.classList.toggle("is-active", connected || demoMode);
  }
  document.querySelector('[data-link="ui-bff"]')?.classList.add("is-active");
  for (const name of ["bff-agent", "agent-ros", "ros-robot"]) {
    document.querySelector(`[data-link="${name}"]`)?.classList.toggle("is-active", connected || demoMode);
  }
}

function renderCommands() {
  const commands = state.commands ?? [];
  if (!commands.length) {
    elements.commandList.innerHTML = `
      <div class="empty-state">
        <span>⌁</span><strong>还没有命令</strong>
        <p>获取控制权，然后发送动力或 Action 任务。</p>
      </div>`;
    return;
  }
  elements.commandList.innerHTML = commands
    .map((command) => {
      const stateClass = `state-${String(command.state ?? "unknown").toLowerCase()}`;
      const result = command.result ? JSON.stringify(command.result) : "";
      return `
        <article class="command-row">
          <div class="command-name">
            <strong>${escapeHtml(command.kind)}</strong>
            <small>${escapeHtml(shortId(command.command_id, 18))}</small>
          </div>
          <span class="state-chip ${stateClass}">${escapeHtml(command.state)}</span>
          <div class="command-detail">
            <strong>${escapeHtml(command.detail)}</strong>
            <small>${escapeHtml(result)}</small>
          </div>
          <time class="command-time">${formatTime(command.updated_at)}</time>
        </article>`;
    })
    .join("");
}

function renderEvents() {
  const events = state.events ?? [];
  if (!events.length) {
    elements.eventList.innerHTML = `
      <div class="empty-state"><span>·</span><strong>事件记录为空</strong></div>`;
    return;
  }
  elements.eventList.innerHTML = events
    .map(
      (event) => `
        <article class="event-row level-${escapeHtml(event.level)}">
          <i class="event-dot"></i>
          <div class="event-content">
            <div class="event-meta">
              <strong>${escapeHtml(event.title)}</strong>
              <time>${formatTime(event.at)}</time>
            </div>
            <p>${escapeHtml(event.detail)}</p>
          </div>
        </article>`,
    )
    .join("");
}

function latestCancelableCommand() {
  return (state.commands ?? []).find((command) => !terminalStates.has(command.state));
}

function enterDemoMode() {
  demoMode = true;
  document.body.classList.add("demo-mode");
  if (socket) socket.close();
  window.clearTimeout(reconnectTimer);
  state = emptySnapshot();
  state.agent = {
    connected: true,
    agent_version: "0.1.0-demo",
    boot_id: "demo-7ad9-4f20-visual",
    transport_mode: "client",
    capabilities_digest: "sha256:demo-local-visualization",
  };
  state.control.reason = "浏览器演示：尚未获取控制权";
  state.robot_state = { "robot.mode": "SAFE_HOLD" };
  state.events = [
    demoEvent("system", "浏览器演示已启动", "此模式模拟协议状态，不调用 ROS 2。", "warning"),
    demoEvent("agent", "Agent → BFF · session.hello", "严格协议握手与 capability digest", "success"),
    demoEvent("bff", "BFF → Agent · session.welcome", "新会话从 SAFE_HOLD 开始", "command"),
  ];
  window.clearInterval(demoState.interval);
  demoState.interval = window.setInterval(() => {
    if (!demoMode || !state.lease.enabled || state.control.state !== "ACTIVE") return;
    state.lease.sequence = demoState.sequence++;
    state.lease.expires_at = new Date(Date.now() + state.lease.timeout_ms).toISOString();
    state.metrics.messages_to_agent += 1;
    state.metrics.messages_from_agent += 1;
    state.generated_at = new Date().toISOString();
    render();
  }, state.lease.interval_ms);
  elements.modeBadge.textContent = "BROWSER DEMO";
  elements.modeToggle.textContent = "返回真实链路";
  elements.uiStatus.classList.add("is-online");
  elements.uiStatus.innerHTML = "<i></i>演示引擎在线";
  render();
  showToast("已进入浏览器演示：页面效果可操作，但不会调用 ROS 2。", false);
}

function leaveDemoMode() {
  demoMode = false;
  autoDemoRunning = false;
  window.clearInterval(demoState.interval);
  document.body.classList.remove("demo-mode");
  elements.modeBadge.textContent = "REAL LINK";
  elements.modeToggle.textContent = "进入浏览器演示";
  state = emptySnapshot();
  render();
  connect();
}

function demoEvent(direction, title, detail, level = "info") {
  return {
    event_id: crypto.randomUUID(),
    at: new Date().toISOString(),
    direction,
    kind: title.toLowerCase().replaceAll(" ", "."),
    title,
    detail,
    level,
  };
}

function pushDemoEvent(event) {
  state.events.unshift(event);
  state.events = state.events.slice(0, 120);
  state.generated_at = new Date().toISOString();
}

function handleDemoAction(action) {
  switch (action.action) {
    case "control.acquire":
      state.control = { state: "ACTIVE", reason: "control acquired by visual-demo-operator" };
      state.lease.enabled = true;
      state.lease.sequence = demoState.sequence++;
      state.lease.expires_at = new Date(Date.now() + state.lease.timeout_ms).toISOString();
      pushDemoEvent(demoEvent("bff", "BFF → Agent · control.acquire", "显式获取控制权", "command"));
      pushDemoEvent(demoEvent("agent", "Agent → BFF · control.status", "ACTIVE · control acquired", "success"));
      state.metrics.messages_to_agent += 1;
      state.metrics.messages_from_agent += 1;
      break;
    case "lease.set":
      state.lease.enabled = Boolean(action.enabled);
      if (action.enabled) {
        state.lease.sequence = demoState.sequence++;
        state.lease.expires_at = new Date(Date.now() + state.lease.timeout_ms).toISOString();
        pushDemoEvent(demoEvent("bff", "续租已恢复", "控制链路重新获得租约", "success"));
      } else {
        pushDemoEvent(demoEvent("bff", "续租已暂停", "2 秒后模拟本地 safe-stop", "warning"));
        const expectedSequence = state.lease.sequence;
        window.setTimeout(() => {
          if (!demoMode || state.lease.enabled || state.lease.sequence !== expectedSequence) return;
          state.control = { state: "SAFE_HOLD", reason: "lease expired; local safe-stop completed" };
          state.robot_state["robot.mode"] = "SAFE_HOLD";
          state.lease.expires_at = null;
          pushDemoEvent(demoEvent("agent", "Agent → BFF · safety.event", "LEASE_EXPIRED · safe-stop latched", "danger"));
          state.metrics.messages_from_agent += 1;
          render();
        }, state.lease.timeout_ms);
      }
      break;
    case "command.send":
      demoCommand(action.kind, action.params);
      return;
    case "command.cancel":
      demoCancel(action.command_id);
      return;
    case "events.clear":
      state.events = [demoEvent("system", "事件记录已清空", "新的协议事件将继续显示")];
      break;
    case "observation.ack_pause":
    case "observation.reject_next":
    case "observation.session_set":
    case "observation.camera_pause":
      showToast("浏览器演示不伪造 RGB-D 协议，请返回真实链路查看。", true);
      return;
    default:
      showToast(`演示模式不支持 ${action.action}`, true);
  }
  render();
}

function demoCommand(kind, params) {
  if (state.control.state !== "ACTIVE") {
    showToast("SAFE_HOLD 下禁止下发命令，请先获取控制权。", true);
    return;
  }
  const commandId = crypto.randomUUID();
  const row = {
    command_id: commandId,
    kind,
    params,
    state: "RUNNING",
    detail: "command dispatched to local adapter",
    progress: kind === "task.execute" || kind === "navigation.navigate_to_pose" ? 0.1 : null,
    result: null,
    issued_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  state.commands.unshift(row);
  state.commands = state.commands.slice(0, 50);
  state.metrics.messages_to_agent += 1;
  state.metrics.messages_from_agent += 1;
  pushDemoEvent(demoEvent("bff", `BFF → Agent · ${kind}`, shortId(commandId, 18), "command"));
  pushDemoEvent(demoEvent("agent", "Agent → BFF · command.status", "RUNNING · local adapter", "command"));
  render();

  if (kind === "task.execute" || kind === "navigation.navigate_to_pose") {
    let step = 1;
    const maxSteps = 5;
    const timer = window.setInterval(() => {
      if (!demoMode || row.state !== "RUNNING") {
        window.clearInterval(timer);
        return;
      }
      step += 1;
      row.progress = Math.min(0.95, step / maxSteps);
        row.detail = kind === "navigation.navigate_to_pose"
          ? `Mock Nav2 feedback · ${(maxSteps - step) * 0.25} m remaining`
          : `action feedback · sequence ${step + 2}`;
      row.updated_at = new Date().toISOString();
      state.metrics.messages_from_agent += 1;
      if (step >= maxSteps) {
        window.clearInterval(timer);
        row.state = "SUCCEEDED";
        row.progress = 1;
        row.detail = "action completed";
        row.result = kind === "navigation.navigate_to_pose"
          ? { executionBackend: "mock_nav2", motion: "simulated" }
          : { sequence: [0, 1, 1, 2, 3, 5, 8] };
        pushDemoEvent(demoEvent("agent", "Agent → BFF · command.status", "SUCCEEDED · action completed", "success"));
      }
      render();
    }, 260);
  } else {
    window.setTimeout(() => {
      if (!demoMode || row.state !== "RUNNING") return;
      row.state = "SUCCEEDED";
      row.detail = params.data ? "power enabled" : "power disabled";
      row.result = { success: true, message: row.detail };
      row.updated_at = new Date().toISOString();
      state.robot_state["robot.mode"] = params.data ? "IDLE" : "SAFE_HOLD";
      state.metrics.messages_from_agent += 2;
      pushDemoEvent(demoEvent("agent", "Agent → BFF · command.status", `SUCCEEDED · ${row.detail}`, "success"));
      render();
    }, 460);
  }
}

function demoCancel(commandId) {
  const row = state.commands.find((command) => command.command_id === commandId);
  if (!row || terminalStates.has(row.state)) {
    showToast("没有可取消的活动命令。", true);
    return;
  }
  row.state = "CANCELLED";
  row.detail = "cancelled from visual test dashboard";
  row.updated_at = new Date().toISOString();
  state.metrics.messages_to_agent += 1;
  state.metrics.messages_from_agent += 1;
  pushDemoEvent(demoEvent("bff", "BFF → Agent · command.cancel", shortId(commandId, 18), "warning"));
  pushDemoEvent(demoEvent("agent", "Agent → BFF · command.status", "CANCELLED · terminal", "warning"));
  render();
}

async function runAutoDemo() {
  if (autoDemoRunning) return;
  autoDemoRunning = true;
  elements.autoDemoBtn.disabled = true;
  if (!demoMode && !state.agent?.connected) enterDemoMode();

  const pause = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  try {
    if (state.control.state !== "ACTIVE") {
      sendAction({ action: "control.acquire" });
      await pause(700);
    }
    sendAction({ action: "command.send", kind: "power.enable", params: { data: true } });
    await pause(1000);
    sendAction({ action: "command.send", kind: "task.execute", params: { order: 12 } });
    await pause(650);
    const activeCommand = latestCancelableCommand();
    if (activeCommand) sendAction({ action: "command.cancel", command_id: activeCommand.command_id });
    await pause(700);
    sendAction({ action: "lease.set", enabled: false });
    showToast("自动演示：已暂停续租，观察 Agent 进入 SAFE_HOLD。", false);
  } finally {
    autoDemoRunning = false;
    elements.autoDemoBtn.disabled = false;
  }
}

elements.acquireBtn.addEventListener("click", () => sendAction({ action: "control.acquire" }));
elements.powerOnBtn.addEventListener("click", () =>
  sendAction({ action: "command.send", kind: "power.enable", params: { data: true } }),
);
elements.powerOffBtn.addEventListener("click", () =>
  sendAction({ action: "command.send", kind: "power.enable", params: { data: false } }),
);
elements.taskBtn.addEventListener("click", () => {
  const order = Number(elements.taskOrder.value);
  if (!Number.isInteger(order) || order < 1 || order > 100) {
    showToast("Action order 必须是 1–100 的整数。", true);
    return;
  }
  sendAction({ action: "command.send", kind: "task.execute", params: { order } });
});
elements.vlnBtn.addEventListener("click", () => {
  const x = Number(elements.navX.value);
  const y = Number(elements.navY.value);
  const yaw = Number(elements.navYaw.value);
  if (![x, y, yaw].every(Number.isFinite) || yaw < -Math.PI || yaw > Math.PI) {
    showToast("请输入有效的 map 坐标，yaw 范围为 -π 到 π。", true);
    return;
  }
  sendAction({
    action: "command.send",
    kind: "navigation.navigate_to_pose",
    params: { goalPose: { frameId: "map", x, y, z: 0, yaw, observedAt: new Date().toISOString() } },
  });
});
elements.cancelBtn.addEventListener("click", () => {
  const command = latestCancelableCommand();
  if (!command) {
    showToast("没有可取消的活动命令。", true);
    return;
  }
  sendAction({ action: "command.cancel", command_id: command.command_id });
});
elements.leaseBtn.addEventListener("click", () =>
  sendAction({ action: "lease.set", enabled: !state.lease?.enabled }),
);
elements.clearEventsBtn.addEventListener("click", () => sendAction({ action: "events.clear" }));
elements.modeToggle.addEventListener("click", () => (demoMode ? leaveDemoMode() : enterDemoMode()));
elements.autoDemoBtn.addEventListener("click", runAutoDemo);
elements.ackFaultBtn.addEventListener("click", () =>
  sendAction({
    action: "observation.ack_pause",
    paused: !Boolean(state.observation?.faults?.acksPaused),
  }),
);
elements.rejectFrameBtn.addEventListener("click", () =>
  sendAction({ action: "observation.reject_next" }),
);
elements.sessionFaultBtn.addEventListener("click", () =>
  sendAction({
    action: "observation.session_set",
    enabled: !Boolean(state.observation?.observationSessionId),
  }),
);
elements.cameraFaultBtn.addEventListener("click", () =>
  sendAction({
    action: "observation.camera_pause",
    paused: !Boolean(state.observation?.faults?.cameraPaused),
  }),
);

window.setInterval(() => {
  elements.footerClock.textContent = new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "medium",
    hour12: false,
  }).format(new Date());
  if (!demoMode && lastSnapshotAt && Date.now() - lastSnapshotAt > 3500) {
    elements.freshness.textContent = "STALE UI DATA";
  }
  renderLease();
}, 200);

render();
connect();

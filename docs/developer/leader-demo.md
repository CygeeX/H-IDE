# 领导演示开发说明

## 1. 验收目标

登录后，会话初始页展示两个可选 Skill：

1. **巡检 Skill (`inspection.office`)**：选择后显示“今天还是巡检开放办公室吗？”。
   用户输入“去会议室里观察”“巡检全部办公室，把看到的信息汇总给我”等中文
   语义时，Control API 将请求交给确定性的 Office V2 地点/巡检编译器，生成
   可审核 Mission；到达、扫描和报告完成后才显示成功。
2. **语音助手 Skill (`assistant.voice`)**：普通问答仍经 Conversation Gateway
   路由，浏览器使用 `SpeechSynthesis` 的 `zh-CN` 声音朗读同一条助手文本；在
   仿真配置启用时，完成回复后向 SONIC 请求一次 `gesture.wave`。

领导录制视频由操作者完成；仓库不内置录制器或自动降级 UI。

## 2. PWA 组件

| 组件 | 位置 | 责任 |
| --- | --- | --- |
| Skill 卡片/模式上下文 | `apps/control-pwa/src/leader-demo.tsx` | 白名单 Skill、欢迎语、模式提示 |
| 助手头像 | `leader-demo.tsx` 的 `RobotAvatar` | 纯 CSS 光环/面板动效、DOM 对话气泡；不加载 GLB |
| 机器人选择器 | `leader-demo.tsx` 的 `RobotPicker` | 三家公司展示卡；只有 Unitree 保持实际控制绑定 |
| 会话接入 | `apps/control-pwa/src/main.tsx` | Skill 状态、`skillId` advisory 字段、语音朗读与挥手触发 |
| 主题 | `apps/control-pwa/src/chat-shell.css` | 登录页/会话页 CSS 头像、卡片和气泡样式 |

### 2.1 资源

资源位于 `apps/control-pwa/public/assets/leader-demo/`。品牌和机器人选择卡仍
使用本地 PNG；助手头像由 CSS 绘制，不再依赖 GLB、Three.js、GLTFLoader 或
WebGL 初始化。

## 3. Skill 请求契约

浏览器向 `/api/v1/assistant/conversations/{id}/messages` 发送：

```json
{
  "text": "去会议室里观察",
  "modality": "text",
  "skillId": "inspection.office"
}
```

`skillId` 只允许 `inspection.office` 或 `assistant.voice`。它在
`packages/control_core/assistant_models.py`、`services/control_api/assistant_service.py`
和 `h-ide-agent/src/conversation/contracts.ts` 中均为 advisory 字段；旧 Agent
返回“不认识 skillId”时，Control API 客户端会剥离该字段重试一次。任何物理
目标仍必须来自场景目录、RGB-D 证据和 Mission 授权边界。

## 4. 语音与挥手链路

### 4.1 语音

`assistant.voice` 回复使用浏览器 `SpeechSynthesisUtterance`（语言 `zh-CN`），
不新增云端 TTS 依赖。已有的流式 TTS 开关仍保留给普通开发者模式；两条路径
不会同时播放。

### 4.2 仿真挥手

启用本地演示环境变量：

```bash
HIDE_LEADER_DEMO_GESTURES_ENABLED=true
HIDE_LEADER_DEMO_GESTURE_URL=http://host.docker.internal:8798
HIDE_LEADER_DEMO_GESTURE_PROXY_TOKEN=dev-leader-demo-gesture-token
```

Control API 的 `POST /api/v1/leader-demo/gestures/wave` 只接受已绑定的仿真
机器人 ID，并通过主机侧的 allow-listed gesture proxy 转发。Control API
容器不会接触 SONIC 原始 command socket；proxy 才会在主机 loopback 上发送
下面的固定 JSON-lines 帧：

```json
{
  "type": "gesture_wave",
  "gesture": "wave",
  "duration_s": 2.4,
  "simulation_only": true
}
```

SONIC 内部的 `GestureWaveController` 在唯一的 50 Hz `LowCmd` 发布边界叠加小幅
右臂轨迹；它不允许行走/停止边界期间执行，不创建第二个 DDS 发布者，也不直接
改写 MuJoCo `qpos`。请求、状态和进度进入 SONIC telemetry。真机 Unitree 动作
ID（如 high wave=26）尚未接入该接口，不能把本仿真挥手宣称为真机能力。

## 5. Office V2 场景变更

唯一源文件：
`/home/phi/learn-projects/h-ide-mujoco-sonic/scenes/office_v2/scene.yaml`。

本演示版本：

- 删除走廊中央圆柱及其 Nav2 障碍；中央通道可用一段连续路径通过；
- 删除可见吊灯面板；保留三个不可见的 studio 点光源，仅用于照明；
- 地面改为 MuJoCo checker 材质，保留原有碰撞平面；
- 增加公告板、白板、纸箱等可识别物件；
- 增加三个 `display_robot_*` 视觉装饰，放在房间角落的地面锚点。它们复用运动
  G1 的官方 STL 视觉网格，并保留完整的 link 层级、以固定的 nominal standing
  姿态展开；生成器会测量脚底偏移，确保脚底贴合 MuJoCo 网格。全部为
  `contype=0`、`conaffinity=0`，没有关节/执行器，不进入 G1/SONIC 控制或 Nav2 地图。

重新生成：

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
python tools/build_scene.py
python tools/build_scene.py --check
```

生成器会同步更新 MuJoCo XML、Nav2 PGM/YAML、巡检目录、runtime/visual manifest
和 SHA256 清单。H-IDE 的内置产品包也必须同步：

```text
services/scene_api/builtin/office_v2/{office_v2.pgm,office_v2.yaml,inspection_catalog.json}
```

## 6. 本地验证

```bash
# PWA
cd /home/phi/learn-projects/h-ide/apps/control-pwa
npm test -- --run
npm run build

# Agent
cd /home/phi/learn-projects/h-ide-agent
npm test
npm run typecheck

# MuJoCo/SONIC
cd /home/phi/learn-projects/h-ide-mujoco-sonic
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_gesture.py tests/test_office_v2_scene.py
```

启动完整仿真仍使用既有工作站脚本（录制前先完成其 readiness/preflight）：

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
HIDE_LEADER_DEMO_GESTURES_ENABLED=true \
  scripts/start-workstation-demo.sh --viewer --full-straight-drive
```

当前已运行的旧 Docker image 不会自动加载未提交源代码；修改后必须重建相关
镜像并冷启动，不能用旧窗口判断新版本是否生效。

## 7. 故障定位

- 页面头像异常：检查 `leader-demo.tsx` 与 `chat-shell.css` 的 CSS 动效；当前头像
  不依赖 WebGL、GLB 或模型静态托管。
- 语音正常但没有挥手：检查 Control API 容器的三个 gesture 环境变量、
  `host.docker.internal:8798/healthz` 连通性，以及 SONIC telemetry 的
  `gesture_state`。
  这不会使语音回复失败。
- 办公室目标被解析为入口：检查内置 `inspection_catalog.json` 是否与
  MuJoCo 生成目录的 `scene_source_sha256`/PGM hash 一致；不要在 Agent 中手写
  坐标。
- 场景地图与窗口不一致：确认已重建 Nav2 image，并按依赖逆序停止旧 Plant、
  SONIC、Robot Agent、Nav2 后重新启动。

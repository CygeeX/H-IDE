# 组件开发者索引

本文描述 `skillstore-office-v2-e2e-20260901-r1` 候选中实际存在的代码边界。产品目标
架构中的 Mission Architect、Field Executive、Cognition Broker 和 Semantic Scene
Compiler 另列为 Proposed，不能用未来名称掩盖当前调用链。

## Current：`components/h-ide`

H-IDE 当前承载 PWA、认证、Conversation/Mission V2、Scene API、Robot Agent BFF、
报告和开发者诊断。

关键入口：

- `apps/control-pwa/src/leader-demo.tsx`：双 Skill、CSS 助手头像和机器人选择模态框。
- `apps/control-pwa/src/main.tsx`：会话历史、Mission 卡片归属、审核、流式进度、语音和
  仿真挥手请求。
- `services/control_api/conversation_gateway.py`：普通问答、巡检历史问答和规划路由。
- `services/control_api/office_inspection.py`：Office V2 巡检语义与目录绑定。
- `services/control_api/mission_service.py`：Mission DAG、执行状态、重试与终态对账。
- `services/control_api/conversation_memory.py`：会话历史和证据绑定语义指代。
- `services/scene_api/`：当前 Office V2 Scene/catalog 发布和场景绑定。
- `docs/developer/leader-demo.md`：当前双 Skill 演示合同。
- `docs/platform-architecture/`：商业化与 PRO 6000 Demo 的 Proposed 架构。

当前 `SemanticReferent` 可以记住同一会话或 Mission 中看到了什么、证据和区域，但不
保存可直接执行的对象姿态。后续“走到刚才的架子前面”仍需用新鲜 RGB-D/TF 重新生成
短 TTL 接近点。

PWA 最低门禁：

```bash
ROOT="$(pwd -P)"
(cd "$ROOT/components/h-ide/apps/control-pwa" && npm test && npm run build)
```

## Current：`components/h-ide-agent`

独立 Node/Mastra 服务负责 Conversation V2 路由、自然语言意图、Supervisor/Worker
编排、Mission 计划提案和修订上下文。它不拥有 ROS、Robot Agent 或 SONIC 凭据。

关键入口：

- `src/conversation/router.ts`、`memory.ts`：普通问答、Skill 上下文与多轮记忆。
- `src/mission/engine.ts`：Mission V2 proposal/revision 生成。
- `src/mission/contracts.ts`、`src/contracts.ts`：跨服务 schema。
- `src/http/routes.ts`：内部 planning/conversation API。

`skillId` 是受白名单约束的 advisory context；它不能绕过 H-IDE 场景绑定、策略和审核。

```bash
ROOT="$(pwd -P)"
(cd "$ROOT/components/h-ide-agent" && npm test && npm run typecheck && npm run build)
```

## Current：`components/h-ide-vision`

独立 Python 3.11 GPU 服务负责 Qwen 语义、SAM2 分割、RGB-D/TF 几何候选和 Evidence。
在当前仿真中，它处理 MuJoCo 第一视角观测；Qwen 是语义 authority，SAM2 只提供分割
和几何支持。

关键入口：

- `src/h_ide_vision/runtime/`：Visual Task/Resolution 执行。
- `src/h_ide_vision/providers/`：VLM 与 segmentation provider boundary。
- `src/h_ide_vision/geometry/`：深度去投影、mask depth 和接近点。
- `src/h_ide_vision/ingress/`：Robot Observation 协议、新鲜度和身份校验。

Vision 不做路径规划或速度控制。模型候选如果缺少 capture-time TF、深度、Boot/Epoch、
置信度或新鲜度，不能进入运动 Mission。

## Current：`components/robotagent`

Robot Agent 是受限机器人协议与 ROS 出口：能力 manifest、控制租约、幂等命令、状态回执、
取消、安全停机和 RGB-D 观测网关。

关键入口：

- `src/robot_agent/robot_agent/runtime.py`、`session.py`：驻留执行 authority 和租约生命周期。
- `src/robot_agent/config/robot_agent.mujoco.yaml`：当前仿真能力映射。
- `src/robot_observation_gateway/`：同步 RGB-D/TF/定位观测与 freshness gate。
- `tools/activate-local-*-jazzy.bash`：当前 Jazzy overlay 选择。

当前修订保持 SONIC 驻留控制 authority，不允许普通服务在自动故障路径中终止平衡策略；
停止新的任务、取消和本体安全状态仍必须真实上报，不能伪造 Mission 成功。

## Current：`components/h-ide-mujoco-sonic`

本组件包含 Office V2 单源场景、MuJoCo Plant、SONIC 50 Hz driver、ROS bridge、Nav2
static planner/strict adapter、仿真挥手代理和只读诊断。

关键入口：

- `scenes/office_v2/scene.yaml`：场景唯一源；生成 MuJoCo、Nav2、巡检目录和 manifest。
- `tools/build_scene.py`：场景构建与哈希一致性检查。
- `scripts/start-workstation-demo.sh`：有序启动、模型锁、ready gate 和可选 profile。
- `tools/strict_navigation_adapter.py`：Nav2 结果到已审核底盘原语的边界。
- `src/hide_mujoco_sonic/sonic_driver.py`：唯一 SONIC LowCmd 发布路径。
- `src/hide_mujoco_sonic/gesture.py`、`leader_demo_gesture_proxy.py`：仅仿真挥手。
- `docs/continuous-drive-handoff.md`：continuous/full-straight profile 合同。

`--full-straight-drive` 只对地图证明安全的直线段放宽单次距离，当前上限 4 m，并保持
SONIC 官方 0.40 m/s 前进速度；转向、终点、取消、故障和不确定区域仍按正常边界处理。

当前 Office V2 没有走廊中央圆柱，不应继续使用“绕中央柱子”作为验收用例。

## Current：`third_party` 与 `optional/DAAAM`

- `third_party/lerobot`、`third_party/unitree_sdk2_python` 是有 provenance 的源码快照。
- G1 网格、GR00T/SONIC checkout 和模型权重由 `integration/bootstrap.sh` 按 manifest
  外置获取，不属于普通源码快照。
- `optional/DAAAM` 只记录上游 revision 与 H-IDE 异步 importer 边界。默认启动链不运行
  DAAAM；它不能生成实时定位、障碍事实或可执行目标。

更新第三方内容必须同步更新 vendor manifest、内容哈希、许可证记录、integration lock、
clean-clone 和对应回归。

## Proposed：平台演进组件

以下是已经完成设计决策、尚未成为当前运行组件的产品角色：

| 产品角色 | 目标职责 | 当前替代/缺口 |
| --- | --- | --- |
| H-IDE Mission Architect | 云端高层 Mission 编写与修订 | 当前由 H-IDE + h-ide-agent 同机完成 |
| H-IDE Field Executive | 边缘唯一 Mission 执行 authority、断云自治 | 当前执行 runtime 仍在工作站 H-IDE 内 |
| H-IDE Cognition Broker | Local/Site/Cloud 模型策略路由 | 当前没有独立 Broker |
| H-IDE Live Visual Resolver | 当前 RGB-D/TF 目标解析与到达证据 | 由 h-ide-vision 演进，但未按新部署边界拆分 |
| Semantic Scene Compiler | 几何与语义候选生成 Scene proposal | 当前 Office V2 catalog/可信坐标仍是固定回退基线 |
| Scene Publisher/Registry | 审核并签名发布不可变 SceneRevision | 当前 Scene API 尚未实现完整新契约 |
| FARM/DAAAM Provider | Shadow memory benchmark 与候选输出 | FARM 未集成；DAAAM 仅有异步 importer |

目标协议、里程碑和第二场景无硬编码验收见
[`components/h-ide/docs/platform-architecture/README.md`](../../components/h-ide/docs/platform-architecture/README.md)。

# H-IDE 联调适配

## 改造范围

这次工作是现有 Robot Agent 的“协议对齐 + NavigateToPose 适配”，不是重写控制代理。
控制 WebSocket client、2 秒租约、safe-stop、状态优先级、SQLite 幂等账本和独立 RGB-D
sidecar 都保留。H-IDE 与 H-IDE Vision 仓库只作为只读协议源和联调端，本次没有修改
它们的代码。

### 2026-08-17 Mission V2 位姿状态 handoff

为满足 H-IDE 首段导航的实时起点准入，当前 local/production profile 通过既有
`state.snapshot/state.update` 增加了 `/localization/pose`
(`geometry_msgs/msg/PoseWithCovarianceStamped`) 的 `map` frame、ROS stamp、位置和四元数
字段。该改动只涉及配置与 `geometry_msgs` 运行时依赖声明，不改变命令适配器、Nav2
执行、watchdog、租约或安全逻辑。H-IDE Gateway 将 ROS header stamp 作为 provenance，
以定位字段变化对应的 Robot Agent UTC 状态时间做跨时钟的新鲜度判断；请在真实联调前
由 Robot Agent owner 重新运行 `validate-config` 并复核 capability digest。

## Observation v1

唯一标准来自 H-IDE Vision：

- `contracts/robot_observation/v1/observation.proto`
- `contracts/robot_observation/PROTOCOL.lock`

Robot Agent 工作区中的逐字节副本位于：

- `proto/robot_observation/v1/observation.proto`
- `proto/robot_observation/PROTOCOL.lock`

Python、类型提示和 gRPC stub 位于
`src/robot_observation_gateway/robot_observation/v1/`。需要重新生成时执行：

```bash
PYTHONPATH=src/robot_observation_gateway .venv/bin/python -m grpc_tools.protoc \
  -I proto \
  --python_out=src/robot_observation_gateway \
  --pyi_out=src/robot_observation_gateway \
  --grpc_python_out=src/robot_observation_gateway \
  proto/robot_observation/v1/observation.proto
```

Hello 固定使用 `1.0` 和 lock 中的
`schema_sha256=d5055d5dbd81f2e098209bf9cba9db5739b74a3be62dae172958bd536648bb14`。
这个值是 H-IDE Vision 锁定的 `descriptor.pb` 摘要，不是 `.proto` 文本摘要。D435i
入口声明 `SOURCE_MODE_LIVE`，当前 source ID 为
`realsense:d435i:261222077185`；session/frame 同时携带 localization epoch，frame 还
携带 clock uncertainty。H-IDE Vision 未指定 scene 时，session/frame 不设置
`scene_version`。

本地开发配置默认主动连接 H-IDE Vision 的 `127.0.0.1:50061`。生产仍使用机器人主动
建立的 mTLS 双向 gRPC，RGB-D 不进入控制 WebSocket。

## `navigation.navigate_to_pose`

请求 Schema 的逐字节副本是
`src/robot_agent/schemas/navigation.navigate_to_pose.schema.json`。执行步骤如下：

1. 严格校验 H-IDE 请求，拒绝未知字段、非有限数、非 `map` frame 和过期命令。
2. 校验 `robotBootId`、`observationSessionId`、`localizationEpochId` 与最近被 H-IDE
   Vision ACK 的观测一致；`sceneRef` 可为空。
3. 把 `goalPose` 转成真实 `nav2_msgs/action/NavigateToPose` 的 `PoseStamped`，使用本机
   ROS clock 作为 Goal stamp，`behavior_tree` 为空。
4. 只向 Nav2 提交这个 map 目标。Robot Agent 不生成路径，也不发布速度。
5. `missionId`、`stageId`、`visualResolutionId`、观测绑定和限速等字段只写日志与命令
   账本，不进入 ROS Goal。
6. H-IDE 持续获得 `RUNNING`/feedback 和最终状态；重复 command ID 返回账本状态，
   不二次发送 Goal。

H-IDE Mission V2 内部可能使用可读的 `skill-...` 请求 ID。工作站 Gateway 在
`robot-agent.v1` 传输边界将该命名空间稳定转换为 UUID5；Robot Agent 看到的
`command_id` 始终是 UUID，重试、查询和取消使用同一 UUID。回执中的
`robotAgentCommandId` 是该线协议 ID，不能把它误当成新的第二个任务。

`tools/mock_robot.py` 提供真实 `NavigateToPose` Action 类型的 Mock Nav2。它只记录 Goal、
发送五次模拟 feedback 并返回成功，不发布 `/cmd_vel`。验收必须使用隔离 ROS Domain，
并确认 action graph 中只有 Robot Agent client 与该 Mock server。

## 安全绑定

- admission：boot/session/epoch 不匹配、观测 ACK 超过 500 ms 或相机采集过旧时拒绝。
- watchdog：运行中 boot/session/epoch 改变或 750 ms 没有新 accepted sequence 时，
  只取消标记为 `requires_observation` 的导航 Goal。
- cancellation：Nav2 未确认取消时进入 `FAILED / CANCEL_NOT_ACKNOWLEDGED` 并 safe-stop。
- PCD：`sceneRef`/`scene_version` 可用于额外审计，但不是必填执行条件。
- transport：Robot Agent 继续主动连接工作站；URI、robot ID 和 TLS 均来自配置，当前
  robot ID 为 `g1-dev`。

## 2026-08-12 本地端到端验收

验收使用 `ROS_DOMAIN_ID=231`，连接真 D435i 和 Mock Nav2，未连接真机导航代理：

- D435i 序列号 `261222077185`，USB 3.2 / 5,000 Mbit/s；官方 RealSense ROS 输出同步、
  对齐的 640×480 RGB-D。
- H-IDE Vision 接受真机 `SOURCE_MODE_LIVE` 帧；health 达到 `USABLE`，scene 为空，
  boot/session/epoch 完整。
- H-IDE Vision 基于真 RGB-D（语义框使用其 deterministic fake provider）生成可执行
  `map` 目标：`x=1.8926623279`、`y=0.8798193280`、`z=0`、
  `yaw=0.4647834434`。这验证传图、深度、TF、定位和目标坐标链路，不宣称已验收生产
  VLM 的识别质量。
- H-IDE Gateway 经现有 WebSocket client 收到 `RECEIVED → RUNNING/feedback →
  SUCCEEDED`；最终 `rosGoalId=f74dfcc3-6203-43fa-901b-b88892a9dc5b`。
- 重发同一个 `command_id=c40b6386-c33a-462a-8f91-8654bc2ec6b7` 返回同一终态和同一
  `rosGoalId`，Mock Nav2 只收到一次 Goal。
- ROS graph 中没有 `/cmd_vel` topic，`/navigate_to_pose` 只有 Robot Agent client 与
  `robot_agent_mock_robot` server；随后停止全部联调进程，RealSense 干净关闭且 Robot
  Agent 再次触发 safe-stop。
- 自动化回归：Jazzy/Python 3.12 为 213 passed、1 个 30 分钟 soak deselected，分支
  覆盖率 90.54%；Humble/Python 3.10（真实 `nav2_msgs`）为 211 passed、3 skipped，
  跳过项仅为容器未挂载两个 H-IDE Vision 只读联调测试及默认不运行 30 分钟 soak。
  Ruff、strict mypy、JavaScript 语法、配置校验、colcon 和 `git diff --check` 通过。

PyTorch3D 不在 Observation Protobuf、RGB-D 编解码、TF/定位绑定、WebSocket 控制协议或
NavigateToPose 适配的依赖路径上；缺少它不会阻塞这些改造。将来若 H-IDE Vision 的某个
生产 3D 算法选择 PyTorch3D，应由感知项目单独管理该可选运行时依赖。

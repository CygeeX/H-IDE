# 架构、调用链与信任边界

| 属性 | 值 |
| --- | --- |
| 发布基线 | `skillstore-office-v2-e2e-20260901-r1` |
| 当前已实现 | Office V2 上的 PWA → Mission V2 → Robot Agent → Nav2 → SONIC → MuJoCo 巡检与证据报告 |
| 当前未实现 | Field Executive、Cognition Broker、FARM/DAAAM Shadow Provider、Canonical World Model、Semantic Scene Compiler |
| 详细设计 | [H-IDE 云—站点—边缘—机器人平台架构](../../components/h-ide/docs/platform-architecture/README.md) |

本文只描述这个发行版真实存在的能力。目标架构已经完成职责设计，但不能因此把
Proposed 组件写成 Current 功能。

## 1. 当前真实调用链

### 1.1 已知 Office V2 巡检

```text
PWA
  → H-IDE Control API / Conversation Gateway
  → Office V2 巡检意图确定性匹配
  → Scene API 已发布 SceneRef + Mission 局部 WorldRecord
  → Office Python Compiler
       navigation → inside-region verification → multi-heading scan → report
  → PWA 人工审核
  → Control API 内的 MissionRuntime / 当前 workstation 执行权威
  → Robot Agent Gateway
  → g1-sim Robot Agent
  → Nav2 NavigateToPose / Spin / DriveOnHeading
  → strict navigation adapter
  → cmd_vel bridge → SONIC 50 Hz LowCmd → MuJoCo G1
  → RGB-D + capture-time TF + localization epoch
  → Observation Gateway → H-IDE Vision
  → Qwen/SAM2/几何 Evidence → Mission report → PWA
```

“今天巡检开放办公室和会议室”这类已知场景任务，优先由
`components/h-ide/services/control_api/office_semantic_navigation.py` 编译完整 DAG，
不会让 LLM 逐步决定坐标或路线。`components/h-ide-agent` 仍负责会话分类、未命中
确定性编译器时的高层语义 Mission 草案以及受约束的重规划；它没有 Robot Agent、
ROS、Nav2 或 SONIC 执行工具。

### 1.2 普通观察、对象接近和开放式行动

- 只观察当前画面时，Control API 创建只读视觉请求，不生成底盘运动。
- “走到这个架子前面”先由 Qwen 提出对象语义和像素候选，再由 SAM2、同步深度与
  capture-time TF 产生短 TTL 的几何接近候选；历史 centroid 不能直接进入 Nav2。
- 未被 Office 确定性编译器覆盖的行动可由 Mission Architect 生成高层语义 DAG，
  但 `h-ide.mission/2` 合同拒绝模型提供 Pose、Waypoint、速度、`cmd_vel` 和 ROS 字段。

## 2. 当前组件职责

| 层 | 当前组件 | 当前责任 | 明确不能做的事 |
| --- | --- | --- | --- |
| 交互/控制面 | `components/h-ide` PWA、Control API | 会话、审核、Mission ledger、Scene binding、当前执行推进 | 让浏览器或模型直接控制 ROS/关节 |
| Mission 规划 | `components/h-ide-agent` | 会话分类、高层语义 DAG、重规划建议 | 输出可执行坐标或访问机器人命令端口 |
| 实时视觉 | `components/h-ide-vision` | Qwen 语义、SAM2 Mask、RGB-D/TF 几何、Evidence | 路径规划、长期世界事实发布、运动授权 |
| 机器人协议 | `components/robotagent` | 能力清单、租约、幂等命令账本、精确取消、ROS Action 映射 | 接受未授权或未知命令 |
| 导航 | Nav2 + strict adapter | 地图路径、旋转/直行原语、实时执行状态 | 代替 Mission 审核或 SONIC 全身控制 |
| 运控/物理 | `components/h-ide-mujoco-sonic` | SONIC、DDS、MuJoCo Plant、桥和遥测 | 读取自然语言或让 LLM 写 LowCmd |
| 异步记忆 | DAAAM importer | Mission 后、区域级、`observation_only` 语义先验 | 从零建图、发布 Anchor、实时控制 |

## 3. Current / Proposed 矩阵

| 能力 | Current（本发行版） | Proposed（后续路线） |
| --- | --- | --- |
| 云端高层规划 | `h-ide-agent` 作为 planning-only Agent | 产品角色固定为 `H-IDE Mission Architect` |
| 物理 Mission 执行权 | Control API/MissionRuntime 与 workstation composite executor | 独立 `H-IDE Field Executive`，成为唯一上游执行权威 |
| 模型升级调用 | Control API/Agent/Vision 的固定调用关系 | Field 侧 `H-IDE Cognition Broker` 按 Local → Site → Cloud 升级；模型仍无运动权 |
| 世界状态 | Mission 局部 `WorldRecord`、Scene v1 annotation | Provider-neutral Canonical World Model + `WorldDelta` |
| 场景发布 | 人工 annotation，`draft → published` | Compiler proposal，`draft → proposed → reviewed → published` |
| 新场地记忆 | 不支持从零自动建立 | FARM/DAAAM Shadow → Admission → Semantic Scene Compiler |
| DAAAM | post-Mission importer，依赖已有 Scene/region | 独立 v2 Shadow Provider，不依赖 Office Polygon |
| FARM | 运行集成为 0 | 独立 Shadow Provider；原生 pickle 不作为产品协议 |
| Scene Explorer | 只有设计文档 | PLY/3DGS 异步质量审计与标注候选 Worker |
| 云边部署 | 当前工作站进程/Compose 耦合部署 | PRO 6000 上以 cloud/site/edge 独立网络、身份和存储模拟正式边界 |
| 断云自治 | 当前未形成正式产品合同 | 签名 `MissionBundle`、Edge ledger/event spool、重连不重放物理动作 |

Field Executive 应当具有调用云端模型的能力，但调用必须经过 Cognition Broker。
云模型返回带证据的候选或推理结果；Field Executive 再执行本地 Schema、授权、
Scene、几何、Localization Epoch 和实时健康校验。模型调用能力不等于运动权限。

## 4. Office V2 硬编码和执行栈限制

当前可信坐标的源头是
`components/h-ide-mujoco-sonic/scenes/office_v2/scene.yaml`：其中包含房间 Polygon、
入口、站点 Pose、扫描方向和固定全场路线。`tools/build_scene.py` 生成 MuJoCo、Nav2
地图和 `inspection_catalog.json`，启动器再将产品包交给
`components/h-ide/services/scene_api/office_v2_bootstrap.py` 发布。

这条链可审计且比让 LLM 猜坐标可靠，但仍是场地硬编码。换场地时，开发者必须改
场景源并重新生成产品包，因而不能作为真机商业化 onboarding 方案。

执行栈也绑定 Office V2：

- `src/hide_mujoco_sonic/scene_runtime.py` 校验固定 scene ID；
- `docker/nav2/static_primitive_planner.py` 校验 Office runtime manifest；
- `docker/nav2/nav2_sim_launch.py` 与 `nav2_params.yaml` 固定 Office map；
- `tools/strict_motion_runner.py`、`verify_office_v2_bundle.py` 和启动器固定 Office 路径。

因此，仅把坐标从 Python 移到数据库不算解决。必须同时实现记忆/几何融合、可审核
SceneRevision、通用 map artifact 加载、SceneRef/map digest/runtime digest/
Localization Epoch 一致绑定及 Last-Known-Good 回退。

## 5. 不变的执行边界

- Nav2 始终是路径和底盘导航 owner；SONIC 是全身运控，不接收地图目标。
- Qwen、SAM2、FARM、DAAAM 和 Scene Explorer 都不能直接发布运动命令。
- 每个物理 Mission 仍需可审核草案、授权、幂等 request、终态回执和证据。
- 仿真中的宽松 Safety Profile 不得直接继承到真机。
- 停止、取消、急停和受支持的返航不能被商业授权锁定。

## 6. 继续阅读

- [PRO 6000 Demo 架构迁移路线](platform-roadmap.md)
- [完整平台架构文档索引](../../components/h-ide/docs/platform-architecture/README.md)
- [当前现状与缺口](../../components/h-ide/docs/platform-architecture/demo/01-current-state-and-gap.md)
- [Semantic Scene Compiler](../../components/h-ide/docs/platform-architecture/demo/03-semantic-scene-compiler.md)
- [详细实施与验收](../../components/h-ide/docs/platform-architecture/demo/04-implementation-roadmap-and-acceptance.md)
- [Scene Explorer 设计](scene-explorer-agent.md)

## 7. 参考的工程思想

- NVIDIA GEAR-SONIC/GR00T：50 Hz whole-body policy、模型和 observation contract 锁定；
- Nav2：生命周期、规划/控制分层和 Action 语义；
- ROS 2/CycloneDDS：仿真、真机和 Unitree DDS domain 隔离；
- DAAAM/FARM：posed RGB-D 上的时空/对象记忆候选；
- Isaac Mission Dispatch：完整 Mission 下发、机器人侧执行客户端和异步状态回传；
- Sedna/Greengrass/IoT Edge：边缘先处理、困难样本升级、离线缓存与 store-and-forward。

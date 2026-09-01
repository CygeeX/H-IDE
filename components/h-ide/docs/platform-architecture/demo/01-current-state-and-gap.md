# 当前 Demo：项目边界、真实调用链与关键缺口

| 属性 | 值 |
| --- | --- |
| 状态 | Current-state audit / 2026-08-31 |
| 当前已证明 | 已知 Office V2 地图上的 PWA → Mission → Robot Agent → Nav2 → SONIC → MuJoCo 巡检 |
| 当前未证明 | 新场地从观测自动产生语义地图、可信 Anchor 和可执行巡检 |
| 第一优先缺口 | Provider-neutral World Model + Semantic Scene Compiler |

## 1. 一句话判断

当前系统已经打通“**已知地图上的可信巡检执行链**”，但没有打通“**新场地从观测
自动形成可审核语义地图**”。继续为 Office V2 添加坐标只能改善演示，不能解决真机和
换场地问题。

## 2. 本地项目与当前职责

现有 [`infra/release-bom.yaml`](../../../infra/release-bom.yaml)只是待升级的旧开发基线：
它不是当前工作树 HEAD，且尚未纳入 DAAAM、FARM、skillstore、当前分支、镜像/模型 digest
和完整 Provider 依赖。新 Release Manifest 必须用仓库逻辑 ID、remote、commit 和 digest
标识版本；本地 checkout 绝对路径只能是部署参数，不能成为产品身份。

| 项目 | 当前职责 | 目标产品角色 |
| --- | --- | --- |
| `h-ide` | PWA、Control API、MissionRuntime、Scene API/Worker、审核、策略和 Robot Agent Gateway | Cloud Control Plane；部分执行职责将迁到 Field Executive |
| `h-ide-agent` | Mastra 对话分类、语义 Mission 提案与重规划；没有 ROS/机器人执行工具 | `H-IDE Mission Architect` |
| `h-ide-vision` | Qwen 语义、SAM2 Mask、同步 RGB-D/TF 几何、临时接近点和视觉证据 | `H-IDE Live Visual Resolver` |
| `robotagent` | 机器人侧受限协议、能力目录、租约、幂等台账、ROS Action/Service 映射 | Robot Runtime Gateway |
| `h-ide-mujoco-sonic` | MuJoCo G1、Nav2、SONIC、ROS/RGB-D/定位桥、严格导航适配器 | 仿真 Robot Runtime |
| `DAAAM` | 上游源码存在；H-IDE 只支持任务后、离线、observation-only 导入 | Shadow Memory Provider 候选 |
| `FARM-Project` | 上游源码存在；产品代码和启动链中当前引用为 0 | Shadow Memory Provider 候选 |
| `skillstore` | 只有 Scene Explorer Proposed 设计文档 | 未来异步 Scene Explorer Worker |

目标角色是产品职责，不要求本轮立即重命名 Git 仓库。迁移期间文档和 Release Manifest
同时记录“仓库名”和“产品角色”，避免部署团队把 `h-ide-agent` 与 Field Executive
混为同一个程序。

## 3. 当前稳定巡检的真实调用链

```text
PWA
 → Control API / Conversation Gateway
 → 已知巡检意图确定性匹配
 → MissionService 构造 Scene WorldRecords
 → Office Python Compiler 生成导航/验证/扫描/报告 DAG
 → 用户审核
 → MissionRuntime
 → Robot Agent Gateway / Robot Agent
 → strict_navigation_adapter
 → Nav2 ComputePathToPose + Spin/DriveOnHeading
 → cmd_vel bridge
 → SONIC MovementState / 29-DoF LowCmd
 → MuJoCo
 → RGB-D + map pose
 → Observation Gateway / h-ide-vision
 → Evidence + 最终报告
```

当前已知房间巡检主要由确定性编译器编排，并不是 LLM 在逐步决定路线：

- [`conversation_gateway.py`](../../../services/control_api/conversation_gateway.py)会识别窄域巡检意图；
- [`mission_service.py`](../../../services/control_api/mission_service.py)依次尝试多房间巡检、
  地点和对象等 Python compiler，只有未匹配时才调用 `h-ide-agent`；
- [`office_semantic_navigation.py`](../../../services/control_api/office_semantic_navigation.py)
  生成房间入口、内部导航、区域验证、扫描和报告节点；
- [`packages/mission_core/models.py`](../../../packages/mission_core/models.py)拒绝模型生成的
  Pose、路径、Waypoint、速度和 ROS 字段。

这种边界保护了执行确定性，但因为语义场景本身是人工固定的，LLM 和长期记忆尚未真正
解决新环境的目标发现问题。

## 4. 可信点位硬编码链

最根本的人工数据源不在某个 Prompt，而在 MuJoCo 场景清单：

```text
h-ide-mujoco-sonic/scenes/office_v2/scene.yaml
  ├─ room polygon
  ├─ entry_pose
  ├─ station x/y/yaw
  ├─ scan headings
  └─ fixed inspection route
             ↓ build_scene.py
generated/inspection_catalog.json
             ↓ 启动时一致性检查/复制
h-ide/services/scene_api/builtin/office_v2/inspection_catalog.json
             ↓ office_v2_bootstrap.py
published Scene annotations / robot Scene binding
             ↓ Office compiler
Mission navigation targets
```

当前固定坐标可以直接在
[`inspection_catalog.json`](../../../services/scene_api/builtin/office_v2/inspection_catalog.json)
看到。例如开放办公区和会议室包含人工 `polygonXY`、`entryPose`、内部站点 Pose 与固定
扫描方向。

[`office_v2_bootstrap.py`](../../../services/scene_api/office_v2_bootstrap.py)把每个站点包装
为 Scene annotation，并以坐标为中心生成约 0.2m 小 Polygon；发布前会做 occupancy
`observed_free` 类检查。这比把坐标写进 LLM Prompt 可靠，但换场地仍需要开发者编辑
场景源并重新生成产品包，因此从商业产品角度仍属于硬编码场地知识。

中文别名也有 Office V2 特定逻辑，例如“办公室”映射到 `open_office`。目标架构应把
名称、别名和实体身份放进版本化 SceneRevision，而不是继续扩大 Python 常量表。

### 4.1 执行栈也固定在 Office V2

即使立即替换上面的语义 catalog，当前第二场景仍不能运行，因为仿真执行栈也绑定了
Office V2：

- `h-ide-mujoco-sonic/src/hide_mujoco_sonic/scene_runtime.py` 拒绝非 `office_v2` 并返回
  固定 scene ID；
- `docker/nav2/static_primitive_planner.py` 要求 runtime manifest 的 scene 为 Office V2；
- `docker/nav2/nav2_params.yaml` 固定加载 `office_v2.yaml`；
- `tools/strict_motion_runner.py` 固定 Office map/runtime manifest 路径；
- `scripts/start-workstation-demo.sh` 固定构建、复制并 bootstrap Office V2。

因此迁移还必须实现：SceneRevision map artifact → Nav2 map 加载/受控切换；任意
`sceneId + mapDigest + runtimeGeometryDigest` 绑定；Robot Agent SceneRef、Localization
Epoch、Field Executive Cache 与 Nav2 map 的一致性；以及 Last-Known-Good map 回退。

## 5. 当前 Scene 系统的能力边界

当前 Scene API/Worker 能够：

- 上传 PCD；
- 用 Open3D/PCL/PDAL 做地面估计、降采样和占据/高度栅格；
- 由人完成 scene→map 标定；
- 由人在 PWA 画区域、选择入口并命名；
- 审核发布不可变 revision 并绑定机器人。

当前不能：

- 自动发现房间、门、走廊和连通关系；
- 自动生成入口、内部观察点和巡检拓扑候选；
- 从 FARM/DAAAM 结果构建 SceneRevision；
- 自动用 Nav2 批量验证候选 Anchor；
- 从零完成新场地 onboarding。

“PCD/Potree 已决定废除”是目标决策；当前运行代码仍存在 PCD、Potree 和人工 Scene
Workspace，不能在文档或演示中声称它们已经删除。迁移完成后再做 breaking removal。

## 6. DAAAM 当前集成程度

现有 [`daaam_memory.py`](../../../services/control_api/daaam_memory.py)和
[`packages/semantic_memory/daaam.py`](../../../packages/semantic_memory/daaam.py)实现了一个
严格但很窄的边界：

- 只接收 Mission 后的离线导出；
- 必须绑定已有 Mission、Robot boot、定位 epoch、Observation Session 和 SceneRef；
- 结果只作为 `authority=observation_only`、`executionEligible=false` 的长期假设；
- centroid 只保留审计，投影进 Mission 前被删除；
- DAAAM 不在实时运动和启动链中。

此外，当前导入工具仍使用 Office V2 已知 Polygon 把 centroid 归属到已有 region。这形成
循环依赖：必须先有 Scene 和 Mission，才能导入记忆。因此当前能力可以帮助回答“过去在
某个已知房间看到过什么”，不能发现房间本身，也不能替换固定 Anchor。

## 7. FARM 当前集成程度

FARM 在 H-IDE、Planning Agent、Vision 和 Compose 中没有 Provider、Router、数据库表或
运行服务，当前集成度为 **0**。

它值得进入 Shadow Mode，因为当前 MuJoCo Observation Bridge 已能提供 FARM 所需的：

```text
RGB + metric depth + intrinsics + T_map_camera + timestamp
```

但产品必须新增 Recorder、ObservationBatch、Provider Adapter、Shadow Store、
`MemoryProviderResult` 和 World Model Admission。FARM 原生 `scene_state.pt` 是 PyTorch
pickle，不可作为产品跨服务协议。

## 8. Scene Explorer 当前状态

Scene Explorer 目前是 `skillstore/docs/architecture/scene-explorer-agent.md` 中的 Proposed
设计，不是已运行服务。项目输入已确认团队具有 PLY/3DGS 可视化能力，但当前 Release
Manifest 和本次可审计仓库中没有登记该 Renderer 的仓库、commit、API 和许可证；接入前
必须把它正式登记为一个 Renderer Adapter，不能仅凭“本机能打开”宣称产品已集成。

Scene Explorer 的正确职责是审计已经生成的 PLY/3DGS、规划虚拟相机视角、分析覆盖/孔洞
并提出标注。它不替代机器人首轮遍历、SLAM 或动态语义记忆。

## 9. 仿真与真机的定位差异

当前 MuJoCo 直接提供近似完美的 `map` Pose，bridge 使用 identity `map→odom` 并发布
MuJoCo root pose。它不是 SLAM。

真机必须由 FAST-LIO2/GTSAM、RTAB-Map 或其他正式定位栈提供同样的时间/坐标契约、
协方差和 Localization Epoch。Semantic Scene Compiler 必须在仿真中使用统一输入协议，
不能依赖 MuJoCo 私有真值接口，否则上真机仍会失效。

## 10. 目标迁移边界

### Cloud 保留

- PWA、租户/商业逻辑；
- Mission Architect、人工审核；
- 长期报告和全局 World Model；
- 重模型和异步 Scene Explorer。

### 从当前 Control API 抽取到 Field Executive

- 当前 Robot Pose 和执行 epoch；
- 可信 target 选择和本地 Goal Resolution；
- SkillRequest 签发、查询、取消和幂等对账；
- 当前 workstation authority 的 `vision.observe`、`scene.verify_inside_region`、
  `inspection.scan_station`、`mission.report` 及其 composite child 对账；
- 离线 Mission、Capability/Entitlement Cache；
- Robot Agent 连接和 Store-and-Forward。

### 保留并泛化

- 现有 Mission DAG、导航→验证→观察→报告结构；
- 现有 PlaceNavigationIntent 与扫描/报告 Skill；
- 将 Office catalog 查询替换为 bound SceneRevision 查询；
- 将固定 station index 替换为通用 `AnchorRegistry`；
- 将固定 route 替换为 Scene topology + route policy；
- 将 Office V2 bootstrap 降级为回归 fixture，不作为新场地上线方式。

## 11. 缺口定义

真正需要实现：

```text
SLAM / occupancy / TF ────────────────────────────────────────────────┐
posed RGB-D → FARM/DAAAM Result → World Model Admission/World Model ──┼→ Semantic Scene Compiler
PLY/3DGS → Scene Explorer Result → World Model Admission ─────────────┘
       ↓
几何/可达性验证 + 人工审核
       ↓
published SceneRevision
       ↓
现有 Mission Compiler / Field Executive / Nav2
```

不需要重写已稳定的 Nav2 规划、SONIC 运控和 MuJoCo 物理链，但必须泛化它们的 scene/map
装载与版本绑定；主要新增工作仍是上游“房间和观察点从哪里来”。

## 12. 开发机与目标 Demo 主机说明

编写本文时，当前编辑主机的 `nvidia-smi` 报告为 RTX 5060 Ti 16GB，并不是本文假设的
RTX PRO 6000。完整 FARM/DAAAM 不应在这台 16GB 主机并发启动。

PRO 6000 Demo 必须在目标工作站重新记录硬件 manifest。若目标卡实际是 RTX 6000 Ada
48GB，而不是 RTX PRO 6000 Blackwell 96GB，应采用顺序离线 Provider 或远程模型服务，
不能照搬 96GB 资源预算。

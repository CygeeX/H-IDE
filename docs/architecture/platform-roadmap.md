# PRO 6000 Demo 架构与代码迁移路线

| 属性 | 值 |
| --- | --- |
| 状态 | Proposed / 尚未实现 |
| 基线 | `skillstore-office-v2-e2e-20260901-r1` |
| 目标 | 同一台 PRO 6000 上以正式协议模拟 Cloud、Site、Edge，并在第二场景消除产品侧坐标常量 |
| 回退原则 | Office V2 当前链始终保留为 Last-Known-Good，逐阶段 shadow、切权和回退 |

详细设计以
[H-IDE 平台架构文档集](../../components/h-ide/docs/platform-architecture/README.md)为准；
本文只给出与当前发行源码一一对应的迁移顺序。

## 1. 目标部署

```text
PRO 6000 Workstation
├─ cloud-zone
│  ├─ PWA / Control API
│  ├─ H-IDE Mission Architect
│  ├─ Entitlement / Registry
│  └─ long-term World Model
├─ site-zone
│  ├─ Site Model Gateway
│  ├─ Observation Recorder
│  ├─ FARM / DAAAM Shadow Provider
│  ├─ Semantic Scene Compiler
│  └─ Scene Explorer（后置）
└─ edge-zone
   ├─ H-IDE Field Executive
   ├─ H-IDE Cognition Broker
   ├─ Mission Validator / Edge ledger
   └─ World/Entitlement cache + event spool
             │ formal Robot Agent protocol
             ▼
Robot/Simulation Runtime
   Robot Agent → Nav2/SLAM/TF → SONIC → G1/MuJoCo
```

同机部署不允许进程内 import 绕过边界。三个 Zone 使用独立 Compose project、网络、
身份、数据库/schema 和对象存储 bucket；Cloud 无法访问 Robot Agent command、ROS DDS
或 SONIC 端口。未来有 Jetson Thor 后迁移 edge-zone，不改变 Cloud API。

## 2. 源码归属

| 产品角色 | 迁移归属 | 当前可复用代码 |
| --- | --- | --- |
| Mission Architect | `components/h-ide-agent` | `src/mission/{engine,contracts,mastra-runtime,routes}.ts` |
| Field Executive | 新部署单元 `h-ide-field-runtime`，先从 H-IDE 孵化 | `packages/mission_core/runtime.py`、Mission issue/reconcile/result 语义 |
| Cognition Broker | Field Runtime | 新增 Local → Site → Cloud ModelInvocation policy |
| Live Visual Resolver | `components/h-ide-vision` | ingress identity/time gates、VisualTask、RGB-D/TF geometry、Evidence |
| World Model / Scene Compiler | 新服务域 `h-ide-scene-intelligence` | Scene API revision/store、现有 observation-only memory 作为 Adapter 输入 |
| Scene Publisher/Review | `components/h-ide` | Scene API、PWA `scene-*` workspace；需升级状态机和 provenance |
| Robot Runtime | `components/robotagent` | capability、lease、ledger、ROS Action、精确取消 |
| 仿真/第二场景 | `components/h-ide-mujoco-sonic` | Plant、传感、Nav2/SONIC、评测隔离 |
| Provider | 上游固定 commit 的独立镜像 | FARM、DAAAM；不复制原生 pickle/DSG 到产品协议 |

## 3. M0–M7

### M0：冻结基线和评测数据

- 固定本发行版 tag、组件 source commit、镜像/模型/场景 digest 和 SBOM；
- 保存启动、停止、进程级复位、双房巡检、报告和 Evidence 回归；
- 用只读 recorder/隔离 rosbag 录制 Office V2 posed RGB-D、内参、capture-time TF、
  covariance 和 Localization Epoch；
- 新增无产品语义坐标清单的第二 MuJoCo 场景，evaluator truth 单独挂载并禁止产品容器读取。

**门：** Office V2 可独立回退；两套 capture 可重复；Ground Truth 隔离测试通过。

### M1：跨 Zone 协议和故障域

- 在 `components/h-ide/packages/` 孵化、随后独立发布协议：`MissionBundle`、
  `EntitlementGrant`、`ObservationBatch`、`MemoryProviderResult`、`WorldDelta`、
  `semantic-scene-revision/2`；复用现有 MissionEvent/Capability Manifest 并增加 wire profile；
- 建 cloud/site/edge 独立 Compose、mTLS、本地 CA、event spool；
- 用 Toxiproxy/`tc netem` 验证延迟、丢包、断网、重复和乱序。

**门：** Cloud 无 Robot command 网络路径；Edge 可验签持久化 Bundle；重复事件不重复动作。

### M2：Field Executive 切权

- 先用 shadow ledger 读取当前 Mission 事件，不发命令；
- 从 `services/control_api/main.py` 和 `mission_service.py` 抽取 Robot Agent dispatch/query/
  cancel、inside verifier、scan composite、视觉调度、报告和 unknown-outcome 对账；
- Field Executive 成为 Robot Agent Gateway 的唯一上游 command client；Cloud 只保留 authoring、
  review 和长期记录；
- Cognition Broker 按置信度、新颖度、隐私、授权、网络和时延选择 Local/Site/Cloud，
  返回 hypothesis，不返回直接运动命令。

**门：** Office 行为不回退；断 Cloud 后已签名 Mission 的导航、验证、扫描、报告全部完成；
重连不重复导航/Spin/observe/report。

### M3：Recorder、Memory Shadow 和 World Model

- 建内容寻址 Observation Recorder，一次采集后 fan-out 给 Live Visual Resolver、FARM、DAAAM；
- FARM/DAAAM 使用独立 CUDA/ROS 镜像和私有卷，顺序 replay，不能发现 Robot Agent/Nav2/SONIC；
- 新建 Provider-neutral Admission/Canonical World Model；Mission `WorldRecord` 只接收有界投影；
- 当前 DAAAM importer 保留为 legacy，新 v2 Adapter 解除已有 Mission/Office Polygon 依赖。

**门：** Provider OOM/退出不影响运动；每个 candidate 能追溯帧、Pose、模型、版本；
关闭任一 Provider 不影响已发布 Scene 的任务。

### M4：Semantic Scene Compiler 和 Review

- 从 occupancy/ESDF/拓扑生成 Region/Corridor/Portal 候选；模型只负责命名、关系和排序；
- 生成 entrance/interior/observation-vantage，并执行 footprint clearance、Nav2
  `ComputePathToPose`、visibility 和 Evidence 验证；
- Scene API 升级为 `draft → proposed → reviewed → published`，Publisher 是唯一发布权威；
- PWA 展示 2D/3D、Provider diff、Evidence、置信度、可达性和 revision diff；
- Office catalog 先适配为 `source=legacy_reviewed`，再把 alias/station/fixed route 查询替换为
  Scene entity、AnchorRegistry 和 topology route policy。

**门：** Reviewer 不改代码可发布；未发布/模型自造/未知 frame/不可达 Anchor 无法进入 Mission。

### M4a：通用地图装载和冷切换

- 泛化 `scene_runtime.py`、Nav2 launch/params、static planner、bundle verifier、motion runner 和 launcher；
- 统一绑定 `sceneId + sceneRevision + mapDigest + runtimeGeometryDigest + localizationEpochId`；
- 第一版只做受控冷切换：停稳并释放 → deactivate Nav2 → 校验加载 → 新 epoch → activate →
  Field cache rebind；失败原子回 Last-Known-Good。

**门：** 第二场景不依赖 `office_v2` 路径/ID；任一 digest/epoch 不一致 fail closed；回退无半激活状态。

### M5：第二场景无坐标常量 E2E

- 自动/监督采集 → ObservationBatch → Provider → Compiler proposal → 人工一次审核 → published revision；
- 验证两种初始 Pose、相反访问顺序、单/双房巡检、记忆问答、fresh visual approach、
  断云/重连、Provider OOM、epoch 变化和 revision 回滚；
- CI 禁止第二场景 room/entry/station/route/scan 坐标进入产品代码/config。

**门：** 单/双房各连续成功三次；每房有 EvidenceRef；无重复 Robot Agent command；
关闭 FARM/DAAAM 后仍可用 published revision 巡检。

### M6：Scene Explorer 与旧 Viewer 退役

- 登记 PLY/3DGS Renderer repo、commit、API、格式、许可证和 SBOM；
- Scene Explorer 生成多视角、覆盖/孔洞/尺度报告和补拍候选，但无机器人凭据；
- 新 Viewer/资产链通过后，以独立 breaking release 删除 Potree/PCD 运行代码。

**门：** Explorer 退出不影响 Field/Robot Runtime；每项报告可跳转 Evidence；未标定资产不能发布 Anchor。

### M7：Thor 和真机 Pilot

- 将 edge-zone 镜像迁到 ARM64/JetPack；验证 mTLS、event spool、OTA、模型资源、功耗、热和长稳；
- 分开验证 simulation、pilot、production Safety Profile；
- Developer Kit 通过后再做背包、载板、电池和散热设计。

**门：** 不改 Cloud API 即完成迁移；真机急停、停止距离、定位漂移、人员/反光场景另行验收。

## 4. 切换与回退规则

推荐显式开关：

```text
FIELD_EXECUTION_AUTHORITY=control_api|shadow|field_runtime
MEMORY_PROVIDER_MODE=off|shadow
SCENE_SOURCE=legacy_office_v2|scene_revision_v2
NAV2_SCENE_BINDING=legacy|revision
```

- 执行权只能在无 active/unknown physical command 时切换，永不双活；
- Shadow Provider 没有事实发布权或运动凭据；
- SceneRevision、map binding 和模型包都不可变、可审计、可回滚；
- 任一阶段失败先停稳、对账、回 LKG，不能用旧 Mission 重放物理动作；
- M5 通过前不得删除 Office V2 legacy 数据源；M6 通过前不得删除当前 Viewer/资产链。

## 5. 成功定义

只有当一个没有产品侧房间/入口/站点/路线坐标的新场景完成自动候选、人工一次审核、
中文巡检、断云继续和重连无重放，才可以宣称“地图硬编码问题已解决”。只让 FARM/DAAAM
识别物体，或把现有坐标搬进另一个 YAML/数据库，都不算完成。

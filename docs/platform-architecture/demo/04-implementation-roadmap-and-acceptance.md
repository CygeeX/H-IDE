# Demo 实施路线、测试与验收

| 属性 | 值 |
| --- | --- |
| 状态 | Executable roadmap / 待排期 |
| 核心目标 | 在 PRO 6000 + 同一 LAN 上证明正式云边架构和新场地无坐标常量巡检 |
| 回退原则 | 现有 Office V2 稳定链在新链完成验收前保持可运行 |

## 1. 最终 Demo 成功定义

Demo 必须证明：

> 在一个没有房间 Polygon、入口、观察站和固定路线常量的新场景中，机器人通过 posed
> RGB-D/SLAM 数据建立语义候选，系统自动生成并让人一次审核 SceneRevision；之后用户用
> 中文要求巡检一间或多间办公室，机器人完成导航、观察和带证据报告。任务执行中断开
> Cloud 后继续，重连不重复动作。

只让当前 Office V2 继续成功、只让 FARM/DAAAM 检出物体，或把坐标从 Python 移到另一个
YAML/数据库，都不算解决问题。

## 2. 推荐源码/服务归属

| 组件 | 推荐归属 | 说明 |
| --- | --- | --- |
| Cloud Control Plane、PWA、Scene Review | `h-ide` | 保留当前产品入口；不再拥有最终 Robot command ownership |
| Mission Architect | 当前 `h-ide-agent` | 产品角色改名，继续只输出高层 Mission |
| Field Executive / Cognition Broker | 新独立部署单元 `h-ide-field-runtime` | 当前可先从 `MissionService` 抽取；未来迁到 Thor |
| Live Visual Resolver | `h-ide-vision` | 保留实时 RGB-D/TF 几何契约，模型 Provider 可共享 Site Gateway |
| Canonical World Model / Scene Compiler | 新 `h-ide-scene-intelligence` 服务域 | 可先在 H-IDE 仓库孵化，但必须独立 API、存储和容器 |
| Scene Explorer | 独立 `h-ide-scene-explorer` Worker | 异步、GPU-heavy、observation-only |
| Robot protocol/runtime | `robotagent` | 保持能力目录和 ROS 出口唯一 |
| 仿真、第二场景、数据真值 | `h-ide-mujoco-sonic` | 真值只用于评测，不进入产品 Provider |
| FARM/DAAAM | 上游固定 commit 的外部 Provider 镜像 | 不复制进主仓库，不共享原生数据格式 |
| 跨端 Schema | 建议独立 `h-ide-protocols` 发布物 | Demo 可先在 `packages/` 孵化，成熟后独立版本发布 |

是否立即创建新 Git 仓库不是第一步；第一步是确保服务通过序列化协议通信，而不是进程内
import。统一 [`infra/release-bom.yaml`](../../../infra/release-bom.yaml)或其后继 Manifest
固定所有仓库、镜像、模型和 Schema。

## 3. 里程碑 M0：冻结基线和评测数据

### 工作

- 为现有稳定 Office V2 联调记录 tag、commit、镜像 digest、模型和配置摘要；
- 将现有旧 `release-bom.yaml` 升级为包含当前 branch/commit、DAAAM、FARM、skillstore、
  镜像/模型 digest、Provider 依赖和 remote 的 Release Manifest；
- 保存启动、停止、回到初始位、巡检和报告的回归用例；
- 先实现最小只读 Observation Recorder，或使用隔离 rosbag/专用采集器录制 Office V2
  posed RGB-D/TF/内参/定位序列；记录 dropped frame、sequence gap、同步误差和背压；
- 创建第二个新 MuJoCo 场景，只在 evaluator 保存房间/对象真值；
- 新场景产品侧禁止 room/anchor/route 坐标清单；
- 第二场景的产品 occupancy/mesh 必须来自模拟 LiDAR/RGB-D + SLAM/点云导入链，不能由
  含 evaluator 房间/门/Anchor 语义的 MuJoCo manifest 直接生成；
- MuJoCo 碰撞几何和初始 spawn 坐标列入显式 allowlist，仅供物理仿真/重置，不进入
  Semantic Scene Compiler 或 Mission；
- 建立数据 manifest、hash、保留策略和 Ground Truth 隔离检查。

### Exit Gate

- Office V2 稳定链可独立回退；
- 两套可重复 capture 存在，并可由受版本控制的转换器生成 ObservationBatch；
- evaluator truth 无法被 H-IDE、Provider、Vision 或 Mission 运行容器读取。

## 4. 里程碑 M1：跨端协议和同机故障域

### 工作

- 审计、复用并升级现有 `MissionEvent v2`、`CapabilityManifest v2` 和资产型
  `SceneRevision`；新增跨 Zone wire profile、签名和兼容 Adapter，而不是假设这些名字
  当前不存在；
- 新增 `MissionBundle`、`EntitlementGrant`、`ObservationBatch`、
  `MemoryProviderResult`、`WorldDelta` 和 `h-ide.semantic-scene-revision/2`；
- 建立 cloud/site/edge 三个 Compose project、独立网络/数据库/身份；
- 建立本地 CA 和 mTLS；
- 加入 event spool 与幂等对账；
- 使用 Toxiproxy/`tc netem` 注入 WAN；
- 生成统一 Release Manifest 和 SBOM。

### Exit Gate

- Cloud 无网络路径访问 Robot Agent；
- Edge 可以验签和持久化 Bundle；
- MissionEvent 重复/乱序不会重复执行；
- 停止 cloud-zone 不会让 Robot Runtime 或停止能力退出。

## 5. 里程碑 M2：Field Executive 上游执行权迁移

### 工作

1. 先以 shadow 模式读取当前 Mission 事件，不发命令；
2. 对比当前 `MissionService` 和新 Field Executive 的节点决策；
3. 抽取 current Pose、target fact、SkillRequest、query/cancel 和 execution epoch；
4. 将唯一上游 Robot Agent session/dispatch authority 从 Control API 切换到 Field
   Executive；Robot Agent 仍拥有本地 ROS Action goal、资源仲裁、租约和取消，Nav2 仍
   拥有路径/base motion，SONIC 仍拥有全身控制；
5. Cloud 只保留 Mission authoring、审核、revision 和长期记录；
6. 建立 Local → Site → Cloud Cognition Broker；
7. 把当前 Control API 的 `vision.observe`、`scene.verify_inside_region`、
   `inspection.scan_station`、`mission.report` 拆为 Edge-local 或允许的 Site service；
8. 迁移扫描复合执行器、多方向视觉调度、inside-region verifier、Evidence 聚合和本地报告；
9. 为重启中的 composite child 建立幂等身份和对账，禁止重复 Spin/observe/report；
10. 实现预授权标准巡检、已知对象查询、停止和返航策略。

### Exit Gate

- Office V2 行为与报告不回退；
- Cloud 断开后已签名 Mission 能继续；
- 断云后导航、房间内验证、旋转、观察和最终报告都能完成，不只是继续 Nav2；
- 新开放式复杂任务离线时返回 `deferred_cloud`；
- 重连对账不产生重复导航、转向、扫描或报告节点。

## 6. 里程碑 M3：Observation Recorder 与 Memory Shadow

### 工作

- 将 M0 最小采集器产品化为 Observation Recorder，从 Observation Gateway 只读 fan-out
  posed RGB-D，并定义 sequence-gap/backpressure 行为；
- 内容寻址保存 RGB、Depth、TF、内参、标定、地图资产与证据 manifest；
- 实现 FARM Adapter 和独立 Provider 镜像；
- 实现 DAAAM Adapter，解除对已有 Mission/Office Polygon 的 onboarding 依赖；
- 两个 Provider 输出 `MemoryProviderResult` 到独立 shadow store；World Model Admission
  单独完成 canonical candidate/entity 身份和 WorldDelta；
- Shadow 优先离线 replay；如需 ROS，Provider 使用独立 ROS Domain，不能发现 Robot
  Agent/Nav2/SONIC，也不持有 Robot Agent token；
- 建立 PWA diff、性能、显存和许可证 manifest 页面；
- 单卡 PRO 6000 对同一数据顺序回放，避免并发 OOM。

### Exit Gate

- FARM/DAAAM 任一退出不影响现有 Mission；
- Provider 原生 pickle/DSG 不进入跨服务 API；
- 每个候选可追溯到帧、Pose、模型和版本；
- 能比较对象召回、ID switch、关系/时序查询、房间候选、时延和峰值显存。

## 7. 里程碑 M4：Semantic Scene Compiler 与审核 UI

### 工作

- 建立 Canonical World Model，不直接复用 Mission 局部 WorldRecord；
- 从 occupancy/ESDF/拓扑生成 room/corridor/portal 候选；
- 融合 FARM/DAAAM/视觉证据完成命名、对象和区域候选；
- 生成 entrance/interior/observation-vantage；
- 批量执行 footprint clearance、Nav2 `ComputePathToPose` 和 visibility 检查；
- PWA 展示 2D/3D、证据、Provider diff、置信度和新旧 revision diff；
- 实现 Compiler proposal → Reviewer decision → Scene Publisher/Registry publish；
- 新 `h-ide.semantic-scene-revision/2` 使用不可变 ID、签名和回滚引用；
- Navigation Anchor 必须来自 Compiler proposal，并绑定 geometry validation、map digest 和
  Evidence provenance；Reviewer 编辑形成显式 diff，不覆盖原 proposal；
- 生产角色不能通过 legacy annotation API 从空白创建可执行 Anchor。

### Exit Gate

- Office V2 Shadow Compiler 可与 Legacy catalog 比较；
- Reviewer 不编辑代码即可接受、修改或拒绝候选；
- 发布日志能证明 Anchor 来源于 Compiler proposal，而不是人工重新录入全部坐标；
- 未发布 proposal 无法进入 Mission；
- 模型生成坐标、未知 frame 或不可达 Anchor 被拒绝。

## 8. 里程碑 M4a：通用地图装载与场景切换

### 工作

- 将 Semantic SceneRevision 绑定 occupancy/map artifact、map metadata、map digest 和
  runtime geometry digest；
- 泛化 MuJoCo scene runtime、Nav2 params、strict planner、motion runner 和 launcher，删除
  对 `office_v2` 文件名/scene ID 的运行时强制；
- 让 Nav2 map_server 加载指定 SceneRevision 的 map artifact；第一版采用**受控冷切换**：
  机器人停止并释放控制 → 停止导航 lifecycle → 校验并加载 map → 新 Localization Epoch →
  激活 Nav2/Robot Agent → Field Executive 重新绑定；不在运动中热切地图；
- strict planner 校验任意 `sceneId + sceneRevision + mapDigest + runtimeGeometryDigest`；
- Robot Agent SceneRef、Localization Epoch、Nav2 map、Field Executive cache 使用同一 binding；
- 保存 Last-Known-Good map/binding，切换失败时原子回退，不保留半激活状态；
- 切换接口不能读取 evaluator room/door/anchor truth。

### Exit Gate

- 第二场景通过 SceneRevision 引用加载，不需要 Office V2 路径或 ID；
- 任一 digest、SceneRef 或 Localization Epoch 不一致时 fail closed，不能导航；
- 切换失败能回退 LKG，且未知物理动作不被重放；
- 新 map 激活后 Robot Agent 和 Field Executive 都报告同一 Scene binding；
- Office V2 仍可作为 Legacy SceneRevision 回归运行。

## 9. 里程碑 M5：第二场景无硬编码 E2E

### 最小场景要求

- 至少 3 个可命名区域；
- 至少 2 个门/Portal 与一段走廊；
- 每个房间包含可区分对象；
- 至少一个重复对象类别，用于验证实例/关系记忆；
- 至少一个需要多视角才能确认的对象；
- 产品配置中没有房间、入口、站点、扫描角或巡检顺序坐标；
- 至少两个不同初始 Pose；MuJoCo 物理 spawn 可在 evaluator/重置 allowlist 中存在，但不能
  被 Mission 或 Compiler 当作固定规划起点；
- SceneRevision 保存拓扑和 approved anchors，不保存唯一巡检路线。

### E2E 用例

| ID | 用户/故障输入 | 预期 |
| --- | --- | --- |
| E2E-01 | 新场景 onboarding | 第一版由人工监督遥控采集；后续处理自动生成 ObservationBatch 和 Scene proposal，不使用预写语义覆盖路线 |
| E2E-02 | Reviewer 发布 | 不改代码即可发布 SceneRevision |
| E2E-03 | “巡检开放办公区” | 进入房间、停稳、多方向观察、按房间报告 |
| E2E-04 | “依次巡检开放办公区和会议室” | 完整两房间 DAG，不漏观察、不重复走位 |
| E2E-05 | “刚刚哪里看到电脑？” | 从任务证据/语义记忆回答位置和证据，不误报没有看到 |
| E2E-06 | “走到刚才的架子前面” | 记忆给出候选，实时视觉/Depth 生成 fresh goal 后导航 |
| E2E-07 | 本地低置信度复杂场景 | Broker 自动升级 Site/Cloud，PWA 展示阶段和来源 |
| E2E-08 | Mission 接收后断 Cloud | 任务继续，事件落盘，报告在重连后回填 |
| E2E-09 | Cloud 重连并重复消息 | 不重复任何物理节点 |
| E2E-10 | FARM/DAAAM Provider OOM | 运动链继续；认知降级或明确 deferred |
| E2E-11 | Localization Epoch 改变 | 旧 live target 失效，重新定位/审核，不使用历史坐标 |
| E2E-12 | 回滚 SceneRevision | 新 Mission 使用指定旧 revision，历史 Mission 可审计 |
| E2E-13 | 第二初始 Pose + 相反房间顺序 | Mission 根据当前 Pose/请求动态排序，不复用固定路线 |

### Exit Gate

- 单房间和双房间巡检各连续成功 3 次；
- 两个不同初始 Pose、两种相反房间访问顺序均通过；
- 每个房间报告至少包含一个带哈希的视觉 EvidenceRef；
- 每个计划扫描方向都有到达目标之后采集的新鲜 Evidence；
- 断云/重连用例无重复 Robot Agent command；
- 新场景上线没有代码或产品配置坐标提交；
- 中文名称和别名来自 Scene entity/reviewer，不来自场景专用 Python 映射；
- 每个导航节点绑定新的 `h-ide.semantic-scene-revision/2`，禁止 fallback 到 Office catalog；
- 所有房间间 connectivity edge 都通过 Nav2 reachability；
- 巡检结果满足 `fullChainAccepted=true`、`missingStations=[]`、`headingGaps=[]`，或使用经
  ADR 定义的等价覆盖阈值；
- FARM/DAAAM 关闭后仍能回退到已发布 SceneRevision 执行已知巡检。

## 10. 里程碑 M6：Scene Explorer 与地图质量

### 工作

- 接入已登记的 PLY 和 3DGS Renderer Adapter；
- 生成 Top/Front/Side/orbit/室内路径视角；
- 输出 RGB/Depth/visibility、coverage、孔洞、密度和尺度报告；
- 产生 proposed annotation 和低覆盖补拍请求；
- 真实补拍请求必须经过新 Mission，而不是 Explorer 直接移动机器人；
- 在替代 Viewer/资产链完成后，单独执行 PCD/Potree breaking removal。

### Exit Gate

- 同一资产和 profile 产生稳定 view plan hash；
- 报告每个结论可跳转 Evidence；
- `sceneToMap` 未确认时无法发布 navigation anchor；
- Scene Explorer 进程退出不影响 Field Executive/Robot Runtime。

## 11. 里程碑 M7：Thor 迁移验证

购买 Thor Developer Kit 后：

- 把 edge-zone 镜像迁到 ARM64，不修改 Cloud API；
- 验证 JetPack、ROS、容器、mTLS、event spool 和 OTA；
- 分别 benchmark T4000/T5000 目标资源 Profile；
- 验证 Field Executive、本地视觉、DAAAM/FARM 候选、量化 VLM；
- 运行热、功耗和长稳测试；
- Developer Kit 通过后才开始量产载板/背包设计。

## 12. 自动化测试层

### 12.1 Contract tests

- Schema 正反例、未知字段、版本兼容；
- 签名、过期、设备/站点作用域；
- idempotency、sequence gap、乱序和 replay；
- Provider output 中禁止可执行 target；
- SceneRevision 状态机和不可变性。

### 12.2 Component tests

- Observation 时间同步、单位和坐标系；
- World entity merge/conflict；
- room/portal/anchor candidate；
- clearance、reachability、visibility；
- Cognition Broker 路由、额度、隐私和 fallback；
- edge event spool crash recovery。

### 12.3 Simulation tests

- 使用 MuJoCo Ground Truth 只做离线评分；
- 从产品容器网络验证无法访问 Ground Truth；
- 自动复位后运行全套 onboarding/inspection；
- 注入障碍、Provider OOM、网络故障、Cloud/Site 重启；
- simulation Safety Profile 记录接触并重规划，除非确实无法继续才失败。

### 12.4 Production-profile tests

仿真放宽测试不能替代真机：Pilot 前另行验证硬件急停、速度、碰撞、停止距离、网络、
定位漂移、玻璃/反光、人员进入和 SONIC 平衡持续性。

## 13. Provider 选择指标

| 类别 | 指标 |
| --- | --- |
| 对象 | instance precision/recall、重复/漏检、3D centroid/AABB error |
| 身份 | ID switches、fragmentation、跨视角/跨任务重识别 |
| 语义 | 中文别名、room assignment、caption 质量 |
| 关系 | Recall@K、Accuracy@1、空间谓词正确率 |
| 时间 | first/last observed、移动/删除对象查询 |
| 地图 | room/portal/connectivity proposal recall、Anchor publish rate |
| 系统 | FPS、端到端时延、峰值显存、磁盘增速、重启恢复 |
| 商业 | 许可证、权重、ARM64、维护成本、商业授权报价 |

通过 benchmark 和法律门槛的 Provider 最多获得经 World Model Admission 向 Semantic
Scene Compiler 提交 proposal 的资格；FARM/DAAAM 都永远不拥有事实发布权。

## 14. 风险登记

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 单卡 GPU OOM | 实时视觉和 Mission 卡顿 | 顺序 Shadow、GPU 队列、共享模型、双卡升级门 |
| FARM/DAAAM 许可证不适合闭源 | 商业发布受阻 | Provider 隔离、SBOM、替代依赖、商业授权/自研决策门 |
| MuJoCo 完美定位掩盖真机问题 | Demo 通过但真机失败 | 统一 Observation contract、噪声/漂移注入、真实数据回放 |
| 模型把对象误认成房间/门 | 错误 Scene proposal | 几何先验、Evidence、人工审核、不可直接发布 |
| Scene Compiler 只适配 Office V2 | 新形式硬编码 | 第二场景、无常量 CI、Provider-neutral Schema |
| 云边同机掩盖故障 | 商业架构假验证 | 独立存储/网络/身份、WAN 注入、后续 Thor/双机验证 |
| 迁移破坏稳定联调 | 演示不可用 | Legacy SceneRevision、feature flag、shadow、LKG 和逐步切权 |
| 仿真放宽策略进入真机 | 安全风险 | 三套独立 Safety Profile、生产配置不可继承 simulation |

## 15. 交付清单

- 架构与协议 ADR；
- Release Manifest、SBOM 和模型/许可证清单；
- PRO 6000 cloud/site/edge 部署清单；
- ObservationBatch 数据集与 Ground Truth 隔离证明；
- FARM/DAAAM benchmark 报告；
- Semantic Scene Compiler 和 PWA Review；
- 第二场景 SceneRevision；
- 12 个 E2E 用例报告与 Evidence；
- 断云、重连、Provider OOM 和回退报告；
- Thor 迁移前的资源/接口基线。

完成 M5 后可以宣称“本地架构和新场地自动候选流程可行”；完成 M7 和真机 Pilot 后，
才能宣称“Edge Pro/Max 商业硬件方案可行”。

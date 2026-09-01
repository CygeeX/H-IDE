# Semantic Scene Compiler：从零建立记忆并替换场景常量

| 属性 | 值 |
| --- | --- |
| 状态 | Demo implementation design |
| 解决的问题 | 新场地不修改后端坐标常量，也能生成可审核房间、门、拓扑与导航 Anchor |
| 不解决 | 不替代 SLAM/Nav2，不让 LLM 直接生成坐标，不宣称一次观察覆盖未知区域 |

## 1. 正确的问题定义

“从零建立对象记忆”和“从零建立可导航地图”是两个问题：

- FARM/DAAAM：从**机器人已经采集的 posed RGB-D**建立对象、关系、时间和区域候选；
- SLAM/Geometry：建立坐标、自由空间、障碍和可通行性；
- Semantic Scene Compiler：把两类结果编译成可审核、版本化的场景产品；
- Mission Compiler：引用发布后的实体，不再引用 Office V2 代码常量。

因此目标不是“用 FARM/DAAAM 自动替代硬编码坐标”，而是：

> 用 FARM/DAAAM 建立语义/时空候选，由 Semantic Scene Compiler 联合
> SLAM/occupancy/Nav2 校验并发布 SceneRevision，从而替代代码内场景常量。

## 2. 从零 onboarding 的完整数据流

```text
新建 site / sceneCandidate
        ↓
人工监督遥控、覆盖路线或 frontier exploration
        ↓
SLAM / localization
├─ occupancy / TSDF / ESDF / mesh
├─ map frame / covariance / localization epoch
└─ posed RGB-D / intrinsics / calibration / timestamps
        ↓
Observation Recorder + immutable Evidence
        ├─ FARM Provider（对象关系记忆，shadow）
        ├─ DAAAM Provider（Track/时间/层级，shadow）
        └─ reconstruction/SLAM asset exporter
                  └─ PLY/3DGS → Scene Explorer（离线质量和标注候选）
        ↓
MemoryProviderResult / SceneAuditResult
        ↓ World Model Admission / identity normalization
Provider-neutral WorldDelta observations + hypotheses
        ↓
Canonical H-IDE World Model
        ↓
Semantic Scene Compiler
├─ Room / Corridor / Door candidates
├─ connectivity graph
├─ entrance / interior / observation-vantage candidates
├─ scan headings / visibility score
└─ semantic names / aliases / evidence
        ↓
deterministic geometry and Nav2 validation
        ↓
PWA review: proposed → reviewed → published SceneRevision
        ↓
Mission Architect / Field Executive / Local Goal Resolver
```

机器人或相机必须实际获得新视角。只扔进一张未覆盖完整场地的 PLY，任何模型都不能凭空
知道墙后房间；第一轮采集需要人工遥控、预设覆盖路线或 frontier exploration。

## 3. 首轮探索策略

第一阶段推荐“**自动探索 + 人工监督**”，而不是完全人工标点或无人看管全自动发布：

1. 操作者确认机器人、传感器、地图坐标和工作区域；
2. frontier exploration service（可参考 m-explore-ros2 思想）选择未知边界目标，Nav2
   只负责规划并执行到该目标；
3. Field Executive 只执行限制区域内的观察/导航节点；
4. 每到一个观察位采集 RGB-D、TF、定位协方差和视觉覆盖；
5. 低覆盖、回环失败、玻璃/反光或定位漂移时请求人工接管；
6. 完成覆盖后冻结 `observationSessionId` 和数据 manifest；
7. Provider 和 Compiler 使用同一份不可变数据进行可重复处理。

可参考 `m-explore-ros2`/frontier exploration 的未知区域选择思想，但探索仍通过 Robot
Agent/Nav2 执行，不让 FARM、DAAAM 或 Scene Explorer 直接发布运动命令。

## 4. `ObservationBatch` 输入契约

每一批必须包含：

```json
{
  "schemaVersion": "h-ide.observation-batch/v1",
  "tenantId": "tenant-a",
  "siteId": "site-a",
  "robotId": "g1-sim",
  "robotBootId": "boot-...",
  "localizationEpochId": "loc-...",
  "sceneCandidateId": "scene-candidate-...",
  "observationSessionId": "obs-...",
  "source": {
    "sensorId": "head-rgbd",
    "sourceId": "gateway-...",
    "sourceMode": "simulation|live|replay"
  },
  "coordinate": {
    "mapFrameId": "map",
    "baseFrameId": "base_link",
    "cameraFrameId": "camera_optical_frame",
    "handedness": "right",
    "axisConvention": "ros-rep-103"
  },
  "sensorModel": {
    "width": 1280,
    "height": 720,
    "intrinsics": {"fx": 0, "fy": 0, "cx": 0, "cy": 0},
    "distortionModel": "plumb_bob",
    "distortionCoefficients": [0, 0, 0, 0, 0],
    "TBaseCameraRowMajor16": [
      1, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 1
    ],
    "calibrationSha256": "sha256:..."
  },
  "clock": {
    "domain": "robot-monotonic+utc-correlated",
    "maximumTimestampUncertaintyMs": 5,
    "maximumRgbDepthTfSkewMs": 20
  },
  "mapAssets": {
    "mapRevision": "map-rev-...",
    "occupancyUri": "...",
    "occupancySha256": "sha256:...",
    "meshUri": "...",
    "meshSha256": "sha256:..."
  },
  "frames": [
    {
      "sequence": 1,
      "capturedAt": "...",
      "rgb": {
        "uri": "...", "sha256": "sha256:...", "encoding": "jpeg",
        "width": 1280, "height": 720
      },
      "depth": {
        "uri": "...", "sha256": "sha256:...", "encoding": "z16",
        "width": 1280, "height": 720, "unit": "m", "scale": 0.001,
        "invalidValue": 0
      },
      "localizationObservedAt": "...",
      "TMapCameraRowMajor16": [
        1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1
      ],
      "poseCovariance36": [
        0.01, 0, 0, 0, 0, 0,
        0, 0.01, 0, 0, 0, 0,
        0, 0, 0.01, 0, 0, 0,
        0, 0, 0, 0.01, 0, 0,
        0, 0, 0, 0, 0.01, 0,
        0, 0, 0, 0, 0, 0.01
      ]
    }
  ],
  "captureManifestSha256": "sha256:...",
  "retentionPolicy": "...",
  "privacyPolicy": "..."
}
```

`TBaseCameraRowMajor16` 和 `TMapCameraRowMajor16` 必须是定义明确的 4×4 row-major 变换；
`poseCovariance36` 必须是 6×6 row-major covariance。所有数组长度、有限值、单位、frame、
clock 和同步偏差都由 Schema 强校验。Provider 不能用“最新 TF”静默替换采集时 TF，也
不能把缺失深度伪造成真实 metric depth。Recorder 另行记录 dropped-frame、sequence gap
和背压统计。

## 5. FARM 与 DAAAM 的角色

| Provider | 当前状态 | Demo 目标状态 |
| --- | --- | --- |
| FARM | 上游源码已下载；H-IDE runtime 集成度 0 | 独立 Shadow Provider，对不可变 capture 做 replay |
| DAAAM | 只有 Mission 后 observation-only YAML/JSON importer | 新建独立 v2 Shadow Provider；不复用现有 importer 充当 live producer |

### 5.1 FARM Provider

FARM 适合输出：

- 开放词汇对象候选；
- caption、视觉/语言 embedding；
- 对象间空间关系与共视；
- 对象的多视角证据和最佳观察视角；
- 关系语言查询的候选排序。

FARM 中的 Gaussian 是对象点分布的统计 geometry，不是照片级 3DGS。它的原生 pickle
只留在 Provider 私有卷，Adapter 输出 JSON/Protobuf `MemoryProviderResult`。

### 5.2 DAAAM Provider

DAAAM 适合输出：

- 跨帧 Track 和对象 ID 候选；
- first/last observed 与场景变化；
- 动态/静态状态；
- place/room/object 层级和区域候选；
- 时序查询和跨 Mission 关联。

DAAAM 的 centroid 和 region 也只作为 hypothesis；当前版本不能继续依赖 Office Polygon
做 region 归属，否则仍然没有实现从零 onboarding。

### 5.3 Shadow 原则

- 两个 Provider 读取同一 ObservationBatch；
- 各写独立 shadow store；
- 不直接写 published World Model；
- 不直接影响正在执行的 Mission；
- 使用统一 benchmark 比较后，通过门槛的 Provider 也只能经 Admission 向 Compiler 提交
  proposal；
- 两个 Provider 永远不是最终事实写入者；唯一发布权威是 Reviewer Policy + Scene
  Publisher/Registry。

## 6. Scene Explorer 的输入和作用

Scene Explorer 消费已建立的 PLY/3DGS 和相机/注册信息：

- 用虚拟相机生成 Top/Front/Side/orbit/室内漫游视角；
- 计算覆盖、孔洞、密度、尺度、遮挡和异常；
- 用多模态模型提出房间、门牌、家具和设备标注；
- 为低覆盖区域提出真实机器人补拍请求候选；
- 输出地图质量报告和 EvidenceRef。

它不拥有实时机器人凭据。真实补拍必须形成新的 Mission 节点，由 Field Executive 和
Nav2 执行。

当前 Release Manifest 尚未登记团队现有 PLY/3DGS Renderer 的 repo、commit、API、格式、
许可证和 SBOM；登记并实现 Renderer Adapter 是前置条件。每个视角必须声明
`provided_camera|reconstructed_camera|synthetic_camera`，证据等级必须声明
`captured_rgbd|synthetic_render|geometry_only`。虚拟室内漫游只是资产相机规划，不证明
机器人实际到过或该路径真实可通行；Explorer 只能提出补拍 request，不能自行创建或
执行 Mission。

纯 FAST-LIO2 PLY 可能没有颜色、门牌细节或原始相机帧。onboarding 必须同时保留 posed
RGB-D；3DGS 的渲染深度用于审计和可见性，不作为 Nav2 碰撞权威。

## 7. Canonical World Model

Provider 首先输出 `MemoryProviderResult`：

```text
provider entity candidate
├─ providerEntityId / sourceEntityId
├─ labels, aliases, captions, embeddings
├─ geometry with frame, unit, covariance and time
├─ object track / dynamic status
├─ relations and region hypotheses
├─ EvidenceRefs
├─ confidence / uncertainty
├─ provider, model, license and version manifest
└─ executionEligible=false
```

World Model Admission 校验租户、证据、坐标、时间和 Provider manifest 后分配 `candidateId`；
Canonical World Model 完成去重/冲突保留或审核绑定后才产生 `entityId`。实际采集记录可形成
`historical_observation`，模型 label/room/关系/合并推断只能形成 `hypothesis`；二者都不可
执行。Provider 和 Scene Explorer 永远不能产生 `published_static_fact` 或
`live_execution_target`。

不要继续无限扩展 Mission 局部 `WorldRecord` 来承担站点长期世界模型。应建立独立 World
Model Schema，再按具体 Mission、场景 revision 和权限投影有界记录。

实体合并不得采用“最后写入者获胜”。合并需要空间重叠、Track、外观 embedding、时间和
人工纠正证据；冲突候选保留来源，直到 Compiler 提案、Reviewer 决策和 Scene Publisher
发布。

## 8. Compiler 的确定性阶段

### 8.1 坐标和资产准入

- 校验 `map` frame、单位、handedness、时间和 Localization Epoch；
- 验证 scene↔map 注册和误差界；
- 关联 occupancy/ESDF/mesh、原始 Evidence 和 Provider 结果；
- 拒绝 NaN/Inf、时间错位、未知尺度和无法追溯的模型输出。

### 8.2 房间/走廊/门候选

几何算法先从自由空间、墙体、拓扑骨架和窄连接生成 region/portal 候选；语义模型负责：

- 给候选命名、生成中文/英文别名；
- 结合门牌、家具和跨视角证据判断房间类型；
- 建议合并或拆分；
- 给出不确定性和需要补拍的位置。

LLM/VLM 不直接画最终 Polygon。最终 Polygon、门宽和连通性来自几何候选和审核修订。

### 8.3 Anchor 候选

对每个 Portal/Room 生成候选：

1. 从门内外法向做多距离采样；
2. 在房间自由空间骨架/medial axis 上采样内部点；
3. 以机器人 footprint + Profile margin 做 clearance；
4. 检查局部 costmap 和障碍膨胀；
5. 调 Nav2 `ComputePathToPose` 验证从一个或多个已知起点可达；
6. 用相机 FOV、遮挡和对象覆盖评分选择 observation vantage；
7. 从覆盖目标计算 scan headings，而不是固定四个角度；
8. 保存验证版本、地图哈希、路径代价和 EvidenceRefs。

模型可以排序候选和解释用途，但不能提交一个未经采样/验证的坐标。

### 8.4 路线

SceneRevision 保存房间/Portal 拓扑和可用 Anchor，不保存唯一固定“走遍所有办公室”路线。
Mission Compiler 根据当前起点、被请求房间、已完成节点和 Nav2 代价生成访问顺序；Nav2
仍负责每一段实际路径和实时障碍。

## 9. Review 与发布

PWA Review 页面至少展示：

- 2D occupancy/ESDF 与 3D PLY/3DGS；
- room/door/connectivity/anchor proposal；
- 每个候选的来源、置信度、clearance、可达性和视觉证据；
- FARM/DAAAM 结果 diff；
- Scene Explorer 低覆盖和异常；
- Reviewer 的接受、编辑、拒绝与理由；
- 新旧 revision diff 和回滚目标。

第一阶段只允许：

```text
draft → proposed → reviewed → published
```

人可以在 UI 修正候选，但不能要求开发者回到 Python/YAML 手填坐标。Reviewer 的决定交给
独立 Scene Publisher/Registry 复核权限、状态和验证摘要；Publisher 才产生不可变
SceneRevision、签名和完整 provenance。旧 revision 保留供历史 Mission 和回滚使用。
`draft/proposed` 可含候选，`published` 只含 approved anchors。

## 10. Runtime 使用方式

Mission Architect 输出：

```text
inspect room entity:meeting-room
constraints: inside + multi-direction observation + final report
```

它不输出 `x/y/yaw`。Field Executive：

1. 固定当前 published SceneRevision；
2. 从当前 published revision 的 AnchorRegistry 选择满足角色和约束的 approved anchor；
3. Local Goal Resolver/Live Visual Resolver 结合 approved anchor、当前 Pose、costmap、SLAM
   或新鲜 RGB-D/TF 解析目标；
4. Field Executive 复核并绑定当前 Mission、Robot Boot、Localization Epoch、
   SceneRevision 和短 TTL，形成 `live_execution_target`；
5. 通过 Robot Agent 调 Nav2；
6. 到达后用 region/视觉证据验证，再扫描和报告。

“去刚才看到的椅子旁边”使用历史记忆查候选 region/entity，但最终接近点仍由 Live Visual
Resolver 基于新鲜 RGB-D/TF 生成；历史 centroid 不直接进入 Nav2。

## 11. 从当前 Office V2 迁移

### 阶段 A：Legacy 适配

- 把当前 Office catalog 投影为 `source=legacy_reviewed` SceneRevision；
- 现有 Mission 行为不变；
- 给所有 Anchor 增加 provenance 和 revision identity。

### 阶段 B：Shadow Compiler

- 对 Office V2 录制 posed RGB-D；
- FARM/DAAAM 分别生成 shadow result；
- Compiler 生成 proposal，但不影响现有任务；
- 与人工 catalog 比较 room/door/anchor 差异。

### 阶段 C：新场景验收

- 新建完全没有 region/anchor/route 常量的第二场景；
- 自动采集、生成 proposal、人工一次审核并发布；
- 使用同一通用 Place/Inspection Compiler 完成中文巡检。

### 阶段 D：切换生产数据源

- Office compiler 改为查询 bound SceneRevision；
- `office_v2_bootstrap` 仅作为 regression fixture；
- 移除 Control API 对静态 Office catalog 的运行依赖；
- PCD/Potree 等旧链在新 Viewer/资产链验收后独立删除。

## 12. 防止“换个地方继续硬编码”

CI/验收应检查：

- 新场景没有提交 room/entry/station/route 坐标文件；
- Control API/Mission Architect 不含新场景名称到坐标的映射；
- Provider 原生 ID/centroid 不能进入 SkillRequest；
- 每个导航 Target 都能追溯到 SceneRevision + geometry validation 或 live visual result；
- 删除/关闭 FARM 或 DAAAM 不影响 Legacy stable Mission；
- 替换 Provider 不改变 MissionBundle、WorldDelta 和 SceneRevision 外部 Schema；
- 同一输入和 Compiler 版本产生相同 proposal hash，人工修改作为显式 revision 保存。

只有通过第二个无常量场景的完整验收，才能宣称地图硬编码问题已解决。

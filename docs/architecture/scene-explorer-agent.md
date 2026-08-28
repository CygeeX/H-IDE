# Scene Explorer Agent：PLY/3DGS 自动场景审计与标注

| 属性 | 值 |
| --- | --- |
| 状态 | Proposed / feature branch design |
| 设计分支 | `feature/scene-explorer` |
| 目标发行 | 在下一次 breaking release 中移除 PCD/Potree 后合入 |
| 运行形态 | 离线优先、异步、observation-only sidecar |
| 当前版本 | `MapAuditJob v1`、`SceneAsset v1`、`MapAuditAnnotation v1` |

本文是给 H-IDE、视觉、图形、Agent、Robot Agent 和测试开发者的共同契约。它
描述为什么要做、每个服务负责什么、如何接入现有联调、应该参考哪些开源项目，
以及第一版何时算完成。

## 0. 结论先行

这个模块可以设计出来，而且适合成为独立的 `scene-explorer` 服务。它不是让
大模型直接“看 PLY 并输出坐标”，而是一条受约束的证据流水线：

```text
PLY / 3DGS asset
        │ 解析、单位/坐标校验、内容哈希
        ▼
统一 SceneAsset / GaussianScene
        │
        ├── 离屏 RendererAdapter
        │       └── RGB + Depth + Visibility + CameraMetadata
        │
        ├── 确定性 ExplorerPlanner
        │       └── 环绕、Top/Front/Side、室内漫游、frontier、异常补拍
        │
        ├── CoverageAnalyzer
        │       └── 覆盖率、盲区、遮挡、密度、孔洞、尺度/单位异常
        │
        └── MultimodalProvider（默认 GPT-5.6 Sol profile）
                └── 跨视角语义、异常解释、标注候选、报告文字
                              ▼
              MapAuditReport + proposed AnnotationSet
                              ▼
                    Viewer 审核、版本化发布
```

第一版不让 Explorer 直接控制机器人、Nav2、SONIC 或 MuJoCo。未来需要真实相机
补拍时，只能通过现有 Robot Agent/Nav2 的受限 observation waypoint 流程申请，
Explorer 本身永远没有 `/cmd_vel`、ROS 控制权或 Robot Agent 凭据。

## 1. 当前代码事实与设计前提

### 1.1 当前 Skillstore 并没有可直接复用的 3DGS/PLY 产品 Viewer

事实核对显示，当前仓库的 Scene 路径仍是：

- `components/h-ide/services/scene_api/` 只接受 PCD 上传和场景任务；
- `components/h-ide/services/scene_worker/pipeline.py` 使用 PCL/PDAL/Open3D，
  生成 analysis PLY、LAZ 和 Potree 产物；
- `components/h-ide/apps/control-pwa/src/scene-workspace.tsx` 的上传入口只接收
  `.pcd`；
- `components/h-ide/apps/control-pwa/src/potree-canvas.tsx` 和
  `potree-runtime.ts` 加载 Potree metadata；
- 当前前端的点选、测量和多边形覆盖不能解析标准 Gaussian-splat PLY。

因此，不能在开发文档里把当前实现描述成“已经支持任意 PLY/3DGS”。本模块需要
先定义 RendererAdapter，并实现可测试的 PLY 后端，再接入现有或外部 3DGS Viewer。
如果团队另有未纳入 Skillstore 的 3DGS Viewer，必须提供仓库地址、提交、接口和
许可证，才能登记为一个 adapter。

### 1.2 可复用的现有能力

- `packages/scene_core/models.py` 已有 `SemanticRegion`、场景 revision、bounds
  和 annotation 概念，可作为发布层参考。
- `services/control_api/scene_memory.py` 已有 `stopped → captured → named →
  confirmed` 的场景教学状态机，可复用其“候选不等于可信事实”的原则。
- `components/h-ide-vision` 已有 Qwen 单帧语义、SAM2 分割、RGB-D/TF 几何和
  evidence 记录，但尚无多视角批处理、coverage 或地图质量报告。
- `components/h-ide-agent` 已有规划 Supervisor/Worker 和 A2A 边界，但其
  `PlanFragment` 是动作规划契约，不适合伪装成地图审计报告。

### 1.3 明确的 breaking 变化

本设计确认废除 `PCD → Potree`：不做旧 PCD 数据迁移，不保留运行时兼容分支，
不把旧 PCD 数据重新解释为 PLY。当前冻结版 `main` 和冻结标签仍作为回退点；
新功能在 `feature/scene-explorer` 完成验收后，以一个标明 breaking 的版本删除
旧路径。

## 2. 目标与非目标

### 2.1 目标

1. 接收经过校验的 PLY 和已支持的 3DGS 资产。
2. 自动生成有限预算的多视角相机轨迹。
3. 输出 RGB、Depth、可见性和相机元数据，使每个判断可回溯。
4. 识别房间、门、家具、设备、障碍物和地图异常。
5. 生成可编辑、可比较、可回滚的 3D 标注候选。
6. 生成地图质量报告，而不是只返回一句模型总结。
7. 在没有 GPU、模型凭据或部分视角失败时明确降级，不伪造成功。
8. 不改变现有 Mission V2、Nav2、SONIC 和 Robot Agent 的安全边界。

### 2.2 非目标

- 第一版不做实时 SLAM、重建原始 3DGS、不替代 FAST-LIO2/GTSAM。
- 第一版不让 LLM 规划机器人路径或直接生成导航坐标。
- 第一版不通过 MuJoCo/真实机器人移动来完成相机探索。
- 不承诺“所有对象都被识别”；语义召回率必须在有标注的评测集上单独统计。
- 不复制 FARM-Project 的 AGPL 代码、受限权重或数据集到主仓库。

## 3. 术语和信任边界

| 术语 | 定义 |
| --- | --- |
| `SceneAsset` | 已校验的统一场景资产描述，不等于原始文件 |
| `GaussianScene` | 将 3DGS 的中心、尺度、旋转、颜色和不透明度规范化后的中间表示 |
| `CameraView` | 一个相机位姿、内参和渲染设置的不可变记录 |
| `EvidenceFrame` | 一组 RGB/Depth/visibility 与其 `CameraView` 的证据 |
| `hypothesis` | 模型或几何算法的待验证假设，不能直接进入可信点位 |
| `proposed` | Agent 生成、等待人工审核的标注 |
| `published` | 经过权限检查并绑定到场景 revision 的标注 |
| `navigation_anchor` | 可能供导航使用的标注，发布门槛高于普通 object/room |
| `coverage` | 几何分析器对可观测表面/体素的覆盖计算，不由 LLM 自报 |

信任顺序固定为：原始资产与元数据 > 确定性几何结果 > 视觉模型假设 > 自然语言
解释。上层证据不能被下层文字覆盖。

## 4. 系统架构

### 4.1 服务分层

```text
PWA Scene Workspace
        │ same-origin Control API
        ▼
H-IDE MapAudit API / job repository
        │ internal bearer + idempotency key
        ▼
scene-explorer sidecar
  ├── AssetAdapter（PLY / 3DGS）
  ├── RendererAdapter（headless EGL）
  ├── ExplorerPlanner（确定性）
  ├── CoverageAnalyzer（确定性）
  ├── MultimodalProvider（GPT-5.6 Sol profile）
  ├── AnnotationProposalStore
  └── Report/Evidence Store
        │ report + proposed annotations only
        ▼
H-IDE Scene revision / review / publish
        │ 仅 published navigation_anchor 可转换
        ▼
SceneMemory → 未来的 Mission/Robot Agent/Nav2
```

`scene-explorer` 不加入当前运动服务的 systemd unit，不读取 SONIC telemetry，
不接收 Robot Agent token。它可以和 H-IDE 共用对象存储，但使用独立 job、日志和
内部 token。

### 4.2 与现有服务的关系

| 现有服务 | 接入方式 | 禁止事项 |
| --- | --- | --- |
| H-IDE Control API | 创建/查询/取消 MapAuditJob、转发审核操作 | 不让浏览器直接访问 Explorer |
| h-ide-agent | 只提取自然语言参数、选择 MapAuditJob route | 不把报告当 PlanFragment，不发运动技能 |
| h-ide-vision | 可选复用 Qwen/SAM2 provider contract | 不让 Vision 决定路径/发布点位 |
| Scene API | 保存资产 manifest、annotation revision、发布记录 | 不把未经审核候选写成可信事实 |
| PWA | 展示任务进度、证据、标注编辑和 diff | 不用浏览器截图代替审计 RGB/Depth |
| Robot Agent/Nav2 | 第一版无连接；未来只接受受限 observation waypoint | Explorer 无 `/cmd_vel`、ROS graph 或租约 |
| MuJoCo/SONIC | 第一版无连接 | 不将虚拟相机探索误报为机器人已完成移动 |

## 5. 统一资产契约：`SceneAsset v1`

### 5.1 必填字段

```json
{
  "schemaVersion": "scene-asset/v1",
  "assetId": "asset_01J...",
  "source": {
    "kind": "ply|gaussian_splat",
    "originalFilename": "office.ply",
    "sha256": "64-lowercase-hex",
    "byteSize": 123456789,
    "ingestedAt": "2026-08-28T00:00:00Z"
  },
  "coordinate": {
    "frame": "scene",
    "handedness": "right",
    "unit": "m",
    "axisConvention": "x-forward,y-left,z-up",
    "sceneToMap": null,
    "registration": {
      "status": "unknown|estimated|confirmed",
      "translationErrorM": null,
      "rotationErrorRad": null
    }
  },
  "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]},
  "attributes": {
    "hasColor": true,
    "hasNormal": false,
    "hasCameraPoses": false,
    "hasRgbFrames": false,
    "hasDepthFrames": false
  },
  "provenance": {
    "producer": "FAST-LIO2|GTSAM|3dgs|other",
    "producerVersion": "optional",
    "metadataHash": "optional"
  }
}
```

`frame=scene` 是 Explorer 的唯一内部坐标系。所有标注、相机、覆盖率和报告都
必须写明 frame 和单位。`sceneToMap` 未 `confirmed` 时，不得发布导航 anchor。

### 5.2 PLY 解析规则

PLY 不是单一语义格式，必须先识别它属于普通点云还是 Gaussian-splat PLY：

- 普通点云至少需要有限的 `x/y/z`；颜色和法线可选但影响证据等级。
- Gaussian-splat PLY 常见属性为 `x/y/z`、`f_dc_*`、`opacity`、`scale_*`、
  `rot_*`。解析器将其转换为中心、协方差、RGB/alpha，并保留原始属性哈希。
- 缺少颜色、相机位姿或深度时，不得静默补成真实采集数据；必须将证据等级标成
  `synthetic_render` 或 `geometry_only`。
- 不允许把 PLY 改名为 PCD 绕过校验，也不允许加载外部 URL、脚本或未授权资源引用。

### 5.3 安全校验

AssetAdapter 在隔离 worker 中执行：

1. 校验 MIME、扩展名、文件大小、顶点/属性上限和有限数值。
2. 拒绝压缩包路径穿越、符号链接逃逸、任意脚本和外部 URI。
3. 对 `scale`、`opacity`、旋转四元数和包围盒做范围检查，拒绝 NaN/Inf。
4. 解析后写入只读规范化资产，计算 `normalizedSha256`。
5. 解析失败生成可诊断的 `asset_invalid` 终态，不调用模型、不创建标注。

## 6. RendererAdapter 契约

### 6.1 对外接口

渲染后端可以不同，但只能通过以下接口被 Explorer 使用：

```text
load_asset(SceneAsset) -> RendererSession
get_scene_bounds() -> Bounds
estimate_ground_and_axes() -> GeometryHypothesis
render_view(CameraView) -> RenderedView
visibility_for_view(CameraView) -> VisibilitySummary
close() -> None
```

`RenderedView` 至少包含：

```json
{
  "viewId": "view_00042",
  "rgbPath": "object-store://.../rgb.jpg",
  "depthPath": "object-store://.../depth.z16",
  "visibilityPath": "object-store://.../visibility.bin",
  "camera": {
    "frame": "scene",
    "position": [1.0, 2.0, 1.6],
    "quaternionWxyz": [1, 0, 0, 0],
    "intrinsics": {"fx": 600, "fy": 600, "cx": 320, "cy": 240},
    "width": 640,
    "height": 480
  },
  "depthSemantics": "metric|relative|synthetic",
  "sourceKind": "provided_camera|reconstructed_camera|synthetic_camera",
  "rendererVersion": "ply-open3d-egl-v1"
}
```

### 6.2 第一后端：PLY 几何离屏渲染

第一版使用 Open3D/等价 EGL 离屏实现几何正确性和 metric depth，允许 CPU fallback。
它不依赖桌面 Viewer，不把浏览器 WebGL 截图当作证据。需要高质量 splat 视觉时，
再接入独立的 3DGS backend（例如 gsplat 或专用 WebGPU/SIBR adapter）。

### 6.3 交互 Viewer

交互 Viewer 只消费 `RenderedView`、标注和 camera metadata。它可以复用现有前端
相机控制思想，但不能继续依赖 Potree runtime。Viewer 必须支持：

- 视点列表/轨迹回放；
- Top/Front/Side/orbit 和室内路径预览；
- RGB、Depth、visibility 三层切换；
- 点击/框选/多边形标注；
- proposed/published/rejected 图层；
- 证据视角跳转和 annotation diff。

## 7. 自动探索规划器

### 7.1 四阶段流程

| 阶段 | 输入 | 主要动作 | 输出 |
| --- | --- | --- | --- |
| coarse coverage | bounds、地面假设 | Top/Front/Side、环绕采样 | 初始视点集 |
| room traversal | 地面/占用估计 | 室内可行区域、走廊/房间路径插值 | 路径视点 |
| frontier refinement | coverage/visibility | 优先观察低覆盖、遮挡和未知区域 | 补拍视点 |
| anomaly recapture | 密度/孔洞/模型疑点 | 局部多 yaw、近景和反向视角 | 异常证据集 |

视点规划完全由确定性代码完成。推荐使用固定 seed、排序规则和预算，使同一
资产与 profile 生成相同的 `viewPlanHash`。

### 7.2 可复现候选评分

每个候选视点计算：

```text
gain = 新覆盖表面/体素
      - 遮挡惩罚
      - 与已选视点的冗余惩罚
      + frontier/异常优先级
```

候选排序使用 `(phase, -gain, distance, stable_view_id)`，禁止依赖 Python dict
偶然顺序或模型输出。室内路径只能在点云推导的占用/可行区域内生成，不使用评测
数据集的 GT object center。

### 7.3 预算和停止条件

默认 profile `map-audit-default-v1` 至少锁定：

- 最大视点数；
- 最大渲染时长；
- 最大 VLM 批次数和图像数；
- 最大单帧分辨率与总存储量；
- 最小新增覆盖增益；
- GPU 显存/并发上限。

达到任一上限就停止并生成 `incomplete_budget` 报告。只有 CoverageAnalyzer 判定
覆盖/盲区指标达标，任务才可以标记 `completed`；“模型觉得看完了”只能写入解释。

### 7.4 没有相机轨迹时

几何估计出的相机必须标记：

- `provided_camera`：输入明确提供并通过校验；
- `reconstructed_camera`：由注册/几何算法重建，有误差；
- `synthetic_camera`：仅为 Viewer/离屏探索生成。

这三种证据不能混合统计。没有真实 RGB 的场景可以做结构质量审计，但文字、门牌、
材质识别的结论必须降低置信度并在报告中警告。

## 8. 多模态模型适配器

### 8.1 模型职责

默认 profile 使用 GPT-5.6 Sol（部署可用时），通过 `MultimodalProvider` 注入。模型
只接收精选关键帧和有限 metadata，负责：

- 跨视角合并同一对象；
- 生成 ontology caption/category；
- 解释几何异常；
- 指出可能需要补拍的区域；
- 生成报告自然语言摘要。

模型不能：

- 直接写 `sceneToMap`、导航坐标或 `/cmd_vel`；
- 修改 coverage 数值；
- 把旧记忆当作当前场景事实；
- 跳过资产/证据校验或直接发布 annotation。

### 8.2 批请求和严格输出

```json
{
  "schemaVersion": "map-audit-vlm-input/v1",
  "jobId": "job_01J...",
  "assetHash": "...",
  "views": [
    {
      "viewId": "view_00042",
      "rgb": "object-store://...",
      "depthSummary": {"kind": "metric", "minM": 0.2, "maxM": 8.0},
      "camera": {"position": [1, 2, 1.6], "frame": "scene"}
    }
  ],
  "ontologyVersion": "office-v1",
  "promptVersion": "scene-understanding-v1"
}
```

模型只返回严格 JSON：对象/房间候选、证据 view ID、2D 区域、可选 3D hypothesis、
置信度、疑似异常和补拍理由。使用 provider 的 structured-output 校验；解析失败
不重试非幂等动作，只记录 `vlm_schema_error` 并继续生成部分报告。

报告记录 provider、model、temperature、prompt/schema 版本、请求/响应 hash 和
调用时间。几何报告的可复现性不依赖模型；语义结果即使因在线服务变化不同，也
必须能够定位到相同的证据帧。

## 9. 标注模型与审核流程

### 9.1 `MapAuditAnnotation v1`

```json
{
  "annotationId": "ann_01J...",
  "revisionId": "scene-rev-7",
  "type": "room|object|geometry|navigation_anchor|quality_issue",
  "ontologyId": "office.desk",
  "rawCaption": "办公桌",
  "geometry": {
    "kind": "point|aabb|obb|surface_mask|room_polygon",
    "frame": "scene",
    "unit": "m",
    "value": {}
  },
  "visibility": {"evidenceViewIds": ["view_00042", "view_00057"]},
  "confidence": {"semantic": 0.91, "geometric": 0.84},
  "status": "proposed|in_review|accepted|rejected|published",
  "provenance": {
    "assetHash": "...",
    "viewPlanHash": "...",
    "vlmResponseHash": "...",
    "source": "geometry|vlm|human"
  },
  "review": {"actorId": null, "reason": null, "reviewedAt": null}
}
```

### 9.2 标签本体

基础 ontology `office-v1`：

```text
room, door, desk, chair, table, shelf, computer, person,
obstacle, corridor, entrance, unknown
```

模型可以保留任意原始 caption，但发布 annotation 必须映射到版本化 ontology ID。
同义词映射要显式记录，不允许静默把 `desk` 改成 `table`。

### 9.3 状态机

```text
proposed → in_review → accepted → published
      └──────────────→ rejected
published ──(新修订)──► proposed
```

模型标注与人工标注冲突时不覆盖任何原记录，创建新的 revision 并记录选择理由。
普通 object/room 可以单人审核；`navigation_anchor` 需要场景管理员或双人审核，
且要求 `sceneToMap=confirmed`、几何误差达标和入口安全检查通过。

### 9.4 Viewer 操作

第一版必须支持：单项编辑、改名、移动几何、合并、删除、批量接受/拒绝、冲突 diff、
视角跳转、发布和回滚。每个操作产生审计事件；发布不能修改历史 revision。

## 10. `MapAuditJob v1` API

### 10.1 REST/SSE 接口

```text
POST   /api/v1/map-audits
GET    /api/v1/map-audits/{jobId}
GET    /api/v1/map-audits/{jobId}/events?after=<eventId>
POST   /api/v1/map-audits/{jobId}/pause
POST   /api/v1/map-audits/{jobId}/resume
POST   /api/v1/map-audits/{jobId}/cancel
GET    /api/v1/map-audits/{jobId}/report
GET    /api/v1/map-audits/{jobId}/annotations
POST   /api/v1/map-audits/{jobId}/annotations/{revisionId}/review
POST   /api/v1/map-audits/{jobId}/annotations/{revisionId}/publish
```

浏览器只访问 H-IDE same-origin API；Explorer 的内部接口使用独立 bearer token。
创建请求必须携带 `Idempotency-Key`，相同 key 和相同资产 hash/config 返回同一 job，
相同 key 但内容不同返回 `409`。

### 10.2 Job 状态

```text
created
  → validating_asset
  → planning_views
  → rendering
  → scoring_coverage
  → vlm_batching
  → aggregating
  → awaiting_review
  → published | completed | incomplete_budget | failed | cancelled
```

每阶段 checkpoint 至少记录：输入 hash、配置/profile hash、已完成 view IDs、VLM
批次 hash、覆盖指标、错误码和下一步 cursor。服务重启后从 checkpoint 恢复；不重放
未知副作用，不自动发布。

### 10.3 进度事件示例

```json
{
  "eventId": 118,
  "jobId": "job_01J...",
  "type": "coverage_updated",
  "stage": "scoring_coverage",
  "progress": {"views": 42, "rendered": 42, "coverageRatio": 0.78},
  "warnings": ["north-east room has low visibility"],
  "checkpointHash": "..."
}
```

## 11. Agent 工具层

这些是 Explorer 内部的 typed tools，不是让 LLM 自由执行的 shell 工具：

| 工具 | 权限 | 结果 |
| --- | --- | --- |
| `load_scene_asset` | 只读输入 | `SceneAsset` |
| `inspect_scene_quality` | 确定性 | 密度/孔洞/尺度/单位报告 |
| `plan_exploration_views` | 确定性 | `viewPlanHash` + CameraView 列表 |
| `render_observation` | 只读渲染 | RGB/Depth/Visibility evidence |
| `score_coverage` | 确定性 | 覆盖/盲区/遮挡指标 |
| `submit_vlm_batch` | 外部模型调用 | 严格语义 hypotheses |
| `aggregate_report` | 结构化聚合 | `SceneInspectionReport` |
| `propose_annotations` | 生成草案 | `MapAuditAnnotationSet(proposed)` |
| `review_annotation_set` | 人工权限 | accepted/rejected revision |
| `publish_annotation_revision` | 管理员权限 | immutable published revision |

`review`/`publish` 只能由服务端用户权限调用，永远不放进模型工具列表。Explorer
工具层无 Robot Agent credentials。

## 12. `SceneInspectionReport v1`

报告至少包括：

```text
reportId / jobId / assetHash / normalizedAssetHash
scene frame / unit / registration status and error
renderer + explorer + coverage + ontology + prompt versions
viewPlanHash / selected view IDs / camera source kinds
surface/voxel coverage ratio / unknown ratio / blind regions
point density / holes / outlier / scale-unit warnings
occlusion and overlap summary
per-object and per-room evidence links
proposed annotation revision
degraded flags / failure codes / budget usage
```

数值由确定性分析器生成；模型只填解释和建议。报告可逐项跳转到 RGB、Depth、
visibility、相机位姿和点云区域。

## 13. 与现有联调接入步骤

### Phase A：先不改运动链

1. 在 `feature/scene-explorer` 新增 sidecar、schema、内部 token 和 job repository。
2. H-IDE 增加 Scene Workspace 的 MapAuditJob 页面和 SSE 进度展示。
3. h-ide-agent 增加“分析上传地图/生成报告”确定性 route；它只提交参数：
   `assetId、profileId、ontologyVersion、privacyPolicy`。
4. Explorer 运行在独立 Compose profile 或 host GPU process，不加入现有 motion
   systemd 启动序列。
5. 先用离线 PLY fixture 和合成相机验证，不连接 Robot Agent、Nav2、SONIC。

### Phase B：接入现有 SceneMemory

1. Explorer 将 `proposed AnnotationSet` 回传 H-IDE。
2. H-IDE 显示证据并走现有的确认/审核状态机。
3. 发布动作创建新的 scene revision，不覆盖旧 revision。
4. 只有管理员发布的 `navigation_anchor` 才可转换为当前 SceneContext 的可用事实。
5. `sceneToMap` 未确认时，anchor 保持 `executionEligible=false`。

### Phase C：未来真实相机补拍（不属于第一版）

1. H-IDE 把“补拍某视点”编译成受限 observation Mission。
2. Robot Agent/Nav2 执行 bounded waypoint 并回传 RGB-D/TF。
3. Explorer 只消费证据，不获得底盘控制权。
4. 任何定位 epoch、权限或 freshness 失败都终止补拍，不影响已有 report。

## 14. PCD/Potree 完全移除清单

在新链路通过验收后，以 breaking release 一次完成：

- 删除 H-IDE PCD 分片上传、PCD manifest 和 PCD MIME 校验；
- 删除 Scene Worker 的 PDAL/PCD reader、LAZ/PotreeConverter 构建步骤；
- 删除 PWA `potree-canvas`、Potree runtime、PCD 上传控件和旧测量 API；
- 删除 PCD/Potree 专用数据库字段、API schema、fixture 和验收测试；
- 更新 Compose、Dockerfile、README、启动手册和第三方 notices；
- 不读取、不迁移旧 PCD 数据；旧冻结标签作为唯一回退路径；
- 在 release note 中明确：旧 PCD API 返回 `410 Gone` 或不再存在，不做隐式转换。

删除前的门禁是：新 Viewer 能完成资产加载、相机探索、报告和标注审核；否则不得
先删旧路径造成不可用的半成品。

## 15. 开源参考与许可证边界

| 项目 | 可借鉴内容 | 本项目处理方式 |
| --- | --- | --- |
| Open3D | 点云几何、法线、离屏/CPU 分析 | 作为可审查依赖，锁定版本；不把浏览器截图当证据 |
| gsplat / 专用 3DGS renderer | Gaussian 属性解析和 splat 渲染 | 独立 adapter，逐项审核许可证和 CUDA 兼容性 |
| Viser / SIBR | 相机动画、splat/box/轨迹交互 | 只借鉴交互思想或隔离 adapter |
| FARM-Project | object memory、covisibility、minimum covering views、LLM caption | AGPL-3.0 及其第三方限制；不直接复制，必要时隔离进程 |
| Nav2 | 生命周期、路径/动作分层和 bounded waypoint 思想 | 只在未来 observation Mission 使用，Explorer 不成为导航 owner |
| ROS 2/CycloneDDS | 消息/坐标/时间戳契约 | 不把离线 Explorer 接进仿真或 live DDS domain |
| DAAAM | 异步空间语义记忆思想 | 保持 post-Mission observation-only，不参与实时探索控制 |

每个新依赖必须记录 URL、完整 commit/version、许可证、内容/二进制哈希和
redistribution 状态。AGPL、非商业、专有模型和数据集不能因为“只是参考代码”而
混入闭源运行时。

## 16. 工作分配建议

| 工作包 | 主要负责项目 | 交付物 | 前置 |
| --- | --- | --- | --- |
| W0 契约/安全 | H-IDE platform + QA | SceneAsset、Job、Annotation、权限 schema | 无 |
| W1 资产适配 | scene-explorer graphics | PLY parser、GaussianScene、hash/limits | W0 |
| W2 离屏渲染 | graphics | PLY EGL backend、3DGS adapter、RenderedView | W1 |
| W3 探索规划 | geometry/agent | 四阶段 planner、coverage/frontier、budget profile | W1/W2 |
| W4 多模态聚合 | h-ide-vision + Agent | provider adapter、严格 JSON、跨视角聚合 | W2/W3 |
| W5 标注工作台 | H-IDE PWA/Scene API | 图层、编辑、diff、审核、发布/回滚 | W4 |
| W6 对话接入 | h-ide-agent | MapAudit route、参数抽取、非 Mission 语义 | W0/W4 |
| W7 联调/验收 | integration owner | fixture、L0-L3、故障注入、审计证据 | W0-W6 |
| W8 breaking cleanup | H-IDE platform | 删除 PCD/Potree 运行时和文档 | W7 通过 |

任何人修改模型锁、坐标契约、SceneMemory 或导航 anchor 都必须同时更新 lock、
测试和回滚说明。W8 不能与 W1-W7 并行删除代码。

## 17. 验收标准

### L0：契约和安全

- 纯 PLY fixture 可解析；损坏属性、NaN、路径穿越和超限文件被拒绝；
- `SceneAsset`、`MapAuditJob`、`Annotation` schema 校验通过；
- 不含 `/cmd_vel`、Robot Agent token、旧 PCD/Potree runtime 依赖；
- 同一输入生成相同 normalized asset hash 和 viewPlanHash；
- `git diff`、secret scan、许可证 manifest 通过。

### L1：离线几何

- headless 环境完成 coarse/orbit/Top/Front/Side 视点规划和渲染；
- RGB/Depth/Visibility/camera metadata 成套保存；
- coverage、盲区、密度、孔洞和尺度异常由确定性测试覆盖；
- 超出预算生成 `incomplete_budget`，不会假报 completed；
- CPU fallback 明确标为 degraded。

### L2：语义与标注

- GPT-5.6 Sol（或替代 provider profile）只返回 schema 合法 hypotheses；
- 每个 object/room/issue 都能回溯至少一个证据 view；
- Viewer 支持编辑、批量审核、冲突 diff、revision publish/rollback；
- 未审核候选不会出现在可执行 SceneContext；
- privacy policy、retention、provider/model/prompt hash 均出现在报告。

### L3：与联调接入

- PWA 创建/取消/暂停/恢复 MapAuditJob，SSE 顺序不乱；
- h-ide-agent 的自然语言 route 不创建 Mission V2，不调用运动技能；
- Scene revision 发布后可由现有 SceneContext 读取，旧 revision 可回滚；
- 未来 observation waypoint 仅通过 Robot Agent/Nav2（第一版可不实现）；
- PCD/Potree breaking cleanup 后，旧 API 明确失败，新链路完整可用。

## 18. 示例：从一句话到发布标注

用户在 PWA 输入：

```text
分析这张办公室 3D 地图，自动看完整个场景，找出房间、桌椅和地图质量问题，给我标注草案。
```

正确流程：

1. Conversation route 识别为 `map_audit`，不创建 Mission。
2. H-IDE 创建 `MapAuditJob`，锁定 asset hash、ontology 和 privacy profile。
3. Explorer 校验 PLY/3DGS，生成 `SceneAsset`。
4. Planner 生成可复现的视点计划；Renderer 输出关键帧证据。
5. CoverageAnalyzer 报告东南房间低覆盖，追加 frontier 视点。
6. VLM 根据批量 RGB/Depth 证据提出 `office.desk`、`office.chair` 等候选。
7. PWA 以 `proposed` 图层显示标注、证据视角、置信度和质量警告。
8. 用户修改一个对象名称并拒绝一个误识别；系统创建新 revision。
9. 管理员发布 room/object 标注；navigation anchor 另走更高权限审核。
10. PWA 输出完整报告和 revision ID；不会声称机器人已经走过地图。

## 19. 实施前检查表

- [ ] 找到并登记真实的现有 3DGS/PLY Viewer，或接受先实现 PLY backend；
- [ ] 确认 Open3D/gsplat/Viser 等依赖的版本和许可证；
- [ ] 合入 SceneAsset/Job/Annotation schema 及 fixture；
- [ ] 建立 `scene-explorer` 独立运行 profile 和内部 token；
- [ ] 完成 headless renderer 与 deterministic planner；
- [ ] 完成 VLM strict output、evidence trace 和报告；
- [ ] 完成 PWA 审核工作台和 Scene revision 发布；
- [ ] 通过 L0-L3 后再删除 PCD/Potree；
- [ ] 用冻结标签验证回退路径；
- [ ] 更新 `integration/integration.lock.yaml`、SBOM、第三方 notices 和 release note。

## 20. 当前明确的开放风险

1. 用户所说的现有 3DGS Viewer 尚未在当前 Skillstore 文件树中定位；这是 W1/W2 的
   第一项输入，不应假设为已完成能力。
2. 纯 PLY 往往没有真实 RGB、Depth 和相机位姿，门牌/文字识别质量可能不足；报告
   必须显示证据等级，必要时要求原始 RGB-D/3DGS 相机资产。
3. GPT-5.6 Sol 的在线可用性、费用和输出稳定性不是仓库能完全冻结的；因此模型
   结果必须可替换、可追溯，几何指标不能依赖它。
4. 3DGS GPU renderer 的 CUDA/驱动差异可能导致像素级不一致；验收应比较几何/结构
   指标和证据 metadata，不要求跨 GPU 完全相同的像素。
5. FARM 等参考项目的许可证不能通过“只复制几段代码”规避；依赖审查必须先于
   代码合并。

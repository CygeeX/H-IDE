# 开源参考、许可证与 Build-vs-Buy

| 属性 | 值 |
| --- | --- |
| 状态 | Reference and legal decision baseline |
| 当前决策 | 先通过 Provider 隔离做集成与 benchmark，后续再决定履行开源义务、与全部权利人谈判许可或替代实现 |
| 重要声明 | 进程/容器隔离提高可替换性，但不会自动消除开源许可证义务 |

## 1. 不整体照搬某一个项目

H-IDE 需要组合多个已经验证的思想：

```text
Mission 下发与机器人侧 Client       ← Isaac Mission Dispatch
困难样本边云升级                     ← Sedna Joint Inference
离线组件、授权缓存、Store-and-Forward ← Greengrass / Azure IoT Edge
Cloud/Site/实时控制分层              ← Intrinsic
多机器人交通与基础设施               ← Open-RMF
对象关系记忆                         ← FARM
动态、时序、房间层级记忆             ← DAAAM / Hydra / Khronos
现场语义记忆 + 云 mLLM 升级          ← Stretch AI / DynaMem
几何与可通行性                       ← SLAM / nvblox / RTAB-Map / Nav2
```

这些项目解决的是不同问题；任何一个都不能直接替代完整 H-IDE 商业平台。

## 2. 云边与 Mission 参考

| 项目 | 可参考的思想 | 不应误用的地方 |
| --- | --- | --- |
| [NVIDIA Isaac Mission Dispatch](https://github.com/nvidia-isaac/isaac_mission_dispatch) | Mission 服务经 MQTT 下发，机器人 Mission Client 调 Nav2；状态持久化与低带宽通信 | 不提供 H-IDE 的自然语言、Entitlement 和世界模型 |
| [KubeEdge Sedna](https://github.com/kubeedge/sedna) | Joint Inference：Edge 处理简单样本，困难样本升级到 Cloud | 不能让模型路由器直接获得运动权 |
| [AWS IoT Greengrass V2](https://docs.aws.amazon.com/greengrass/v2/developerguide/offline-authentication.html) | 本地组件、离线认证缓存、IPC、消息存储转发 | AWS 云产品不是 Mission/机器人安全语义 |
| [Azure IoT Edge Offline](https://learn.microsoft.com/en-us/azure/iot-edge/offline-capabilities) | 初始同步后模块离线运行、本地认证、消息回填 | 不替代 Field Executive |
| [KubeEdge](https://kubeedge.io/) / [OpenYurt](https://openyurt.io/) | Edge 工作负载、配置缓存和断云继续运行 | Kubernetes 不是 Agent、Mission 或 World Model 架构 |
| [FogROS2](https://github.com/BerkeleyAutomation/FogROS2) | ROS 节点云端卸载、WireGuard 和图像压缩 | 不作为商业 Mission 协议，不把 ROS 图直接暴露公网 |
| [Open-RMF](https://github.com/open-rmf/rmf) | 多 Fleet、门、电梯、交通日程和 Fleet Adapter | 机器人达到 2–3 台且站点调度成为问题后再接入 |
| [Intrinsic Architecture](https://www.intrinsic.ai/architecture) | Cloud 开发/ML/数据，现场 IPC 负责机器人和实时控制 | 商业平台思想参考，不是可直接复制的开源实现 |
| [NVIDIA Fleet Command](https://www.nvidia.com/en-us/data-center/products/fleet-command/) | 签名容器、OTA、GPU Edge 应用生命周期 | 不负责 Mission 语义和具身认知 |

InOrbit/Formant 可作为 RobOps、遥测、Teleop 和现场 Agent 的 Build-vs-Buy 参考，但不
应替代 H-IDE 的差异化 Mission、记忆和 Semantic Scene Compiler。

## 3. FARM

[FARM](https://github.com/GoldenGait/FARM-Project)从带位姿 RGB-D 构建开放词汇对象级
关系记忆。每个对象包含统计 3D geometry、稀疏 voxel、caption、多模态 embedding、
观察视角和共视关系，可根据“货架左边的梯子”这类关系语言检索对象。

它对 H-IDE 的价值：

- 大场景对象实例记忆；
- 自然语言关系查询的候选检索；中文能力需要 H-IDE alias/translation 层和自有数据评测，
  不能从上游英文示例直接推定；
- 记录最佳观察视角和对象共视关系；
- 为 Scene Compiler 提供对象、关系和区域候选。

它不提供：

- 自主探索控制；
- SLAM、占据图或 Nav2 安全目标；
- Mission 执行和机器人接口；
- 照片级 3DGS。FARM 中“Gaussian”是对象点分布的均值/协方差，不是带 SH、opacity、
  scale/rotation 的辐射场表示。

### 3.1 许可

FARM 仓库为 **AGPL-3.0-or-later**，并含需单独审计的模型和第三方组件。其原生
`scene_state.pt` 使用 PyTorch pickle，既不适合跨信任域，也不应成为 H-IDE 产品协议。

决策：

- **当前状态：H-IDE 集成度为 0。** 上游源码已下载，但没有运行中的 Provider、Router、
  数据库或 Compose 服务；
- 目标状态是在独立 CUDA 12.8/ROS 2 Humble 镜像中实现 Shadow Provider；
- 不复制 FARM 源码进 H-IDE 主仓库；
- Adapter 只输出 `MemoryProviderResult`，再由 H-IDE World Model Admission 产生
  canonical WorldDelta；
- 商业化前选择：遵守 AGPL；与 FARM 和所有相关依赖/权重权利人分别谈判所需商业许可；
  或使用 clean-room/替代 Provider。购买 Ultralytics 商业许可不会自动重许可 FARM 作者的
  AGPL 代码；官方仓库当前也没有公开承诺 FARM core 的商业双授权；
- 最终方案必须经过法律审核，不能把“网络服务/独立容器”等同于自动规避 AGPL。

FARM 官方完整栈约需 50GB GPU memory、最大场景约 65GB，并使用 CUDA 12.8 + ROS 2
Humble。该运行事实要求它与 DAAAM 使用独立镜像、网络和私有存储。

## 4. DAAAM / Hydra / Khronos

[DAAAM](https://github.com/MIT-SPARK/DAAAM)使用分割、跟踪、grounding 与 Hydra/
Spark-DSG/Khronos 构建动态 4D 场景记忆，可保存对象第一次/最后一次出现、时间变化、
对象—place—room 层级和区域摘要。

它对 H-IDE 的价值：

- 跨视角对象身份和时间记忆；
- 动态对象与场景变化；
- 房间/区域层级候选；
- “刚才/以前在哪里看到”的时序检索。

限制：

- 同样要求 RGB、metric depth、CameraInfo 和 TF/Pose；
- 区域聚类当前仍有 post-hoc 部分；
- 不包含 Nav2 执行或完整自主探索；
- 图中的对象坐标不能未经当前几何验证成为 Mission target。

**当前状态：** H-IDE 只有任务完成后的 observation-only YAML/JSON 导入器；没有常驻
DAAAM producer、live tracking 或 Shadow Provider。目标 v2 Shadow Provider 是新的独立
工作，不能与现有 `packages/semantic_memory/daaam.py` 导入契约混称为已实现能力。

DAAAM 官方要求 24GB+ NVIDIA GPU，推荐 Ubuntu 24.04/ROS 2 Jazzy；区域聚类仍含
post-hoc 阶段。它不能与 FARM 的 Humble/CUDA 环境拼入同一个 ROS workspace。

### 4.1 许可

DAAAM 核心和 DAAAM-ROS 为 BSD-3-Clause，但默认栈并不等于可直接闭源商用：

- Ultralytics、FastSAM、BoxMot/BotSort 含 AGPL；
- BotSort 是当前唯一已接入 Tracker，替换需要开发；
- DAM-3B 权重和 FoundationStereo 含非商业限制；
- TensorRT 等专有 Runtime 有独立再分发条款。

EfficientViT-SAM 等宽松组件可以替代部分依赖，但 Tracker、grounding、depth 和权重仍需
逐项替换或购买授权。

## 5. 其他语义记忆与几何参考

| 项目 | 价值 | 初步商业判断 |
| --- | --- | --- |
| [Stretch AI / DynaMem](https://github.com/hello-robot/stretch_ai) | 机器人 ROS Bridge + 外部 GPU + 本地语义记忆 + 不确定时云 mLLM + 主动观察 | 架构最接近 H-IDE 当前阶段；具体依赖仍需 SBOM |
| [ConceptGraphs](https://github.com/concept-graphs/concept-graphs) | posed RGB-D 构建开放词汇对象场景图 | MIT 核心，研究质量与模型依赖需评测 |
| [VLMaps](https://github.com/vlmaps/vlmaps) | 将语言目标映射到空间 | MIT；适合快速验证“语义目标不靠硬编码” |
| [Hydra](https://github.com/MIT-SPARK/Hydra) | objects/places/rooms/buildings 分层场景图 | BSD 系列核心，适合作为层级设计参考 |
| [Khronos](https://github.com/MIT-SPARK/Khronos) | 动态和长期变化的 metric-semantic map | ROS 2/产品成熟度需单独验证 |
| [nvblox](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox) | TSDF/ESDF、mesh、动态障碍与 costmap | Apache-2.0；适合作为 Edge 几何权威 |
| [RTAB-Map](https://github.com/introlab/rtabmap_ros) | RGB-D/双目/LiDAR SLAM、多会话定位与占据图 | BSD；商业构建注意可选受限特征组件 |

语义记忆和几何系统必须并存。FARM/DAAAM 回答“是什么、何时看到、和什么有关”，
SLAM/nvblox/Nav2 回答“现在在哪里、哪里可走、如何安全到达”。

## 6. Scene Explorer 参考组合

没有一个开源项目直接提供“PLY/3DGS → 自动看完整场景 → VLM 跨视角理解 → 地图质量
报告 → 可审核导航标注”的完整产品。建议组合：

- [Open3D](https://github.com/isl-org/Open3D)：点云、几何和离屏渲染；top-level MIT；
- [gsplat](https://github.com/nerfstudio-project/gsplat)：3D Gaussian rasterization；
  top-level Apache-2.0；
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)：Viewer 和相机路径；
  top-level Apache-2.0；
- [ActiveSplat](https://github.com/Li-Yuetao/ActiveSplat)：Next-Best-View/主动探索思想；
  top-level MIT，但 submodule、diff-gaussian、Habitat/MP3D/Gibson 数据需逐项审计；
- [Isaac Sim Replicator](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/index.html)：
  RGB、Depth、实例/语义标注和合成数据；属于 NVIDIA 专有软件/EULA，不是普通开源依赖。

以上仅是计划组合。团队已说明存在 PLY/3DGS Viewer，但当前 Release Manifest 未登记其
repo、commit、接口、格式、许可证或 SBOM；正式接入前必须补齐。所有参考项目都要 pin
commit/version；top-level 宽松许可证不代表其权重、数据集和 submodule 可直接商用。

Scene Explorer 保持异步、observation-only。纯 FAST-LIO2 PLY 若缺少颜色和原始相机帧，
VLM 很难识别门牌、显示器等语义；采集链必须同时保存 posed RGB-D。3DGS 渲染深度只用于
观察/审计，不替代 Nav2 碰撞几何。

## 7. 统一 Provider 边界

### 7.1 输入：`ObservationBatch`

```text
tenant/site/robot/boot/localizationEpoch/sensor/source/sourceMode
+ sceneCandidateId / observationSessionId / captureManifestHash
+ RGB encoding/width/height/URI/hash
+ metric depth encoding/width/height/unit/scale/invalid-value/URI/hash
+ intrinsics/distortion/image size/calibration hash
+ coordinate frame IDs/axis convention/handedness
+ 4x4 T_map_camera + T_base_camera
+ localization observedAt/36-value covariance/clock domain/time uncertainty
+ RGB-depth-TF maximum synchronization error
+ occupancy/mesh/map revision and hashes
+ frame capture timestamp and monotonic sequence
+ retention/privacy policy
```

### 7.2 输出：`MemoryProviderResult`

```text
provider/version/license manifest
+ providerEntityId/sourceEntityId / aliases / captions
+ centroid covariance / AABB / voxels / frame / unit
+ first/last observed / dynamic status
+ EvidenceRefs / track identity
+ relations / covisibility / region hypotheses
+ confidence / uncertainty / perf / peak VRAM
+ executionEligible=false
```

Adapter 不能分配 canonical `entityId`。World Model Admission 校验身份、坐标、时间和证据
后分配 `candidateId`，并按记录内容产生 `historical_observation` 或 `hypothesis` WorldDelta；
两者都不可执行。FARM 的 pickle、DAAAM 原生 DSG、Provider ID 和历史 centroid 不能直接
成为 Mission target。

## 8. Shadow Mode 与选择门

1. 用同一份 posed RGB-D/TF 数据离线回放两个 Provider；
2. 两个 Provider 写独立 shadow store，不写 Canonical World Model 的 published facts；
3. PWA 只展示候选、冲突和证据 diff；
4. 按统一指标评测；
5. 通过门槛的 Provider 最多获得通过 Admission 向 Semantic Scene Compiler 提交 proposal
   的资格；
6. 两个 Provider 都永远不是最终事实写入者；唯一发布权威是 Reviewer Policy + Scene
   Publisher/Registry。另一个 Provider 可继续作为 benchmark。

评测至少包括：对象 precision/recall、跨视角 ID switch、3D centroid/AABB 误差、房间归属、
关系查询 Recall@K、时序查询、吞吐/时延、峰值 GPU memory、重启恢复、错误隔离、中文别名
和许可证成本。

## 9. Build-vs-Buy 决策门

进入商业 Pilot 前，为每个 Provider 回答：

- 是否达到新场景基准，而不是只跑官方数据集；
- ARM64/Thor 或 Site Appliance 是否稳定；
- 是否能替换受限依赖和权重；
- 开源义务是否与 SaaS、Edge 分发和私有化模式兼容；
- 商业授权费用是否低于自研/维护成本；
- Provider 故障能否不影响 Mission 和停止能力；
- 客户数据是否会流向未批准的第三方服务；
- 是否能导出版本化、无 pickle 的 `MemoryProviderResult`，再由 H-IDE Admission 产生
  WorldDelta。

当前决定是“先验证，后选择”，不是“许可证以后再管”。从第一天就必须生成 SBOM、
记录源码 commit、模型权重、数据集和运行镜像摘要，才能保留选择权。

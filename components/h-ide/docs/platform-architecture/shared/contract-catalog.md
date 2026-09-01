# H-IDE 核心契约、权威与故障语义

| 属性 | 值 |
| --- | --- |
| 状态 | Contract design baseline |
| 目的 | 让同机 Demo、未来 Thor、Site Appliance 与 SaaS 使用同一边界 |
| 原则 | 模型输出是候选；物理事实、授权与执行状态必须可验证、可审计、可对账 |

## 1. 契约目录

| 契约 | 生产者 | 消费者 | 作用 |
| --- | --- | --- | --- |
| `MissionBundle` | Cloud Mission authoring/signing service | Field Executive | 完整、版本化、签名的任务 DAG 与策略 |
| `MissionEvent` | Field Executive | Site/Cloud/PWA | 节点状态、执行结果、失败和对账事件 |
| `EvidenceRef` | Vision/Robot/Scene workers | World Model、报告、审核 | 内容寻址的图像、深度、TF、模型与几何证据 |
| `ObservationBatch` | Observation Recorder | Memory Providers、World Model Admission | 不可变 posed RGB-D、时钟、标定和地图资产输入 |
| `MemoryProviderResult` | FARM/DAAAM adapters | World Model Admission / Shadow benchmark | Provider 私有 ID、候选语义、时序和关系，不是平台事实 |
| `SceneAuditResult` | Scene Explorer | World Model Admission / reviewer | PLY/3DGS 覆盖、质量、虚拟视角证据和标注候选 |
| `CapabilityManifest` | Registry / Edge runtime | Mission Architect、Field Executive | 已安装能力、版本、输入 Schema 和执行域 |
| `EntitlementGrant` | Cloud entitlement service | Field Executive/Site Brain | 某租户/站点/设备可使用的商业能力和离线租约 |
| `WorldDelta` | World Model Admission 或受权事实 Projector | Canonical World Model | H-IDE canonical candidate/entity 的观察、假设或受权事实增量 |
| `SceneRevision` | Scene Publisher/Registry | Field Executive/Mission Architect | 不可变房间、门、连通关系和已批准 Anchor |
| `ModelInvocation` | Cognition Broker | Local/Site/Cloud provider | 模型调用预算、隐私策略、证据和返回类型 |
| `SkillRequest/Result` | Field Executive / Robot Agent | Robot Agent / Field Executive | 有界、幂等的单个物理或认知技能调用 |

这些契约必须使用 H-IDE 自有、Provider-neutral 的 Schema。FARM、DAAAM、Qwen、
DeepSeek 或其他供应商只通过 Adapter 映射，不得把私有数据库或 pickle 文件变成跨端协议。

## 2. 观测、Provider 与 Admission

`ObservationBatch` 至少固定 tenant/site/robot/sensor/source、Robot Boot、Localization Epoch、
场景候选、地图资产 hash、clock domain、RGB/Depth 编码与尺寸、depth scale/invalid-value、
内参/畸变、`T_map_camera`、`T_base_camera`、36 元定位协方差、同步误差、逐帧 URI/hash 和
capture manifest hash。缺少这些字段的数据不能作为可重复的 FARM/DAAAM 输入。

身份分配严格分三步：

```text
FARM/DAAAM providerEntityId
        ↓ MemoryProviderResult
World Model Admission candidateId
        ↓ normalize / deduplicate / conflict retention
Canonical World Model entityId（合并或审核后绑定）
```

Provider 不能自行分配 H-IDE `entityId`。`MemoryProviderResult` 只携带
`providerEntityId/sourceEntityId`、来源版本、证据、geometry、时间、关系、置信度和许可
manifest。Admission 校验 Schema、坐标、时间、证据和租户作用域后，才产生 WorldDelta。

记录级分类固定为：传感器实际采集及其可验证时间/位姿可以形成
`historical_observation`；模型生成的 label、room assignment、关系或对象合并只能形成
`hypothesis`。两者都 `executionEligible=false`。FARM、DAAAM 和 Scene Explorer 永远不能
产生 `published_static_fact` 或 `live_execution_target`。

`SceneAuditResult` 还必须声明视角来源 `provided_camera|reconstructed_camera|synthetic_camera`
和证据等级 `captured_rgbd|synthetic_render|geometry_only`。虚拟视角不能证明机器人实际
到过该处，也不能证明真实可通行。

## 3. `MissionBundle`

建议最小结构：

```json
{
  "schemaVersion": "h-ide.mission-bundle/v1",
  "missionId": "mission_...",
  "revision": 3,
  "issuedAt": "2026-08-31T00:00:00Z",
  "expiresAt": "2026-09-01T00:00:00Z",
  "tenantId": "tenant_...",
  "siteId": "site_...",
  "robotSelector": {"robotId": "g1-01"},
  "sceneRef": {"sceneId": "office-a", "revision": 7},
  "requiredCapabilities": [],
  "nodes": [],
  "budgets": {},
  "offlinePolicy": {},
  "safetyProfile": "simulation|pilot|production",
  "signature": {"keyId": "...", "digest": "sha256:...", "value": "..."}
}
```

约束：

- Bundle 包含完整 DAG，不依赖云端逐节点遥控；
- 节点声明 `edge_required`、`edge_preferred` 或 `cloud_required`；
- Bundle 只能引用 `entityId`、`roomId` 和约束，不能携带模型猜测的路径或速度；
- 已开始 revision 的已完成/运行节点不可被后续 revision 原地改写；
- 签名覆盖 DAG、预算、场景、能力和离线策略。

## 4. `MissionEvent` 与对账

```json
{
  "schemaVersion": "h-ide.mission-event/v1",
  "eventId": "evt_...",
  "missionId": "mission_...",
  "revision": 3,
  "executionEpoch": "epoch_...",
  "nodeId": "inspect-room-2",
  "sequence": 41,
  "state": "accepted|running|succeeded|failed|deferred_cloud|unknown",
  "idempotencyKey": "...",
  "evidenceRefs": [],
  "occurredAt": "..."
}
```

`eventId` 和 `(executionEpoch, sequence)` 唯一。Site/Cloud 接收重复事件必须幂等；
缺号可以补拉，但不能根据缺失事件推测一个物理动作成功。重连后 Edge 报告当前快照和
最后序列，Cloud 只做对账，不重发已知或未知终态的物理节点。

## 5. `CapabilityManifest` 与商业权益

运行时有效能力不是“数据库里已购买”这么简单：

```text
EffectiveCapabilities
  = PurchasedEntitlements
  ∩ InstalledCapabilities
  ∩ SitePolicy
  ∩ RuntimeHealth
```

`CapabilityManifest` 描述设备实际安装与健康，`EntitlementGrant` 描述商业允许范围。
Mission 编写和 Edge 验证都必须计算交集；Cloud 不能因为购买记录存在，就假定机器人
已经安装模型、机械臂或传感器。

离线 Grant 使用签名租约和明确宽限期。租约过期时付费认知能力停止接受新任务，但以下
能力不进入付费墙：硬件急停、停止、取消、安全诊断，以及产品承诺并已安装的返航能力。

## 6. `WorldDelta`

每条 WorldDelta 必须携带：

- Admission 分配的 `candidateId`，以及已完成 canonical binding 时的 `entityId`；
- `kind`：`published_static_fact`、`historical_observation`、`hypothesis` 或
  `live_execution_target`；
- 来源 Provider、模型/算法版本、输入证据哈希；
- 坐标系、时间、Localization Epoch 和 TTL；
- 置信度与不确定性；
- 父 revision、幂等 ID 和冲突信息；
- 是否允许进入执行目标解析。

合法生产者：

| kind | 唯一允许的生产路径 | 可执行性 |
| --- | --- | --- |
| `historical_observation` | World Model Admission 从可验证的 Observation/Provider provenance 归一化 | 否 |
| `hypothesis` | World Model Admission 从模型/Provider 结果归一化 | 否 |
| `published_static_fact` | Scene Publisher 从已审核、已发布 SceneRevision 投影 | 仅作为稳定语义/Anchor 依据 |
| `live_execution_target` | Edge Local Goal Resolver/Live Visual Resolver 基于新鲜 SLAM、costmap、RGB-D/TF 和 approved anchor 生成 | 是；绑定 Boot/Epoch/Mission/SceneRevision 和短 TTL |

Cloud/Site 模型和通用 Worker 不能产生 `live_execution_target`。Scene Publisher 也不能把
DAAAM 动态 Track 或历史对象位置冻结成静态事实。

冲突规则：

1. 当前 Robot/SLAM 几何覆盖过期几何；
2. 已发布 SceneRevision 不被普通模型假设覆盖；
3. 新观察不会删除历史，只产生 supersede/tombstone 事件；
4. 云端长期合并结果不能反向改写 Edge 已执行的历史；
5. 多 Provider 对同一对象意见不一致时保留候选集，不能“最后写入者获胜”。

## 7. `SceneRevision`

一个可发布 revision 至少包含：

```text
SceneRevision
├─ coordinate contract / map hashes / localization provenance
├─ navigable regions and polygons
├─ rooms, corridors, doors and connectivity graph
├─ reviewed static semantic annotations and aliases
├─ approved observation-vantage anchors
├─ approved entrance/interior anchors
├─ clearance/reachability/visibility validation results
├─ EvidenceRefs and compiler version
└─ review decision, author, timestamp and signature
```

状态固定为：

```text
draft → proposed → reviewed → published → superseded
```

Semantic Scene Compiler 只产生 proposal；Reviewer/Policy 记录决定；独立 Scene
Publisher/Registry 复核状态、验证摘要和权限后签名发布。`draft/proposed` 可以包含候选，
`published` 只能包含 Reviewer 接受且绑定几何验证结果的 approved anchor。Field Executive
只能解析当前 published revision 中的 approved anchor。

生产 Profile 禁止模型从 `proposed` 直接跳到 `published`。仿真可配置自动批准高置信度
候选，但仍必须产生不可变 revision 和完整审核事件，不能静默写回原文件。

## 8. `ModelInvocation`

Cognition Broker 发送的请求不只是 Prompt，还要声明：

- 任务类型与允许的返回 Schema；
- 本地、Site、Cloud 候选 Provider；
- 最大时延、成本和 token/GPU 预算；
- 是否允许上传原始 RGB-D；
- 是否仅允许 keyframe/crop/mask/embedding；
- 需要的置信度与升级阈值；
- 当前 Entitlement、网络健康和数据驻留策略；
- 失败后的 fallback 或 `deferred_cloud` 语义。

Provider 返回值必须包含模型标识、版本、证据引用、置信度、耗时和费用维度。自然语言
文字可以用于解释，但执行链只消费通过 Schema 校验的字段。

## 9. 权威优先级

从高到低：

```text
硬件 E-stop / 驱动限制
  > Robot Runtime safety / SONIC balance / Watchdog
  > Field Executive execution policy
  > published SceneRevision + live geometry
  > Mission Architect
  > Site/Cloud model hypothesis
  > conversation memory / user-provided unverified coordinates
```

高层可以请求停止新的任务，但不能解除低层安全状态。低层安全也不能伪造 Mission 已完成；
它只能报告真实终态，由 Field Executive 决定重试、补偿、等待或失败。

## 10. 版本与升级规则

- 所有契约使用显式 `schemaVersion`；破坏性变化升级 major；
- Field Runtime 至少兼容当前和前一个 Cloud major；
- Release Manifest 固定每个仓库、镜像、模型、Schema 和配置摘要；
- Capability/模型包原子安装，验签后激活，保留 Last-Known-Good；
- 数据库迁移必须支持升级前备份和失败回滚；
- Cloud 不能要求一个离线 Edge 在升级完成前执行不兼容 Mission。

## 11. 传输边界

- Cloud ↔ Site/Edge：HTTPS/gRPC/MQTT/Event API + mTLS；
- Site ↔ Edge：站点局域网的相同应用协议；
- Edge ↔ Robot Agent：受限、版本化机器人协议；
- ROS DDS 只存在于机器人或受控现场实时网络；
- 浏览器不直接持有 Robot Agent、ROS、SONIC 或模型 Provider 凭据。

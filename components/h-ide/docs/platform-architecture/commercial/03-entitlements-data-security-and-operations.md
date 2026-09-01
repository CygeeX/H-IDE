# 商业授权、数据、安全与运维

| 属性 | 值 |
| --- | --- |
| 状态 | Commercial control-plane baseline |
| 核心目标 | 能销售云服务和离线能力包，同时不让收费逻辑进入本体安全路径 |

## 1. 商业能力不是一个布尔开关

每次任务编排都基于以下交集：

```text
EffectiveCapabilities
  = PurchasedEntitlements
  ∩ InstalledCapabilities
  ∩ SitePolicy
  ∩ RuntimeHealth
```

四项分别回答：客户是否购买、设备是否安装、站点是否允许、运行时是否健康。任何一项
缺失都不能签发相应节点。例如客户购买“本地复杂视觉”不代表 16GB Edge 已安装并能
健康运行对应模型；Field Executive 必须拒绝或升级到 Site/Cloud，而不是伪造本地能力。

## 2. 能力商品模型

建议把商品拆成三类：

| 商品 | 运行位置 | 计费方式示例 | 断云行为 |
| --- | --- | --- | --- |
| Cloud Service | H-IDE Cloud | 订阅、调用量、机器人/月、站点/月 | 不可用时使用批准 fallback 或 `deferred_cloud` |
| Site Service | Site Appliance | 站点许可、并发机器人、年度维护 | 公网断开可继续，依赖现场硬件 |
| Offline Capability Pack | Field Runtime | 设备许可、模型包版本、租约周期 | 在签名离线租约内继续运行 |

一个 Capability Package 至少包含：

- `capabilityId`、版本和兼容的硬件 Profile；
- 输入/输出 Schema 与资源预算；
- 容器/模型/Skill 摘要和签名；
- 许可证清单、模型权重条款和 SBOM；
- 数据驻留、是否允许云升级、最大离线租约；
- 安装、健康检查、回退和卸载策略。

购买服务后所谓“自进化”只能表现为签名能力包被安装和激活，不能让 Agent 自行生成、
下载或修改执行代码。

## 3. 永不进入付费墙的能力

以下能力不受订阅到期、云端欠费或模型额度影响：

- 硬件急停和驱动安全；
- 停止新运动、取消当前目标；
- 读取基本健康和故障状态；
- 已安装且产品承诺的本地安全返航/撤离；
- 导出客户自己的安全审计和事故证据。

收费服务不可用时可以失去高级识别、长期记忆或开放式规划，但不能失去安全停止能力。

## 4. 离线授权租约

Field Executive 缓存 Cloud 签发的 `EntitlementGrant`：

```text
tenant/site/device scope
+ allowed capability/version
+ issuedAt/notBefore/expiresAt/gracePeriod
+ maximum offline use
+ package digests
+ key ID and signature
```

离线可用时间越长，商业方越无法即时撤销。这是产品取舍，不是技术可以完全消除的问题。
建议按商品定义不同租约：Cloud Service 使用较短租约；购买的 Offline Pack 使用较长
租约和明确宽限期；完全离线企业许可使用现场许可证服务和合同审计。

Edge 只能用可信时钟、签名和已安装包验证授权。修改系统时间、复制租约到另一设备或
回滚到旧 Package 都必须被审计并拒绝。

## 5. Cognition Broker 商业策略

一次认知请求由 Field Executive 发起，Broker 按顺序检查：

1. 任务是否属于本地预授权范围；
2. 当前视觉/记忆结果是否达到置信度和新颖性阈值；
3. 本地 Provider 是否安装、健康且满足时延；
4. Site Brain 是否可用、允许处理该数据；
5. Cloud Service 是否已购买且网络可用；
6. 原始 RGB-D、关键帧或对象 Crop 哪一种允许上行；
7. 费用和时延预算是否允许升级。

推荐默认路由：

```text
Local → Site Model Gateway → Cloud
```

不建议本地和云端默认并行调用：它会增加成本、产生两个互相竞争的答案，并让任务完成
语义难以确定。只有明确的 benchmark、冗余或仲裁任务才并行。

模型无法解决时必须返回结构化 `insufficient_evidence`、`unsupported` 或
`deferred_cloud`，不能用低置信度文字填充成功结果。

## 6. 客户数据与隐私边界

### 6.1 所有权

- 客户拥有其原始图像、RGB-D、音频、点云、3DGS/PLY、场景内容和业务报告；
- H-IDE 拥有平台软件、协议、Schema、模型路由、编译器和通用算法；
- 客户应能导出自身 `SceneRevision`、WorldDelta、MissionEvent 和 Evidence manifest；
- 未经单独、可撤回授权，不使用客户原始数据做跨客户训练；
- 去标识运营指标也必须受合同、租户策略和数据字典约束。

### 6.2 默认上行策略

默认不持续上传原始 RGB-D。允许按站点配置上传：

- 任务关键帧；
- 目标 Crop、Mask 和必要上下文；
- Object Track、Embedding 或 WorldDelta；
- 故障前后有限时间窗的证据；
- 客户明确开启的连续原始流。

每个 EvidenceRef 记录采集目的、保留期限、访问主体、内容哈希和删除状态。Scene Explorer
与长期记忆需要的大规模数据优先在 Site Appliance 处理。

## 7. 身份、网络与密钥

### 7.1 身份层级

- 租户、站点、用户、服务、Site Appliance 和机器人分别拥有身份；
- 每台设备使用独立证书，不共享一个全局 API Key；
- Cloud ↔ Site ↔ Edge 使用 mTLS；
- MissionBundle、Capability Package、模型包和 EntitlementGrant 使用离线可验证签名；
- 签名密钥、设备身份和数据加密密钥分离，支持轮换与吊销。

### 7.2 网络边界

- 浏览器只访问同源 H-IDE API；
- Mission Architect 不访问 ROS；
- ROS DDS 只存在于机器人或现场实时 VLAN；
- Field Executive 通过受限 Robot Agent 协议调用能力；
- Model Provider 不持有 Robot Agent 凭据；
- Scene Explorer 和离线记忆 Worker 不在停止/急停路径中。

站点建议至少拆分管理、机器人控制、视觉证据、模型服务和访客网络，并对所有跨区调用
执行明确 allowlist。

## 8. Safety Profile

| Profile | 目的 | 典型策略 |
| --- | --- | --- |
| `simulation` | 验证 Agent、编排和恢复 | 接触可触发重规划而非立即失败；尽量不因次数上限终止；保留日志 |
| `pilot` | 有现场人员的受控真机试点 | 保守速度、碰撞/区域限制、人工急停、强诊断和任务批准 |
| `production` | 无开发者持续看护的商业运行 | 完整硬件安全、认证限制、站点策略、变更审核和事故流程 |

SONIC 的平衡控制应独立持续运行；上层可以取消 Mission 运动目标，但不得用订阅、VLM
失败或普通服务异常停止底层平衡控制。相反，生产真机也不能因为仿真“撞上没关系”而
关闭本体碰撞保护。

## 9. 审计与可观测性

每个用户任务必须能串联：

```text
conversationTurnId
  → missionId/revision
  → executionEpoch/nodeId
  → ModelInvocation / WorldDelta
  → SkillRequest / Robot Agent command
  → EvidenceRef / MissionEvent
  → final report
```

最低审计字段：操作者、租户/站点/机器人、模型与 Prompt/Schema 版本、Entitlement、
SceneRevision、输入证据哈希、策略决策、执行结果、费用维度和时间。敏感 Prompt/图像可以
加密或按策略不长期保存，但必须保留足够的不可逆摘要用于对账。

关键告警包括：

- Cloud/Site/Edge 心跳与事件队列积压；
- 证书/离线租约即将过期；
- 模型置信度、超时和费用异常；
- GPU OOM、热降频和 Provider 重启；
- Mission sequence 缺口或重复；
- SceneRevision 不匹配、定位 epoch 变化；
- Robot Agent/SONIC/Nav2 健康与安全事件。

## 10. 升级、回退与供应链

- 统一 Release Manifest 固定所有仓库、镜像、模型、配置和 Schema 摘要；
- 镜像和模型包必须签名并生成 SBOM；
- 使用分批发布：开发 → 内部站点 → 试点 → 生产；
- Edge/Site 保留 Last-Known-Good，安装失败或健康门失败自动回退；
- 数据 Schema 至少支持当前/前一主版本；
- 远端升级不能中断硬件停止、当前安全控制或破坏未完成任务台账；
- 许可证或模型权重变化视为新的产品评审，不做浮动依赖升级。

## 11. Site Brain 可用性

- 基础 Site Enterprise 可为单 Appliance；故障时 Edge 退化到已安装能力，并报告
  `site_unavailable`；
- 高可用企业档使用两台独立 Appliance、复制事件/对象存储元数据和显式 leader lease；
- GPU 任务不能在故障切换后自动重复产生物理动作，只能重跑幂等认知/批处理作业；
- 目标 PRO 6000 Demo 不需要假装实现双机，但契约和部署清单不得依赖共享进程内状态。

## 12. 商业上线门

商业 Pilot 前必须具备：

- 产品 SKU、Entitlement 和 Offline Pack 契约；
- 数据处理协议、保留/删除和客户导出机制；
- 设备证书、签名发布、SBOM 与漏洞响应；
- simulation/pilot/production Profile 隔离；
- 断云、重连、不重复物理动作和 Last-Known-Good 回退测试；
- FARM/DAAAM/权重/数据集/驱动的法律评审；
- 现场网络、UPS、散热、备份和事故处置手册。

# 商业化目标架构与架构选择

| 属性 | 值 |
| --- | --- |
| 状态 | Target architecture / 决策已冻结 |
| 面向 | 产品、平台、机器人、视觉、SRE、安全与商务团队 |
| 核心选择 | Cloud + 可选 Site Brain + Field Executive + Robot Runtime |

## 1. 为什么不是简单的“Agent 放云端还是机器人”

商业架构至少要分别回答五个问题：

1. 谁拥有实时运动和急停权；
2. 谁编写 Mission，谁推进 Mission；
3. 模型推理在本地、现场还是云端运行；
4. 当前物理事实与长期世界记忆分别由谁管理；
5. SaaS、客户 VPC、现场私有化如何复用同一产品。

把五个问题压缩成一个部署位置，会产生两种坏结果：纯云方案断网即失能，纯本地方案
则把 GPU、散热、升级、IP 保护和多机器人共享成本全部压到机器人上。

## 2. 评估过的架构选项

| 选项 | 优点 | 主要问题 | 结论 |
| --- | --- | --- | --- |
| 纯云大脑 + 机器人运控 | 机器人 BOM 低、SaaS 管理简单 | WAN 抖动进入任务关键路径；断云无法接受任务；隐私和带宽压力大 | 仅适合早期演示/低价在线档，不作为主架构 |
| 云端规划 + 每机器人 Edge | 低延迟、单机断云自治 | 每台机器人增加功耗、重量、成本和模型运维 | 作为 Edge Pro 产品 |
| 云端 + Site Brain + 瘦机器人 | 一张现场 GPU 可服务多机器人，隐私好 | Site Brain 成为站点故障域；机器人离开站点网络后能力下降 | 作为企业站点产品 |
| Cloud + Site + Edge + Robot 四层 | 能按客户需求折叠，在线/离线边界清晰 | 工程复杂度最高，需要正式协议和版本治理 | **商业主线** |
| 全部部署在机器人 | 完全离线、最低网络时延 | 热设计、续航、GPU 成本和 IP 暴露最严重 | 只用于特殊离线 SKU |
| 完全现场私有化 | 数据主权和弱网能力最好 | 客户现场升级和支持成本高 | 同一架构的企业交付 Profile |
| ROS/DDS 图直接跨公网 | 开发时看似直接 | 安全、NAT、时延、版本和故障语义不可控 | 禁止作为商业边界 |

四层架构并不意味着每个客户必须购买四台机器。逻辑边界固定，物理位置按产品档位折叠。

## 3. 目标逻辑架构

```text
┌──────────────────────── H-IDE Cloud / Customer VPC ────────────────────────┐
│ PWA、租户、订单、Entitlement、Mission Architect、重型 LLM/VLM            │
│ 全局长期 World Model、Scene Explorer、Scene Publisher、报告、审计、Registry│
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ HTTPS/gRPC/Event API
                               │ signed MissionBundle / WorldDelta
┌──────────────────────────────▼─────────────────────────────────────────────┐
│ H-IDE Site Brain（可选）                                                    │
│ Site Model Gateway、共享 GPU、FARM/DAAAM Provider、Site World Model、Fleet │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ site-local authenticated API
┌──────────────────────────────▼─────────────────────────────────────────────┐
│ H-IDE Field Runtime                                                        │
│ Field Executive、Cognition Broker、Mission Validator、World Cache、        │
│ Entitlement Cache、Live Visual Resolver、Store-and-Forward                 │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ bounded Robot Agent protocol
┌──────────────────────────────▼─────────────────────────────────────────────┐
│ Robot Runtime                                                              │
│ Robot Agent、Nav2、SLAM/TF、SONIC、驱动、Watchdog、E-stop                  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 H-IDE Mission Architect

Mission Architect 负责“应该做什么”：

- 理解自然语言、上下文与商业能力；
- 选择 Skill、生成高层 DAG、预算和完成条件；
- 产生版本化 MissionDraft；通过 Schema、策略和人工审核后，由 Cloud 签名服务生成
  `MissionBundle`；
- 根据 Edge 返回的事实提出修订或补充任务。

它不持有 ROS、Nav2、Robot Agent 或 SONIC 凭据，不输出速度、路径、关节目标和未经
验证的可执行坐标。

### 3.2 H-IDE Field Executive

Field Executive 负责“现在可以执行哪一步”：

- 持有当前 Mission 的唯一执行 epoch；
- 校验签名、授权、场景版本、机器人能力和依赖；
- 调度本地工具并对 Robot Agent 发出有界 SkillRequest；
- 维护当前物理状态、幂等键、证据和事件队列；
- 断云后完成缓存 Mission，并接受预授权标准任务；
- 发现复杂或低置信度问题时请求认知升级。

云端可以编写新版本，但不能绕过 Field Executive 直接推进一个物理节点。

### 3.3 H-IDE Cognition Broker

Cognition Broker 是 Field Runtime 的模型路由和能力策略层。一次推理按以下顺序处理：

```text
Task + Evidence + latency/privacy/entitlement policy
        │
        ├─ local model can answer with sufficient confidence ──► local result
        │
        ├─ Site Brain available and allowed ────────────────────► site result
        │
        ├─ Cloud available and allowed ────────────────────────► cloud result
        │
        └─ no valid route ─────────────────────────────────────► deferred_cloud
```

路由因素包括任务类型、置信度、新颖性、时延预算、原始图像上传策略、网络健康、GPU
负载和购买的能力。Site/Cloud 的返回值仍是 `hypothesis`；执行目标必须经过当前视觉、
几何和 Mission Validator 验证。

### 3.4 H-IDE Site Brain

Site Brain 是可选现场产品，适用于多机器人、数据不出场、弱公网或需要共享重模型的客户：

- 为多台机器人提供低延迟 VLM、Embedding、语义记忆和 Scene Compiler；
- 缓存云模型、Capability Package 和长期 WorldDelta；
- 在公网中断时维持站点级认知服务；
- 未来承载 Open-RMF 类交通、门、电梯与 Fleet 协调。

基础版可以没有 Site Brain；企业版可购买单机或主备 Appliance。

### 3.5 Robot Runtime

Robot Runtime 是实时和物理安全权威：

- SLAM/TF 提供当前定位；
- Nav2 负责路径、局部避障和恢复；
- SONIC 持续负责全身运动和平衡；
- Robot Agent 把版本化能力映射为有限 ROS Action/Service；
- 硬件急停、驱动限制和 Watchdog 的优先级高于所有上层服务。

Cloud、Site Brain 或 Field Executive 都不能解除硬件急停，也不能绕过 Robot Agent
发送 `/cmd_vel` 或关节指令。

## 4. 在线 Mission 生命周期

```text
用户输入
  → Mission Architect 生成 MissionDraft
  → 策略/人工批准
  → 签名 MissionBundle 下发
  → Field Executive 验签并创建 executionEpoch
  → 逐节点解析 entity/constraint
  → Live Visual Resolver / Local Goal Resolver 生成当前事实
  → Robot Agent → Nav2 → SONIC
  → MissionEvent + EvidenceRef 异步回传
  → 报告与长期 World Model 更新
```

Cloud 不逐步遥控机器人。完整 DAG、预算、依赖和降级规则在执行前下发，Edge 只异步
回传事件。这种方式使 WAN 抖动不会直接变成步态停顿或 Mission 重复执行。

## 5. 断云与重连语义

### 5.1 断云时允许

- 完成当前已签名 Mission；
- 运行已缓存且仍在离线授权租约内的标准巡检；
- 停止、取消、急停和受支持的返航；
- 查询本地已知对象和当前状态；
- 使用本地模型或 Site Brain 做允许的数据处理。

### 5.2 断云时不允许

- 接受无边界的新开放式复杂任务；
- 使用未安装或未授权的能力包；
- 让本地 LLM 临时生成新的运动接口或任意代码；
- 把无法确认的云模型调用伪装成成功。

云必需节点先尝试签名策略允许的本地替代；仍不能完成时标记为 `deferred_cloud`，不
阻塞 DAG 中无依赖的其他节点。

### 5.3 重连对账

Edge 的实际执行状态优先。对账键至少包含：

```text
missionId + revision + executionEpoch + nodeId + idempotencyKey
```

已完成或正在运行的节点不能被云端原地修改；状态未知的物理动作不自动重放。需要纠正
时使用新的补偿节点或 Mission revision，而不是覆盖历史。

## 6. World Model 权威分配

| 数据 | 权威位置 | 同步方式 |
| --- | --- | --- |
| 当前定位、障碍、可通行性、机器人健康 | Robot/Edge | 高频本地；按需摘要上行 |
| 当前对象 Track 与执行证据 | Edge | `WorldDelta` 异步上行 |
| 站点房间、门、发布 Anchor | 已发布 `SceneRevision` | 不可变 revision 分发 |
| 跨 Mission 长期语义记忆 | Site/Cloud | 事件与证据合并 |
| 租户、订单、服务权益 | Cloud | 签名 `EntitlementGrant` 下发 |

统一模型必须区分：

1. `published_static_fact`：已审核静态事实；
2. `historical_observation`：历史看到过；
3. `hypothesis`：模型推断；
4. `live_execution_target`：由当前几何验证、短时有效的执行目标。

只有第一类可作为稳定语义引用，第四类可在 TTL 内成为当前导航目标。历史观察或模型
假设不能直接变成 Nav2 Pose。

## 7. 交付 Profile

| Profile | 物理部署 |
| --- | --- |
| 目标本地 Demo | 验机后的 PRO 6000 同机运行 cloud-zone、site-zone、edge-zone；Robot Runtime 在仿真主机/同一 LAN |
| Cloud Basic | H-IDE SaaS + 机器人最小 Field Runtime；重认知依赖云 |
| Edge Pro | H-IDE SaaS + 每机器人 T4000 64GB 评估候选 Field Runtime |
| Edge Max | H-IDE SaaS + 每机器人 T5000 128GB Field Runtime |
| Site Enterprise | H-IDE SaaS/客户 VPC + Site Appliance + 最小或 Pro Edge |
| Private/Offline | Cloud 控制面与 Site Brain 均部署客户现场，Field/Robot 契约不变 |

这些是同一套镜像、协议和版本清单的不同部署清单，不是独立分叉的产品代码。

## 8. 明确拒绝的设计

- 云端 LLM 直接产生 `/cmd_vel`、路径、航点数组或关节目标；
- Cloud 与 Edge 同时推进同一 Mission；
- 把 ROS DDS 当作公网商业协议；
- 把 FARM/DAAAM 的私有序列化文件当作 H-IDE 核心协议；
- 断网重连后自动重发未知终态动作；
- 购买能力后动态下载并执行未经签名的 Agent 代码；
- 为了 Demo 放宽安全条件后，将相同 Profile 部署到真机。

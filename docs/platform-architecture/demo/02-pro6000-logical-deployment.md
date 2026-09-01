# PRO 6000 同机云边验证部署

| 属性 | 值 |
| --- | --- |
| 状态 | Demo deployment design / 尚未实现 |
| 目标硬件 | 用户计划使用、尚待本方案验机的工作站；确认是 RTX PRO 6000 Blackwell Workstation Edition 96GB，还是 RTX 6000 Ada Generation 48GB |
| 目标 | 无计算背包时，在同一 LAN 验证 Cloud、Site、Edge 和 Robot 的正式契约与断网语义 |

## 1. Demo 拓扑

推荐使用两台逻辑/物理主机：

```text
PRO 6000 Workstation
├─ cloud-zone
│  ├─ PWA / Control API
│  ├─ Mission Architect
│  ├─ Entitlement / Registry
│  └─ long-term World Model
├─ site-zone
│  ├─ Site Model Gateway
│  ├─ FARM Provider（shadow）
│  ├─ DAAAM Provider（shadow）
│  ├─ Semantic Scene Compiler
│  └─ Scene Explorer（后续）
└─ edge-zone
   ├─ Field Executive
   ├─ Cognition Broker
   ├─ Live Visual Resolver client/local profile
   ├─ Mission/Entitlement/World Cache
   └─ Store-and-Forward
              │ same LAN / formal Robot Agent protocol
              ▼
Robot/Simulation Host
├─ Robot Agent + Observation Gateway
├─ Nav2 / SLAM-or-sim-pose / TF
├─ SONIC
└─ MuJoCo G1
```

如果暂时只有一台物理主机，也可以把 Robot Runtime 放在第四个网络/进程域，但这种方式
只能验证协议和网络故障，不能证明 GPU/电源/内核故障隔离。

## 2. 同机不等于同一进程

即使三个 Zone 位于同一台 PRO 6000，也必须满足：

- 独立 Compose project 或等价容器编排；
- 独立容器网络、服务身份和证书；
- 独立数据库/Schema、对象存储 bucket 和凭据；
- 只通过正式 HTTPS/gRPC/MQTT/Event API 传输；
- 禁止直接 import 另一个 Zone 的业务包；
- 禁止共享 SQLite 文件、内存对象或宿主机临时状态；
- 禁止 cloud-zone 访问 Robot Agent、ROS DDS 或 SONIC 端口；
- Edge 是唯一可访问 Robot Agent command API 的 Zone；
- Site Provider 只接收 ObservationBatch，不持有运动凭据。

同一物理 PostgreSQL/对象存储实例可以用于 Demo，但必须使用独立 database/bucket、账号和
网络策略，不能让服务通过共享表绕过契约。

## 3. Zone 服务映射

### 3.1 cloud-zone

| 当前服务 | Demo 角色 | 变化 |
| --- | --- | --- |
| `gateway` / PWA | 用户入口 | 保留 |
| `control-api` | Cloud control plane、审核、报告 | 把执行推进逐步移交 Field Executive |
| `planning-agent` (`h-ide-agent`) | Mission Architect | 产品名改变，仓库可暂不改名 |
| Scene API | SceneRevision 管理与审核入口 | 不再从 Office 静态 catalog 自动成为生产事实 |
| PostgreSQL/Redis | Cloud repository/event fanout | 从当前 SQLite 运行态迁移 |

### 3.2 site-zone

| 新服务 | 作用 |
| --- | --- |
| `site-model-gateway` | 对 Provider 做鉴权、配额、GPU 队列和统一 ModelInvocation |
| `observation-recorder` | 保存 posed RGB-D、TF、标定和 Evidence manifest |
| `farm-provider` | 独立 CUDA 12.8/ROS 2 Humble 环境，先只写 shadow store |
| `daaam-provider` | 独立 Ubuntu 24.04/ROS 2 Jazzy 环境，先只写 shadow store |
| `world-model-service` | Canonical entity/observation/relations，不保存 Provider 原生 pickle 作为协议 |
| `semantic-scene-compiler` | 产生房间、门、拓扑和 Anchor proposal |
| `scene-explorer` | 异步 PLY/3DGS 审计，后续里程碑启用 |

FARM 和 DAAAM 的 ROS/CUDA 依赖不同，必须保持独立镜像，不能放进一个 ROS workspace。
Shadow Mode 优先读取离线 ObservationBatch、frames-json 或隔离 rosbag。若 Provider 必须
启动 ROS，则使用与 Robot Runtime 不同的 ROS Domain 和受限容器网络，使其无法发现
Robot Agent、Nav2 或 SONIC graph；Provider 不持有 Robot Agent token 或 motion capability。
FARM、DAAAM、DAAAM-ROS 及全部上游依赖 commit 都必须进入 Release Manifest。

### 3.3 edge-zone

| 新服务 | 作用 |
| --- | --- |
| `field-executive` | 唯一 Mission execution epoch、节点推进、Robot Agent 对账 |
| `cognition-broker` | Local → Site → Cloud 的认知升级策略 |
| `mission-validator` | Bundle 签名、Scene/Capability/Entitlement/预算验证 |
| `edge-world-cache` | 当前物理事实、已发布 SceneRevision 和短期对象 Track |
| `edge-event-spool` | 断云 MissionEvent/Evidence 索引持久化与重连回填 |

为了模拟未来 Thor，edge-zone 应设置明确 CPU/RAM/disk 配额。Field Executive 的纯逻辑
路径不直接使用整张 PRO 6000；需要模型时必须通过 Cognition Broker 调显式 Provider。

## 4. 数据流

### 4.1 Mission 下行

```text
PWA → Cloud Mission Architect → review/sign
    → Mission Dispatch API/queue
    → Edge Mission Validator
    → Field Executive ledger
```

Cloud 下发完整 Bundle，不通过逐节点同步 HTTP 控制机器人的每一步。

### 4.2 执行与事件上行

```text
Field Executive → Robot Agent → Nav2/SONIC
       └─ MissionEvent / EvidenceRef → local spool
                                      → Cloud when available
```

同一事件重发必须幂等，Cloud 不因网络超时重发物理动作。

### 4.3 视觉和记忆

```text
Observation Gateway
  → Observation Recorder（一次写入、内容寻址）
      ├─ Live Visual Resolver（实时）
      ├─ FARM Provider（shadow replay）
      ├─ DAAAM Provider（shadow replay）
      └─ reconstruction/SLAM asset exporter
             → PLY/3DGS SceneAsset → Scene Explorer（异步）
```

Provider 读取同一 Evidence，不各自向机器人建立第二条控制或采集链。

## 5. GPU 调度

单卡 PRO 6000 使用优先级队列：

1. `P0`：当前 Mission 的 Live Visual Resolver；
2. `P1`：用户正在等待的 Cognition Broker 请求；
3. `P2`：Semantic Scene Compiler 的有限 VLM/Embedding 作业；
4. `P3`：FARM/DAAAM Shadow replay；
5. `P4`：Scene Explorer、离线报告和 benchmark。

Shadow 阶段默认：录制一次 → FARM 顺序回放 → 卸载/释放 → DAAAM 顺序回放。禁止为了
“看起来集成了”而让最大场景 FARM、尚未实测组合峰值的 DAAAM 和重 VLM 同时常驻。

队列必须落实为可执行资源机制，而不只是 P0–P4 标签：

- 单 GPU exclusive lease；
- 启动前 GPU memory admission watermark；
- 为 Live Visual Resolver 和运动相关 CUDA context 保留不可借用的余量；
- Provider 采用 `load → run → checkpoint → terminate → verify-memory-released` 生命周期；
- OOM 后执行 CUDA context/GPU health probe，健康未恢复时不启动下一作业；
- 48GB RTX 6000 Ada Profile 明确拒绝超过预算的 FARM/DAAAM 作业；
- 普通容器队列不能抢占已占满显存的 CUDA 进程，终止/卸载策略必须在 Provider Adapter
  中实现并测试。

如果 Provider 共享同一检测/Embedding/VLM 服务，仍要记录每个结果的模型版本和证据；
共享模型不能把两个 Provider 的实体身份悄悄合并。

## 6. 网络故障注入

同机 loopback 延迟不能代表真实云。cloud-zone 与 edge-zone 之间必须经过 Toxiproxy、
`tc netem` 或等价代理，至少验证：

| 场景 | 建议注入 | 期望 |
| --- | --- | --- |
| 普通 WAN | 40–120ms RTT、轻微抖动 | Mission 不因每个节点等待云而停顿 |
| 丢包 | 1–3% + 重排 | 事件可重传，物理动作不重复 |
| 短断网 | 30–120 秒 | 当前 Mission 继续，事件进入 spool |
| 长断网 | 10 分钟以上 | 预授权标准任务可用，复杂任务 `deferred_cloud` |
| 重连 | 恢复连接并注入重复消息 | execution epoch 对账，无重复导航/扫描 |
| Cloud 重启 | 清空易失连接，不清 Edge | Edge 继续；Cloud 从事件/快照恢复 |
| Site Provider OOM | 杀掉 Provider | 当前运动链不退出；Broker 降级或延迟 |

测试代理只位于应用协议路径，不能对 Robot Runtime 内部 SONIC 控制循环注入 WAN 延迟。

## 7. 身份和网络规则

建议网络：

```text
frontend-net       PWA ↔ Cloud gateway
cloud-services-net Mission Architect / entitlement / repositories
cloud-edge-wan     仅 MissionBundle、MissionEvent、WorldDelta
site-model-net     Cognition Broker ↔ Site Model Gateway
edge-robot-net     Field Executive ↔ Robot Agent
evidence-net       Observation Recorder / Provider read-only
robot-ros-domain   Robot Agent / Nav2 / SONIC only
```

每个 Zone 使用独立证书。Demo 可以由本地 CA 签发，但不能继续共享 `dev-token` 来证明生产
安全。浏览器、Mission Architect 和模型 Provider 都不能访问 `edge-robot-net`。

## 8. 存储布局

| 数据 | 存储 | 规则 |
| --- | --- | --- |
| Cloud Mission/Entitlement | cloud Postgres | 业务与长期状态 |
| Edge Mission ledger | edge Postgres/SQLite-WAL 独立卷 | 断云可用，不能与 Cloud 共库 |
| Mission event spool | edge append-only store | sequence + idempotency |
| RGB-D/PLY/3DGS/Evidence | site object store | 内容寻址、保留期、权限 |
| FARM/DAAAM native output | provider-private volume | 不跨信任域、不成为产品 API |
| Canonical World Model | site/cloud database | 只接收 Provider-neutral WorldDelta |
| SceneRevision | Cloud/Site revision store | 不可变发布、可回滚引用 |

## 9. 当前 Demo 的迁移方式

1. 保留现有 Compose 和 Office V2 稳定基线；
2. 先建立 MissionBundle/Event API，并让新的 Field Executive 以 shadow ledger 对账；
3. 再把唯一上游 Robot Agent session/dispatch authority 从 Control API 切到 Field
   Executive；Robot Agent 仍拥有本地 ROS goal/资源/租约；
4. Office V2 仍通过 legacy SceneRevision 运行，验证行为不回退；
5. 添加 Observation Recorder 和 FARM/DAAAM Shadow Provider；
6. Semantic Scene Compiler 在第二个场景产生新 revision；
7. 泛化 Nav2 map_server、strict planner、runner 和 launcher，用 scene/map/runtime digest
   做受控冷切换与 LKG 回退；
8. 新场景通过验收后，生产路径停止直接读取 Office catalog；
9. PCD/Potree 在替代 Viewer/资产链验收后再做 breaking removal。

每一步都有单独 feature flag 和 Last-Known-Good，不做一次性大爆炸迁移。

## 10. 同机 Demo 能证明与不能证明的事

能够证明：

- 正式 Cloud/Edge 契约可用；
- 断网、事件回填、不重复动作语义成立；
- Cognition Broker 能按策略路由模型；
- Provider 可以被替换而不改 Mission；
- 新场景不写后端坐标也能发布并巡检。

不能证明：

- Thor ARM64/JetPack 兼容与性能；
- 背包重量、功耗、散热、续航和 G1 平衡；
- 真实公网 NAT、运营商网络和跨地域时延；
- Site Appliance/Edge 的物理断电隔离；
- 真机 SLAM、传感器噪声、玻璃/反光和真实碰撞安全；
- 企业双机 HA 和生产安全认证。

这些能力在后续硬件和 Pilot 阶段分别验收，不能用同机容器测试替代。

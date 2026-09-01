# 产品档位、硬件与成本模型

| 属性 | 值 |
| --- | --- |
| 状态 | Commercial planning baseline |
| 价格核对日期 | 2026-08-31 |
| 货币 | USD；均为预算或官方参考价，不含中国渠道、税费和集成研发 |
| 采购原则 | 先对目标 PRO 6000 验机并完成 Demo，再采购 Thor Developer Kit 做迁移验证 |

本文只估算 H-IDE 计算、网络和存储子系统，不包含 G1 本体、相机/LiDAR、背包电池与
结构、认证、现场部署、质保、SaaS 成本、运维人力、税费、毛利和完整 TCO。

## 1. 先确认“PRO 6000”具体型号

“PRO 6000”可能指不同硬件，不能只凭名称做容量承诺：

- [RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)：96GB GDDR7 ECC，最高 600W；
- [RTX 6000 Ada Generation](https://www.nvidia.com/en-us/products/workstations/rtx-6000/)：48GB GDDR6 ECC，最高 300W。

本文的 Site Appliance 容量表默认 **RTX PRO 6000 Blackwell 96GB**。部署前必须保存
`nvidia-smi` 型号、总显存、驱动、CUDA、功耗上限和整机配置；48GB Ada 不能按
96GB 方案承诺完整 FARM。

## 2. 推荐产品档位

| 产品 | 机器人/现场硬件候选 | 目标能力（通过对应验收后） | 断云目标 | 目标客户 |
| --- | --- | --- | --- | --- |
| `H-IDE Cloud Basic` | G1 现有计算单元，必要时加 Orin NX 16GB | Field Executive、任务续跑、标准模板、轻量检测/跟踪；复杂认知上云 | 完成已签名 Mission、预授权标准任务 | 网络稳定、成本敏感的单机器人客户 |
| `H-IDE Edge Pro` | Jetson T4000 64GB 评估候选；待供应与生命周期确认 | 更强本地视觉、短期语义记忆、中型量化模型；FARM 留在 Site/Cloud | 大部分常规巡检和已知场景任务 | 需要较强离线能力的客户 |
| `H-IDE Edge Max` | Jetson T5000 128GB 量产背包 | 本地重视觉、DAAAM/FARM 候选、长时间离线能力包 | 在购买/安装能力范围内接受更多本地任务 | 弱网、保密、单机自治优先客户 |
| `H-IDE Site Enterprise` | 单/双 RTX PRO 6000 Appliance | 多机器人共享 FARM/DAAAM、Scene Explorer、重 VLM、长期记忆 | 公网断开时维持站点服务 | 工厂、医院、园区、多机器人客户 |
| `H-IDE Private` | 客户 VPC 或现场双机 Appliance + 可选 Thor | 完整私有化控制面、模型与数据驻留 | 可配置完全不依赖公网 | 强合规/数据主权客户 |

产品能力以 `EntitlementGrant ∩ CapabilityManifest ∩ SitePolicy ∩ RuntimeHealth`
为准，不能仅凭硬件型号宣称某项功能可用。

当前 G1 EDU 资源规划暂按“16GB 统一内存中，SONIC/Nav2/SLAM 和系统占用后仅约 3–4GB
可分配给 H-IDE”这一项目预算假设设计。Field Executive 的状态机、协议和缓存预计可在
此范围内运行，但必须实测；轻量跟踪前端也必须通过峰值内存和长稳门后才能启用。该预算
不满足完整 FARM、DAAAM 或重型 VLM，Cloud Basic 必须通过 Cognition Broker 使用
Site/Cloud 认知能力。

## 3. Jetson 选型

规格和官方参考价来自 [NVIDIA Jetson FAQ](https://developer.nvidia.com/embedded/faq)、
[Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
与 [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)。

| 硬件 | 统一内存 | 模组功耗 | NVIDIA 价格口径 | 量产整机预算 | 推荐用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| Orin NX 16GB | 16GB LPDDR5 | 10–40W | Module suggested price at 1KU+: $999 | $1,500–3,000 | Cloud Basic / 最小 Field Runtime |
| AGX Orin 64GB | 64GB LPDDR5 | 15–60W | Module suggested price at 1KU+: $2,999；Dev Kit 单件参考 $3,499 | $4,000–7,000 | 兼容/过渡，不作为新主线 |
| Jetson T4000 | 64GB LPDDR5X | 40–70W | Module suggested price at 1KU+: $2,999 | $5,000–9,000 | Edge Pro 评估候选，待生命周期/供应确认 |
| Jetson T5000 | 128GB LPDDR5X | 40–130W | Module suggested price at 1KU+: $4,999；Dev Kit 单件参考 $5,499 | $8,000–14,000 | Edge Max |

量产整机预算包含载板、NVMe、散热、电源和机壳的粗略成本，不包含背负结构、电池、
EMC/安规认证、税费、进口、小批量溢价和研发费用。

### 3.1 开发套件不是机器人背包

Thor Developer Kit 使用 T5000 和 1TB NVMe，尺寸约 `243 × 112 × 57 mm`，适合工作台
移植和性能验证。商业机器人必须使用量产模组、定制/认证载板和经过结构、热、供电、
续航及 G1 动力学验证的背负系统，不能把 Developer Kit 直接绑在机器人上出售。

40–130W 是模组功耗范围，不是背包整机功耗。加入载板、NVMe、风扇/泵、DC/DC 和外设后，
Edge Max 供电、散热和电池峰值会明显更高；必须实测整机功率，不能直接用 130W 计算续航。

### 3.2 统一内存不是“可全部分给模型的显存”

Jetson 的 64/128GB 同时被系统、ROS、传感器缓冲、容器和 GPU 使用。容量评估必须记录：

- 冷启动和稳态 resident memory；
- RGB-D 分辨率、帧率与队列；
- TensorRT/CUDA workspace 和 KV cache；
- ROS 2、SLAM、Nav2、日志与文件缓存；
- 30–60 分钟热稳定后的降频和功耗。

## 4. 工作负载资源可行性矩阵

下表是容量和平台适配假设，不表示 FARM/DAAAM 已集成、已移植或达到产品 SLA。

| 平台 | Field Executive | 轻量视觉 | DAAAM | 完整 FARM | 本地重 VLM |
| --- | --- | --- | --- | --- | --- |
| Orin NX 16GB | 预计可行，待实测 | 基础检测/跟踪待实测 | 不满足官方 24GB+ 设备门槛 | 不满足 | 仅小模型候选或升级到云 |
| AGX Orin 64GB | 预计可行 | 预计可行 | 容量候选，需 ARM64 移植验证 | 不建议 | 小/中型量化模型候选 |
| T4000 64GB | 预计可行 | 预计适合 | 容量候选，需移植与长稳测试 | 仅裁剪版/小场景候选 | 中型量化模型候选 |
| T5000 128GB | 预计可行 | 预计适合 | 容量候选，仍需验证依赖 | 首选机器人侧验证平台 | 中大型量化模型候选 |
| 单 RTX PRO 6000 96GB | 可模拟 | 资源适合 | x86 容量候选，需 H-IDE benchmark | 官方 FARM 测试平台，仍需 H-IDE benchmark | 可验证，但必须调度 |
| 双 RTX PRO 6000 | 可模拟 | 资源适合 | 可分卡验证 | 可分卡验证 | 可与记忆流水线做 GPU 隔离 |

[FARM 官方仓库](https://github.com/GoldenGait/FARM-Project)报告完整 mapping、LLM/
embedding 和 Viewer 通常约占 50GB GPU memory，最大场景约 65GB，并测试过 RTX 5090、
RTX PRO 6000 和 Jetson Thor。它没有因此保证所有 Thor 配置、传感器和场景都满足
H-IDE 的 FPS、热稳定和并发要求。

[DAAAM 安装文档](https://github.com/MIT-SPARK/DAAAM/blob/main/INSTALL.md)要求 NVIDIA
GPU 且至少 24GB，但未承诺官方 Jetson ARM64 产品支持。AGX Orin、T4000 和 T5000
必须逐项验证 ROS、Python wheel、TensorRT、Tracker 和模型权重。

### 4.1 单卡 PRO 6000 的并发边界

FARM 报告的最大场景约 65GB，在 96GB 卡上只留下约 31GB 给系统、CUDA context、
DAAAM、VLM、渲染和缓存。DAAAM 的“24GB+”是官方设备最低门槛，不是已公开的实际峰值
占用，不能与 65GB 直接相加。由于 H-IDE 尚未测得组合峰值，单卡 Site Appliance 不承诺
FARM、DAAAM 和重型 VLM 全部常驻并发。推荐：

1. 共享检测、Embedding 和 VLM 服务，避免每个 Provider 重复加载；
2. 用 GPU 作业队列和优先级区分实时视觉、离线记忆和 Scene Explorer；
3. Demo Shadow Mode 对同一录制数据顺序回放 FARM/DAAAM；
4. 需要持续并发时采购双 GPU Appliance，并让实时与批处理分别占卡。

## 5. Site Appliance 预算

RTX PRO 6000 Blackwell Workstation Edition 为 96GB ECC、PCIe 5.0 x16、最高 600W。
[NVIDIA Marketplace](https://marketplace.nvidia.com/en-us/enterprise/laptops-workstations/nvidia-rtx-pro-6000-blackwell-workstation-edition/)
在本文核对日显示单卡约 $16,000 且供应状态会变化。该价格不是固定 MSRP 或采购保证。

| 设备 | 建议预算（不含税） | 能力定位 |
| --- | ---: | --- |
| 已购且验机通过的 PRO 6000 工作站 | 增量 GPU 成本接近 0 | 当前 Demo 首选 |
| 单卡 PRO 6000 塔式工作站 | $22,000–40,000 | 研发、单站点 Site Appliance |
| 单卡服务器 Appliance | $30,000–50,000 | 正式现场、远程运维和冗余部件 |
| 双卡服务器 | $55,000–90,000 | 重模型隔离、多机器人、企业档 |

[RTX PRO 6000 Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
需要服务器风道，不能随意装入普通机箱。双卡 2U 商业整机可参考
[Supermicro GPU Systems](https://store.supermicro.com/us_en/systems/gpu.html?system_form_factor=1296)，
具体价格和配置以采购询价为准。

推荐单卡正式 Site Appliance：

- 24 核以上服务器/工作站 CPU；
- 256GB ECC RAM；
- 4–8TB 本地企业级 NVMe；
- 10GbE；
- 独立 40–100TB 可用容量 NAS；
- 1500W 以上合规电源和 3kVA 在线 UPS；
- TPM 2.0/HSM、Secure Boot 与受控 BMC；
- 独立设备、管理、机器人和证据网络/VLAN。

## 6. 网络、存储和供电预算

| 子系统 | 小型站点预算 | 企业/冗余预算 | 说明 |
| --- | ---: | ---: | --- |
| 2.5/10GbE + 企业 Wi-Fi | $2,000–5,000 | $8,000–25,000 | 机器人独立 SSID/VLAN，控制与证据流量分离 |
| 40–100TB NAS + 企业盘 | $5,000–12,000 | 按保留期扩展 | RAID 不是备份；原始 RGB-D 要有保留策略 |
| 单工作站在线 UPS | $3,000–7,000 | — | 约 3kVA/2.7–3kW |
| 双 GPU 服务器供电 | — | $10,000–25,000 | 6–10kVA、220V、现场配电和散热评估 |

预算参照可使用 [Ubiquiti Pro Max 24 PoE](https://store.ui.com/us/en/products/usw-pro-max-24-poe)、
[Ubiquiti E7](https://store.ui.com/us/en/products/e7)、
[Synology DS1825+](https://www.synology.com/en-us/products/DS1825%2B) 和
[Eaton 9PX 3000](https://www.eaton.com/us/en-us/skuPage.9PX3000RT.html)；它们只用于
规模估算，不构成品牌绑定。

## 7. 推荐采购顺序

1. 对用户计划使用的 PRO 6000 记录 hostname、精确 SKU、`nvidia-smi` 和整机 manifest，
   验机通过后完成 cloud/site/edge 逻辑隔离和 Semantic Scene Compiler Demo；
2. 优先补齐 NVMe、NAS、10GbE 与 UPS，建立可重复数据和证据链；
3. Demo 通过后购买一台 Thor Developer Kit，在工作台验证 ARM64、JetPack、容器和模型；
4. 如果 FARM/长期记忆主要留在 Site Brain，量产 Edge 把 T4000 作为首选评估候选；
   在 NVIDIA 书面确认供应和生命周期前，不锁定量产 BOM；
5. 如果产品明确销售“单机器人离线重认知能力包”，再选择 T5000；
6. 试点阶段才投入量产载板、电池、散热、背负结构和认证设计。

## 8. 采购前必须重新验证

- 型号、显存、生命周期、渠道价格和交货期；
- Developer Kit 与量产模组的软件/接口差异；
- 机器人背负后的重心、平衡、续航、散热、噪声和防护；
- ARM64 依赖、模型许可证和 TensorRT 再分发条款；
- 真实场景 60 分钟持续运行的 FPS、峰值内存和温度；
- G1 3–4GB 预算 Profile 的峰值 RAM/GPU、CPU/GPU 占用、控制周期抖动和温度；
- Site Appliance 故障后的 Edge 降级能力；
- 数据保留带来的实际存储增速和备份成本。

NVIDIA 的模组生命周期应在 BOM 锁定前从
[Jetson Product Lifecycle](https://developer.nvidia.cn/embedded/lifecycle)重新核对。本文核对
时页面明确列出 T5000 和 Orin 系列，但未给出 T4000 的同等生命周期承诺；量产选择必须
向 NVIDIA/授权渠道取得书面确认。

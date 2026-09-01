# H-IDE 云—站点—边缘—机器人平台架构

| 属性 | 值 |
| --- | --- |
| 状态 | Architecture baseline / 已完成设计评审问题收口 |
| 决策日期 | 2026-08-31 |
| 适用范围 | 商业化目标架构、PRO 6000 本地验证 Demo |
| 不包含 | 本轮不修改运行代码、不宣称 FARM/DAAAM 已进入执行链 |

## 1. 结论

H-IDE 采用一套**逻辑固定、物理部署可折叠**的四层架构：

```text
H-IDE Cloud
        │ signed MissionBundle / CapabilityGrant / WorldDelta
        ▼
H-IDE Site Brain（可选）
        │ site inference / semantic memory / fleet services
        ▼
H-IDE Field Executive
        │ validated skill request / execution event
        ▼
Robot Runtime（Robot Agent / Nav2 / SLAM / SONIC）
```

当前没有机器人计算背包时，Cloud、Site Brain 和 Field Executive 可以部署在同一台
RTX PRO 6000 工作站上，通过独立容器、网络、身份和存储模拟不同故障域。未来增加
Jetson Thor 后，只迁移 Field Executive、Cognition Broker 和批准的边缘模型，不重写
Mission、World Model 或机器人协议。

Cloud 与 Edge 是职责不同的独立产品角色，不作为同一个 Agent 的大小版本命名。产品角色固定为：

| 产品角色 | 部署位置 | 职责 |
| --- | --- | --- |
| `H-IDE Mission Architect` | Cloud / 客户 VPC | 理解自然语言、编写和修订高层 Mission，不持有运动权 |
| `H-IDE Field Executive` | 机器人计算背包或现场边缘域 | 唯一 Mission 执行权威、断云自治、工具调度、状态对账 |
| `H-IDE Cognition Broker` | Field Runtime | 判断本地能力与置信度，并按策略调用本地、Site Brain 或云模型 |
| `H-IDE Site Brain` | 可选现场 GPU Appliance | 多机器人共享的重模型、长期记忆、现场私有化与弱网服务 |
| `H-IDE Live Visual Resolver` | Edge / Site | 当前 RGB-D、TF、分割、目标接近点和到达证据；由现有 `h-ide-vision` 演进 |
| `H-IDE Semantic Scene Compiler` | Site / Cloud worker | 将几何、语义记忆和证据编译为 Scene proposal |
| `H-IDE Scene Publisher/Registry` | Cloud / Private control plane | 复核审核、验证摘要和权限，签名发布不可变 `SceneRevision` |
| `H-IDE Scene Explorer` | Site / Cloud 异步 worker | 对 PLY/3DGS 做多视角审计、覆盖分析、质量报告和标注候选 |
| `Robot Runtime` | G1 实时计算域 | Robot Agent、Nav2、SLAM/TF、SONIC、本体急停与看门狗 |

`Field Executive` **应该具备调用云端模型的能力**。它不是只会执行固定状态机的
“小 Agent”，而是一个受策略约束的具身任务执行器：本地先处理低延迟、低复杂度问题；
置信度不足、场景新颖或缺少模型时，通过 `Cognition Broker` 升级到 Site Brain 或
Cloud。模型返回的是带证据的候选，不会直接获得 Nav2、速度或关节控制权。

## 2. 文档导航

### 商业化目标架构

1. [目标架构与架构选择](commercial/01-target-architecture.md)
2. [产品档位、硬件与成本模型](commercial/02-product-profiles-and-hardware.md)
3. [商业授权、数据、安全与运维](commercial/03-entitlements-data-security-and-operations.md)
4. [开源参考、许可与 Build-vs-Buy](commercial/04-open-source-and-build-vs-buy.md)

### 当前条件下的 Demo

1. [现状、项目边界与关键缺口](demo/01-current-state-and-gap.md)
2. [PRO 6000 同机逻辑隔离部署](demo/02-pro6000-logical-deployment.md)
3. [Semantic Scene Compiler：从零建图与记忆](demo/03-semantic-scene-compiler.md)
4. [实施路线、测试和验收](demo/04-implementation-roadmap-and-acceptance.md)

### 跨端协议

- [核心契约、权威与故障语义](shared/contract-catalog.md)

## 3. 已冻结的关键决策

- 物理执行权只在 `Field Executive → Robot Runtime` 链路中，云模型不能直接执行。
- 断云后继续当前/缓存 Mission，并允许预授权标准巡检、停止、返航和已知对象查询；
  新的开放式复杂任务进入 `deferred_cloud`。
- Site Brain 是可选企业产品，不是每个部署的强制组件。
- 基础产品使用最小 Field Runtime；Jetson Thor 是 Edge Pro/Max 档位。
- SaaS、客户 VPC、现场私有化共享相同容器和协议，不维护三套业务代码。
- 原始 RGB-D 默认留在现场；关键帧、裁剪、掩码、Track 和 WorldDelta 按站点策略上传。
- 云服务与可离线能力包分别授权；停止、取消、急停和受支持的返航能力永不收费锁定。
- 能力升级只能来自签名模型包、Skill 包和 Capability Manifest，不允许运行时自改代码。
- FARM 与 DAAAM 先以 Shadow Provider 运行；统一 H-IDE World Model 不依赖其私有格式。
- Scene Explorer 独立于实时运动链，输出审计结果和候选标注，不直接控制机器人。
- 新场地先自动生成候选，再由人审核并发布不可变 `SceneRevision`。
- 仿真、试点和生产使用不同 Safety Profile，仿真放宽规则不能直接继承到真机。
- 客户拥有其原始传感数据和场景内容；跨客户训练必须单独授权。
- Edge 以实际执行状态为准；断网重连后不自动重放状态未知的物理动作。

## 4. 一个必须避免的误解

FARM 或 DAAAM 可以从 posed RGB-D 构建对象记忆，但它们不能单独替代：

- SLAM、定位和 `map` 坐标系；
- 占据图、ESDF、机器人足迹与实时障碍；
- 门洞净空、房间连通性和 Nav2 可达性；
- 对导航 Anchor 的审核、发布与回滚。

因此本项目真正需要新增的是 `Semantic Scene Compiler`。它让 FARM/DAAAM 的语义
输出与 SLAM/Nav2 的几何事实协作，将“模型记得这里像会议室”变成“经过几何验证、
人工发布、可供 Mission 引用的 meeting-room entity”，而不是让 LLM 猜一个坐标。

## 5. 文档约束

- 文档中的“目标架构”表示已决定的产品边界，不表示代码已经实现。
- 文档中的硬件价格是预算区间，不是采购报价；采购前必须重新询价和做热设计。
- FARM/DAAAM 的许可证结论不是法律意见；商业发布前必须完成依赖和权重审计。
- 当前仓库中的固定 Office V2 坐标仍是稳定 Demo 的回退基线，在新 Scene Compiler
  完成验收前不得直接删除。

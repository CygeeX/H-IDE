# Skillstore · H-IDE Office V2 端到端联调发行仓库

Skillstore 把 H-IDE PWA、Conversation/Mission V2、视觉解析、Robot Agent、Nav2、
SONIC 和 MuJoCo Office V2 的候选源码、跨组件锁、启动脚本与验收入口冻结在同一个
Git 版本中。它是可复刻的联调发行仓库，不是模型权重、机器人资产或运行数据库的
分发仓库。

| 发行属性 | 当前值 |
| --- | --- |
| Release candidate | `skillstore-office-v2-e2e-20260901-r1` |
| 审计分支 | `release/office-v2-e2e-20260901` |
| 目标标签 | `integration/office-v2-e2e-20260901-r1` |
| 状态 | Candidate；完成 clean-clone 与 L0/L1/L2/L3 门禁后才创建标签并更新 `main` |

真实 API key、内部令牌、账号数据库、日志、视觉证据、模型权重和本机缓存不得提交
到 Git。公开读取仓库不等于取得外部模型、网格或品牌资产的独立再分发权。

## 当前可运行能力（Current）

```text
PWA
  → H-IDE Control API / Conversation V2 / Mission V2
  → h-ide-agent（语言理解与计划提案）
  → h-ide-vision（Qwen + SAM2 + RGB-D/TF）
  → Robot Agent → Nav2 strict adapter → SONIC → MuJoCo Office V2
```

- PWA 会话首页提供两个白名单 Skill：`inspection.office`（巡检）和
  `assistant.voice`（语音助手）。
- 巡检 Skill 支持中文地点与别名、单房间或多房间顺序巡检；Mission 必须进入房间、
  停稳、多方向观察后才生成按房间汇总的报告。
- 同一会话可继续询问“刚刚有看到电脑吗？”；回答来自上一轮 Mission 的证据与
  `SemanticReferent`，不会把历史对象位置直接当成可执行坐标。
- 语音助手通过 Conversation Gateway 生成文本，并由浏览器 `zh-CN` Web Speech
  朗读。仿真配置启用时，空闲 G1 可执行一次受限挥手并自动回到正常手臂姿态。
- 三家公司机器人卡片用于界面展示；当前只有“宇树 Unitree”绑定 `g1-sim` 执行后端。
- Office V2 已移除走廊中央圆柱和可见吊灯，使用网格地板并增加可识别物件及无碰撞
  展示机器人；移动主体仍是 SONIC 驱动的 G1。
- `--full-straight-drive` 是可选导航 profile：在地图证明安全的直线段减少中途停顿，
  保持官方 SONIC 前进速度、终点停止、取消、故障与转向边界；不满足条件时自动回退。
- DAAAM 当前仅为可选、异步、Mission 后的区域级语义导入器，默认启动链不运行它。

详细演示契约见
[`components/h-ide/docs/developer/leader-demo.md`](components/h-ide/docs/developer/leader-demo.md)。

## 已决定但尚未实现的架构（Proposed）

商业化目标采用 `H-IDE Mission Architect`、`H-IDE Field Executive`、
`H-IDE Cognition Broker`、可选 `H-IDE Site Brain` 和 Robot Runtime 的云—站点—边缘—
机器人边界。当前代码尚未完成 Field Executive 独立部署、FARM/DAAAM Shadow Provider、
统一 World Model、Semantic Scene Compiler 或无硬编码新场景 onboarding。

在计算背包到位前，计划在 PRO 6000 工作站上用独立容器、网络、身份和存储模拟 Cloud、
Site 与 Edge 故障域。目标架构、当前缺口和实施门禁见
[`components/h-ide/docs/platform-architecture/README.md`](components/h-ide/docs/platform-architecture/README.md)。

## 快速入口

发布标签创建后，复刻者应做完整历史 clone 并检出精确标签，而不是浮动的 `main`。禁止
使用 `--depth`、`--shallow-*` 或 `--filter`；发布审计需要检查全部可达 Git 对象：

```bash
git clone http://192.168.50.189:37085/zhongwenjun/skillstore.git
cd skillstore
git checkout --detach integration/office-v2-e2e-20260901-r1
ROOT="$(pwd -P)"
./integration/bootstrap.sh
./integration/setup-environments.sh
```

clean audit 必须先停止其他 clone 的固定工作站 topology，并在本 clone 配置全新的
`SKILLSTORE_EXTERNAL_ROOT`。随后按[快速开始](docs/quickstart/README.md)配置隔离环境、
供应商凭据和外部资产，强制重建并运行：

```bash
./integration/preflight.sh --level L0
./integration/stop-e2e.sh
./integration/start-e2e.sh --rebuild --viewer --full-straight-drive
```

`skillstore-e2e`、`h-ide-mujoco-demo-*` 和 `h-ide-nav2-sim` 是固定 topology 名称，
因此同一工作站不能从两个 clone 并行启动本 profile。

当前 Git HTTP 地址只允许在受控内网使用。匿名 clone 不需要 token；写入凭据必须由
credential helper/askpass 提供，不能出现在 URL、命令、文档或提交信息中。
PWA 用户通过同源 server-side Session Cookie + CSRF 登录；`CONTROL_API_TOKEN` 只用于
legacy loopback/internal compatibility，不能注入前端或出现在浏览器 URL。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| `components/h-ide` | PWA、Control API、Conversation/Mission、Scene 与产品架构文档 |
| `components/h-ide-agent` | 独立 Node/Mastra 语言理解和计划提案服务 |
| `components/h-ide-vision` | Qwen、SAM2、RGB-D/TF 视觉解析服务 |
| `components/robotagent` | 机器人能力、租约、命令回执与观测网关 |
| `components/h-ide-mujoco-sonic` | MuJoCo、SONIC、ROS bridge、Nav2 strict adapter 与场景 |
| `optional/DAAAM` | 可选 DAAAM provenance 和异步导入边界 |
| `third_party` | 经记录的源码快照和补丁；不包含受限 G1 网格/模型权重 |
| `integration` | 统一 bootstrap、preflight、启动、停止、验收和跨组件锁 |
| `manifests` | 当前纳入 Git 的外部来源与资产 manifest（含 SAM2 checkpoint provenance） |
| `docs` | 快速开始、组件、运维、复刻、安全与验收文档 |

## 执行与事实边界

- LLM/VLM 只生成语言计划或带证据候选，不直接发布速度、关节命令或自造坐标。
- 当前稳定房间与站点来自已发布的 Office V2 catalog；当前画面对象的接近点必须由
  新鲜 Qwen/SAM2 + RGB-D/TF 几何生成并复核。
- Mission V2 负责审核、授权、幂等执行与终态对账；Nav2 负责地图路径，Robot Agent、
  SONIC 和 MuJoCo 负责执行与真实反馈。
- `SemanticReferent` 和 DAAAM 记忆都不是实时定位或障碍事实；需要行动时必须重新解析
  当前地图与传感器证据。

## 文档索引

- [2026-09-01 R1 候选发布说明](docs/releases/skillstore-office-v2-e2e-20260901-r1.md)
- [快速开始与管理员初始化](docs/quickstart/README.md)
- [当前组件和源码入口](docs/components/README.md)
- [当前运行、监控、复位与演示](docs/operations/README.md)
- [从干净克隆复刻](docs/reproduction/README.md)
- [L0/L1/L2/L3 验收](docs/acceptance/README.md)
- [秘密、许可证与供应链](docs/security/README.md)
- [架构与信任边界](docs/architecture/README.md)
- [商业化与 PRO 6000 Demo 目标架构](components/h-ide/docs/platform-architecture/README.md)
- [跨组件锁](integration/integration.lock.yaml)
- [变更记录](CHANGELOG.md)

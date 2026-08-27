# H-IDE / Robot Agent 本地预联调记录（2026-08-13）

**结论：开发预联调通过 / 正式验收待执行**  
**Release：** `code_complete_robot_agent_pending`（保持不变）  
**范围：** 工作站本地 D435i、真实 Robot Observation Gateway、Qwen、SAM 2、H-IDE、真实本地 Robot Agent 与 Mock Nav2。未授权物理运动，未完成正式 RA-0～RA-7 验收。

本记录是开发阶段的预联调证据，不是 `robot_agent_integrated`、Nav2 正式验收或物理机器人成功声明。运行时使用 `lab_execute`，Mock Nav2 不发布 `/cmd_vel`。

## 1. 版本与运行边界

| 项目 | 记录值 |
|---|---|
| H-IDE 基线提交 | `1b30e0a3e2ded36f9415e35f1539e1699313f66e`，工作树有未提交开发改动 |
| Robot Agent 基线提交 | `2d3d4b5ef56cfa0aa829841a90611e2c9a5b3e66`，工作树有其所有者的未提交开发改动 |
| h-ide-agent / h-ide-vision | 新建工作树，尚无可记录提交 |
| Robot ID / boot | `g1-dev` / `2fea13db-c85a-41c7-b949-7d2c5e8e9da2` |
| 成功任务的 Observation Gateway instance | `763072c7-ac3e-478a-9b46-3230aa258570` |
| Localization epoch | `q40-map-epoch-dev-1` |
| Robot capabilities digest | `sha256:038575b99105c456bd9a890fa389d0b70187be848ac1c241483ae731968b16a6` |
| 文本模型 | 普通阿里云 Model Studio API：`qwen3.8-max` |
| 视觉模型 | 普通阿里云 Model Studio API：`qwen3-vl-plus` |
| 分割模型 | SAM 2 / CUDA；FoundationPose 未启用 |

密钥只由进程环境注入，没有写入报告或仓库。Robot Agent 仓库未由 H-IDE 修改。为启动预联调，临时配置只把 ledger 移到临时目录，并将 safe-stop 超时从 `1000 ms` 改为 `3000 ms`；原配置实测 safe-stop 约需 `1.783 s`，因此原 `1000 ms` 配置仍是正式验收阻塞项。

## 2. 已得到的真实链路证据

一次分段视觉任务成功通过如下链路：

```text
真实 D435i RGB-D
  -> 真实 Robot Observation Gateway / gRPC
  -> H-IDE Vision / qwen3-vl-plus / SAM 2
  -> H-IDE VisualMissionRuntime / ActionRuntime
  -> H-IDE 入站 Robot Agent Gateway
  -> 真实本地 Robot Agent
  -> Mock Nav2 NavigateToPose
```

成功记录：

- mission：`visual-preintegration-candidates-v2-20260813`
- VisualResolution：`vr-60f7811846164db88f8b412280f89782`
- command：`0895d837-fde8-4b11-91fb-200638d8a8c1`
- ROS goal：`20fa182d-4f8b-40ba-b9f2-76646a2186ad`
- map 目标：`x=1.1778948969, y=0.0869248894, z=0, yaw=0.0757784261`
- 三帧稳定、有效深度和聚类半径门均通过；Robot Agent/Mock Nav2 返回 `SUCCEEDED`。

这条记录验证了真实组件间的坐标与命令交接，但 PlanDraft 是预联调脚本构造的视觉草案，不单独满足“自然语言 TaskRequest 起点”的 RA-5。

## 3. 完整自然语言链路结果

早期完整运行暴露了 Robot Agent 下发边界的过期观测 ACK，以及 DashScope
偶发 TLS 连接重置。H-IDE 在创建新任务前完成了以下 fail-closed 修复：

- 投影 Robot Agent `state.snapshot`（嵌套树）和 `state.update`（点号路径）；
- 导航前要求 `USABLE`、相同 observation session / localization epoch；
- 同时校验状态接收、ACK 与采集的有效年龄；
- 失败生成持久化 `REJECTED`，发送零个 `command.request`；
- Agent/Vision 模型调用只对传输错误、HTTP 429 和 HTTP 5xx 做最多三次有界重试；schema、校验、安全判断和机器人操作均不重试。

还修复了 Qwen 重复候选 ID、gRPC 与 GPU 推理共用事件循环、视觉拒绝证据、Scene API 错误归因，以及作业启动时稳定帧缓存竞态。视觉服务现在最多等待 1 秒收齐三帧，但不降低 `1000 ms` 帧新鲜度、定位新鲜度或任何几何安全门。

相机重新插拔后，系统从新的 TaskRequest 开始生成独立视觉 PlanDraft，经人工审核等价的 API 确认后完成全链路：

- 自然语言：`定位当前头部相机画面中央唯一的银白色人形机器人，生成一个可审核的视觉目标点`
- conversation/task：`conversation-817d0e5d591044039db237582ff842c6` / `task-faf68c6d687649d09a913ce641192e41`
- mission/draft：`draft-85a6a252edea41658773050c29a86de8`
- VisualResolution：`vr-def5770bfca848b0b2d296df59b6b880`
- RGB 证据 SHA-256：`ccc27c7c89abddce5692d1e753103e943c7793611f9e8b5f3b4c20a23962cd1a`
- landmark：`map x=2.6109231690, y=-0.1533292391, z=0.7949386473`
- approach 目标：`map x=1.8627605066, y=-0.1008637200, z=0, yaw=-0.0700112055`
- command：`d5b33e08-f67a-427d-ad75-fe8e15e2e052`
- ROS goal：`63e9ec18-d614-4210-9bad-f15b03cfe287`

D435i（串号 `261222077185`，USB 3.2）恢复了约 30 Hz 的彩色与对齐深度流。Qwen3-VL 返回唯一目标，SAM 2 分割和三帧几何校验全部通过：有效深度比例 `0.99963`，map 聚类半径 `0.01172 m`。Robot Agent ledger 记录 `RECEIVED → RUNNING → SUCCEEDED`，Mock Nav2 收到相同的 `x=1.863, y=-0.101` 目标。ROS 图中不存在 `/cmd_vel`，因此本轮没有产生实体速度指令。

这证明开发链路已具备互操作性，但不证明导航精度或物理到达；本轮定位使用开发 fixture，Mock Nav2 的成功结果也是模拟结果。

### 3.1 用户原始指令与 PCD 开发代理复验（17:30 CST）

在保持 `lab_execute`、真实 D435i、真实本地 Robot Agent 与 Mock Nav2 的前提下，
再次从用户原始指令开始运行：`找到银白色的机器人并去机器人的旁边`。本轮先修复
了 Supervisor 将可见目标误判为“缺少地图坐标”的问题；路由规则只对带明确静态目标
的视觉寻找/靠近请求确定性进入 `vision-worker`，抓取、跟随、探索、显式坐标及命名
PCD 地点仍交回 Supervisor。结果如下：

- conversation/task：`conversation-4c57fd105cb5438c87b5dfeb07fed55d` / `task-b6f91ffe9efb4f08843b612991789a72`
- mission/draft：`draft-52d5dba11a394cd68a4dae81eff5baa6`
- VisualResolution：`vr-b1376f6bb3d6488091f625a2febde63b`
- observation session：`690d90c9-4320-4acf-a073-72b3af4d37a0`
- approach 目标：`map x=1.6700495766, y=-0.0452945370, z=0, yaw=-0.0382984318`
- command：`0762c05c-be74-49ca-ba53-ccf821a7cd6e`
- ROS goal：`c8afa421-9099-45b9-8f57-b64974912934`
- 最终结果：VisualMission `completed`，Robot Agent / Mock Nav2 `succeeded`；一个任务只产生一个命令。

同轮修复了 Vite 开发服务器的 API 分流：控制、场景、场景绑定及
`/scene-assets` 现在由同源页面分别代理到 Control API / Scene API，Scene token 仅在
Vite 服务器侧注入。通过 5173 上传的 27 点 PCD 已转换为 8 点 Potree 预览，
`metadata.json` 与资产会话均返回 200。该微型场景只验证导入/转换/预览通路，不用于
视觉任务推理；本轮视觉任务采用 `vision_only`，没有场景绑定也可工作。

## 4. 当前验证结果

- H-IDE 必要回归：23 项通过（视觉任务、Robot Agent 网关、Scene client）。
- H-IDE Vision：26 项通过；Ruff 与所改源码的严格 mypy 通过。
- 模型有界重试新增验证：Agent 13 项聚焦测试及 typecheck 通过；Vision Qwen 7 项聚焦测试及 Ruff/严格 mypy 通过。
- Robot Agent 隔离 ROS Domain 聚焦回归：55 项通过，覆盖观测协议、gRPC/mTLS、ROS bridge、观测准入、ROS 执行及 NavigateToPose→Mock Nav2；未修改 Robot Agent 源码。
- PWA 视觉任务：3 项通过；生产构建通过。
- 工作站预检负向测试：观测 `DEGRADED` 时持久化拒绝且零 `command.request`。
- 相机恢复后的正向测试：一个自然语言任务对应一个审核后 mission、一个不可变 resolution、一个 Robot Agent command 和一个 Mock Nav2 ROS goal，最终 `SUCCEEDED`。

仍未完成：工作树版本封存、原 Robot Agent safe-stop 配置修复及其所有者复核、语音 RA-5、取消/断线/重启矩阵、精度与 P95 性能、正式证据 manifest，以及任何物理运动。因此 [robot-agent-deferred-validation.md](robot-agent-deferred-validation.md) 继续保持 `FROZEN`，release label 保持不变。

# H-IDE Mission V2：连续 Supervisor–Worker 架构

## 目标

Mission V2 将 H-IDE 从“一次提问、一次 PlanDraft、一次执行”改为可持久化的连续任务控制面。H-IDE 负责语义任务、人工授权、世界事实、事件和重规划；它不规划路径，也不直接控制底盘。

```text
PCD / 文本 / 语音
        ↓
统一 MissionBrief + MissionEnvelope
        ↓
Mastra Mission Executive（Qwen3.8-Max）
        ↓
语义 Mission DAG（一个或多个专属技能）
        ↓
人工审核授权边界
        ↓
H-IDE MissionRuntime 持续推进
        ├─ vision.observe → H-IDE Vision → Qwen3-VL-Plus 假设
        │                                  ↓
        │                         RGB-D + TF 确定性几何事实
        ├─ PCD 场景语义 → Scene API 静态先验事实
        └─ 单个可信 map 目标位姿
                         ↓
                 ActionRuntime
                         ↓
                 Robot Agent
                         ↓
        Nav2 NavigateToPose / Spin / 本地 BT 与安全
```

## 权威边界

| 层 | 可以决定 | 不可以决定 |
|---|---|---|
| Qwen / Mastra | 任务语义、技能选择、DAG、重规划候选 | 坐标事实、路径、速度、ROS 接口、是否安全 |
| H-IDE Vision | 视觉假设、RGB-D 投影、TF 变换、几何候选与证据 | 路径、底盘控制、最终任务授权 |
| H-IDE MissionRuntime | 审核、事件、修订、事实选择、技能签发、距离预算 | Nav2 路径和底层运动 |
| Robot Agent | 能力清单、命令生命周期、ROS 适配、本地安全 | 改写已审核 MissionEnvelope |
| Nav2 / BehaviorTree.CPP | 路径、避障、恢复、NavigateToPose、Spin | 自然语言任务和云端视觉推理 |

机器人的头和 RGB-D 相机固定。任何底盘位移或朝向改变，包括原地扫描，都必须使用 Robot Agent 发布且 `executor=nav2` 的能力。H-IDE 拒绝 `cmd_vel`、线/角速度、路径和航点输入。

## 核心契约

- `MissionEnvelope`：冻结 Robot Boot、Localization Epoch、RobotProfile、CapabilityManifest、工作站能力、模型版本、场景版本、距离/速度/时长/重试边界。
- `MissionPlanDraft`：只含语义技能节点，禁止模型输出可执行几何。
- `PlanRevision`：完整替换 DAG；低风险且仍在原授权范围内的修订可继承策略，其余必须人工审核。
- `WorldRecord`：区分 `hypothesis` 与 `fact`。`source=qwen` 永远不能直接成为事实。
- `SkillRequest`：绑定 Mission、Revision、Node、Authorization、能力版本和幂等键。
- `SkillProgress` / `SkillResult`：驱动状态机继续、等待、重规划或结束。

SQLite 以单调事件序列和比较交换事务持久化草案、授权、修订、节点状态、技能执行与世界记录。进程重启不会重放未确认的物理动作；活动 Mission 会转入显式恢复状态。

过期是签发策略而不是历史数据格式错误：草案批准、Mission 启动和每次技能签发都会检查当前授权时间窗；已经过期的 Plan、Authorization 和 SkillRequest 仍可解析，用于审计、查询与取消，不能因此变成无法恢复的坏记录。

## 视觉与坐标

`vision.observe@2.0.0` 是工作站能力，不是 Robot Agent 能力。支持：

- `object_approach`：定位目标并计算保持安全距离的候选点；
- `landmark_observation`：只记录地标，不产生移动目标；
- `directional_free_space`：依据深度图判断前/左/右有限距离的自由空间；
- `arrival_verification`：到达后进行视觉验证。

Qwen3-VL-Plus 只返回标签、像素框和置信度。H-IDE Vision 使用对齐深度、相机内参、固定外参和 `map` TF 产生几何候选；只有全部确定性安全门通过的候选才写成 `source=geometry, kind=fact`。MissionRuntime 只从与当前视觉依赖分析 ID 绑定的事实取目标点。

## PCD 场景

PCD 继续作为辅助工具，不取代实时视觉和 Nav2：

1. Scene API 只读取已发布且版本固定的场景。
2. 语义区域作为 `semantic_region` 静态事实提供给 Agent。
3. 区域入口点只有在当前 Scene Map 检查为 `observed_free` 时，才额外形成 `target_pose` 场景事实。
4. 人工审核后，H-IDE 仍只发送一个目标位姿；Robot Agent/Nav2 负责当前障碍、路径和恢复。

因此“去厨房”可以使用已标注厨房的入口点；“去银白色机器人旁边”则优先走实时视觉。若没有可信坐标，Mission 进入 `needs_input`，不会由 LLM 猜坐标。

## 连续推进与重规划

`POST /api/v2/missions/{id}/advance` 是有界协调循环：

1. 启动已审核 Mission；
2. 选择一个依赖就绪节点；
3. 从语义意图与可信事实构造严格技能输入；
4. 签发一次幂等 SkillRequest；
5. 同步结果立即推进，异步结果停止本轮并等待下次对账；
6. 对同一 Robot Agent 命令只执行 `query`，不重发；
7. 成功解锁下游节点，失败触发 Mastra 事件评估和 PlanRevision；
8. 缺少事实时进入 `needs_input`，超出策略时进入 `awaiting_review`。

运行中的物理技能完成或明确取消前，不接受改变目标的对话，避免新计划与仍在执行的 Nav2 Goal 并存。

取消遵循相反顺序：H-IDE 先以原 `SkillRequest.requestId` 请求 ActionRuntime / Robot Agent 取消活动 Nav2 Goal，获得终态回执后才把 Mission 标记为 `cancelled`。若进程重启后丢失物理命令所有权，接口会明确拒绝宣称取消成功并要求对账。

`SkillRequest.requestId` 是 H-IDE 的可审计逻辑 ID；由于 `robot-agent.v1`
的线协议要求 UUID，Gateway 仅在传输边界对 `skill-` 命名空间做稳定
UUID5 映射。重试、查询和取消使用同一映射；若发生进程重启，仍须遵循
冻结的对账流程。回执同时保留 Robot Agent 的 `robotAgentCommandId`，不在
Mission 数据库中维护第二个易失性命令身份。

每次 Nav2 `succeeded` 后，H-IDE 将已达到的单一目标位姿记录为下一段距离校验的保守当前位置；若 Robot Agent 后续提供更新鲜的 `map` 定位，可覆盖该投影。它不改变 Nav2 对实际路径与到达容差的权威。

首次导航不使用虚构原点。Robot Agent 从 `/localization/pose` 将 `map` 位姿放入现有 `state.snapshot/state.update`，Gateway 归一化四元数并投影 `currentPose`；新鲜度绑定到发生定位字段变化的 Robot Agent UTC 状态时间（不把可能处于 ROS 仿真时钟的 header stamp 当作墙上时钟），H-IDE 默认只接受 2 秒内且与当前 Boot/Epoch 绑定的位姿。无关遥测不会刷新定位时间；缺失或过期时，含导航节点的草案不能获批。

## 扩展新功能

Robot Agent 先发布版本化 `CapabilityDescriptor`。H-IDE 为非内置技能注册可信输入解析器，Mastra 才能在 Mission DAG 中选用它。运动类能力必须声明 `executor=nav2`；未知能力不会把 Agent 的自由文本直接透传到机器人，而是进入 `needs_input`。

BehaviorTree.CPP 选择性保留在 Robot Agent/Nav2 的局部执行层。PlanSys2 暂不引入；只有当跨导航、抓取、门、电梯等长期符号域稳定后再评估。SkiROS2 的技能描述和世界模型思想用于契约设计，但不成为当前运行时依赖。

## 主要 API

- `POST /api/v2/missions/drafts`
- `POST /api/v2/missions/drafts/{draftId}/approve`
- `POST /api/v2/missions/drafts/{draftId}/reject`
- `POST /api/v2/missions/{missionId}/advance`
- `POST /api/v2/missions/{missionId}/messages`
- `POST /api/v2/missions/{missionId}/cancel`
- `POST /api/v2/missions/{missionId}/revisions/{revisionId}/approve`
- `GET /api/v2/missions/{missionId}`
- `GET /api/v2/missions`

旧 V1 Assistant / VisualMission 在 V2 人工验收前仅作为迁移兼容代码保留，不再是首页新任务的默认入口。验收通过后按删除清单移除，而不是长期维护两套执行权威。

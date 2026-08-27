# 架构与信任边界

## 1. 端到端职责

| 层 | 组件 | 允许做的事 | 明确不能做的事 |
| --- | --- | --- | --- |
| 交互 | `components/h-ide` PWA/Control API | 会话、Mission 草案、授权、事件流、场景绑定 | 直接发布 ROS/运动命令 |
| 规划 | `components/h-ide-agent` | 意图分类、Worker 编排、生成 `PlanProposal` | 访问机器人执行工具、发坐标或越过人工审核 |
| 视觉 | `components/h-ide-vision` | Qwen 语义识别、SAM2 分割、RGB-D/TF 几何候选 | 路径规划、速度控制、机器人凭据 |
| 机器人边界 | `components/robotagent` | 能力、租约、观察快照、命令回执、取消 | 让未授权计划直接执行 |
| 导航 | Nav2/strict adapter | map 坐标路径、旋转/行走原语、可达性 | 取代 Mission 授权或 SONIC 控制回路 |
| 运动 | `h-ide-mujoco-sonic` | MuJoCo Plant、SONIC、ROS bridge、遥测 | 让 LLM 直接控制关节 |
| 记忆（可选） | DAAAM | Mission 后的区域/语义先验 | 实时定位、障碍事实、运动授权 |

## 2. 一次行动的时序

```text
用户文本
  │
  ├─普通问答/观察 ───────────────► 事实或回复，不创建运动 Mission
  │
  └─行动意图
       │
       ▼
  h-ide-agent：Supervisor → Worker → Structuring
       │ 只返回 PlanProposal（含语义地点/视觉目标，不含自造坐标）
       ▼
  H-IDE：SceneContext + 能力 + 当前 pose/epoch 复核
       │
       ▼
  PWA：用户确认 draft
       │
       ▼
  Mission V2：授权、幂等 dispatch、进度/事件流、终态对账
       │
       ▼
  Robot Agent → strict Nav2 adapter → ROS primitive
       │                                  (Spin/DriveOnHeading/pose)
       ▼
  SONIC 50 Hz → MuJoCo Plant → odom/TF/RGB-D/telemetry
```

任何一层失败都必须保留可诊断的状态；不能把“进程存活”误报为“运动完成”。
取消和安全停机路径独立于 LLM、Vision 和场景 Worker。

## 3. 坐标与可信事实

1. Office V2 `scene.yaml` 生成 MuJoCo XML、Nav2 PGM/YAML 和 Scene API catalog；
   `tools/build_scene.py --check` 保证它们来自同一个场景源哈希。
2. 房间/门/静态站点是已发布 SceneMemory 的可信语义事实，不由模型临时发明。
3. “走到当前画面中的架子前”先由 Qwen 抽取对象语义，再由 SAM2 + RGB-D +
   capture-time TF 计算候选；候选必须带 `visualResolutionId`、传感器、定位 epoch、
   新鲜度和几何置信信息。
4. H-IDE 在草案生成和确认两个时点重新校验场景 revision、机器人 pose 和
   localization epoch。视觉候选是待验证假设，不是地图坐标的替代品。

## 4. 模型与可执行 profile

`config/sonic-models.lock.json` 将 SONIC 权重、控制器源文件、观察契约和 runtime
profile 绑定。当前可执行的是 `sonic-python-experimental`（默认 SONIC revision
和 vendored LeRobot snapshot）；官方 C++ profile 只有完整 bundle provenance，
没有宣称已接入 H-IDE 运动执行。启动前 verifier fail-closed，任何大小、哈希、
来源或配置 selector 漂移都会拒绝 motion-capable 进程。

## 5. DAAAM 的正确位置

DAAAM 只作为异步 post-Mission importer。它可以帮助“上次某区域常见什么”这类
语义先验，但不能回答“机器人此刻在哪里”“前方是否可通行”，也不能改变 Nav2
路径、SONIC 速度或安全停机。完整 research weights/datasets 不在本仓库。

## 6. 参考的开源思想

- NVIDIA GEAR-SONIC/GR00T：50 Hz whole-body policy、官方 observation contract
  和模型/控制器成套锁定。
- Nav2：生命周期管理、规划器/控制器分层、NavigateToPose/Spin 等 action 语义；
  本项目用 strict adapter 把它约束到已验收的底盘原语。
- LeRobot：机器人/策略模块化和 SONIC Python 控制器实现；当前 snapshot 作为
  普通 monorepo 文件并用内容哈希锁定。
- ROS 2/CycloneDDS：仿真 ROS domain 232 与 Unitree domain 230 隔离，避免接触
  live domain 231。
- DAAAM：空间-语义记忆的异步导入思想，仅取其 observation-only 部分。

详细的接口字段和源码入口请先看各组件 README，再看 `integration/integration.lock.yaml`。

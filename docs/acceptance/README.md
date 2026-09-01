# L0 / L1 / L2 / L3 验收标准

## 1. 分级定义

| 等级 | 证明什么 | 不证明什么 | 是否允许自动运动 |
| --- | --- | --- | --- |
| L0 静态 | Git 内容、秘密边界、manifest、脚本语法和路径可移植 | 依赖可安装、服务可启动 | 否 |
| L1 单测与构建 | 关键合同/单测、Agent/PWA 构建、Compose 和场景生成一致 | GPU、ROS、模型和 live motion | 否 |
| L2 Readiness | GPU、模型、ROS、Plant、SONIC、Nav2、Vision、Robot Agent 全链健康 | 任一自然语言 Mission 实际成功 | 否 |
| L3 Mission | 操作者明确选择的自然语言任务完成，并保存物理/视觉/报告证据 | 真机安全认证、未知场地泛化 | 仅人工审核后允许 |

`integration/acceptance.sh` 永远不会仅因选择 L2/L3 就自动发送运动。改变机器人状态
需要操作者在 PWA 审核具体 Mission，或显式运行一个有名称、目标和证据目录的用例。

## 2. L0：仓库静态验收

```bash
./integration/preflight.sh --level L0
./integration/acceptance.sh --level L0
```

必须满足：

- clean clone 中无 gitlink、开发机绝对路径、数据库、日志、Evidence、模型权重或秘密；
- Release Manifest、vendor manifest、JSON/YAML 和 shell 可解析；
- `.env` 不入 Git，存在时权限为 `0600`；
- G1、SONIC、SAM2 等不可再分发资产只记录来源、版本、哈希和安装方法。

## 3. L1：单元测试、合同和构建

主入口：

```bash
./integration/preflight.sh --level L1
./integration/acceptance.sh --level L1
```

发布负责人还应确认以下构建边界，不能只保留一次本机缓存结果：

```bash
(cd components/h-ide-agent && npm test && npm run typecheck && npm run build)
(cd components/h-ide/apps/control-pwa && npm test -- --run && npm run build)
(cd components/h-ide-mujoco-sonic && .venv/bin/python tools/build_scene.py --check)
```

Robot Agent 的 ROS 相关测试必须先加载与发布一致的 Jazzy/overlay 环境；不要通过禁用
pytest-asyncio 或遗漏 `rclpy` 把 collection error 误记为代码失败：

```bash
(
  cd components/robotagent
  source /opt/ros/jazzy/setup.bash
  source tools/activate-local-nav2-jazzy.bash
  source install/setup.bash
  .venv/bin/python -m pytest -q \
    tests/test_safety_core.py tests/test_runtime.py \
    tests/test_motion_fault.py tests/test_coordinator.py
)
```

L1 失败时不得进入 live 服务验收。

## 4. L2：服务、ROS、GPU 和模型 Readiness

```bash
./integration/preflight.sh --level L2
./integration/start-e2e.sh --viewer
./integration/acceptance.sh --level L2

SKILLSTORE_ROOT="$PWD" \
  ./components/h-ide-mujoco-sonic/scripts/workstation-demo-status.sh
```

L2 至少确认：

- PWA/Control API、Planning Agent、Scene API 和 Redis/PostgreSQL 依赖可用；
- H-IDE Vision 明确报告 `qwen3-vl-plus + sam2` 可用，不是 fixture provider；
- MuJoCo Plant、RGB-D、SONIC telemetry、Robot Agent diagnostics 健康；
- Nav2 lifecycle 为 active，ROS Domain 为仿真 `232`，Unitree DDS 为 `230`；
- Robot Agent 的 boot/session、Capability Manifest、Localization Epoch 与 Scene binding 一致；
- strict adapter 的 profile（默认、continuous 或 full straight）与本轮证据记录一致；
- readiness 检查没有发送导航、旋转、挥手、reset 或其他改变状态的请求。

## 5. L3：明确的自然语言 Mission

### 通用前置条件

1. PWA 选择 `g1-sim`，显示 Office V2 published Scene binding 和 fresh observation；
2. 新建会话，清除或终止旧 Mission；
3. 每个物理计划都先查看 DAG，再由操作者点击确认；
4. 保存 conversation/turn/Mission ID、SceneRef、最终 Pose、Robot Agent/Nav2 回执、
   SONIC/Plant boot ID、视觉 Evidence SHA-256 和报告；
5. 任一用例失败后先诊断，不能把手工拖拽、瞬移或旧 Evidence 当成功。

### L3-01：双房间巡检和最终报告

选择“巡检 Skill”，输入：

> 今天巡检开放办公室和会议室

验收：同一 Mission 包含两个完整房间段，每段均为进入内部、inside-region 验证、
多方向扫描；最后只有一个 `selected_locations` 报告。报告按开放办公区、会议室分组，
包含到达后采集的新鲜 Evidence，不能漏观察、重复走位后误报完成或只描述初始画面。

### L3-02：报告记忆中的电脑和椅子

L3-01 完成后，在同一会话切换“语音助手 Skill”，依次输入：

> 刚刚有看到电脑吗？

> 刚刚有看到椅子吗？

验收：回答只使用刚完成报告及其 Evidence，能把 `monitor` 等已确认类别归一为用户所说
的电脑，并说明对应房间；报告中已出现 `chair` 时不能回答“没有确认看到”。这两个
问题不创建新的运动 Mission，不重新使用当前画面冒充历史证据。

### L3-03：语音回复、挥手和手臂回落

机器人静止时输入：

> 你叫什么名字？

验收：PWA 展示并以 `zh-CN` 语音朗读同一文本；启用 leader-demo gesture profile 时，
只向 `g1-sim` 请求一次 simulation-only wave。挥手只在 stationary/IDLE 接受，由 SONIC
唯一 50 Hz LowCmd owner 叠加；默认 2.4 秒后 envelope 回到零、目标恢复到 SONIC 基础
姿态，telemetry 为 `completed`。若期间开始导航或停止流程，gesture 必须取消并回到底层
目标，不能挂住右臂或影响下一 Mission。该结果只证明仿真动作，不代表真机动作 ID 已接入。

### L3-04：`full_straight_drive`

冷启动时显式选择 profile：

```bash
./integration/stop-e2e.sh
./integration/start-e2e.sh --viewer --full-straight-drive
```

选择一个已审核、宽净空的直线路段执行导航。验收：strict adapter health 显示
`fullStraightDrive=true`；通过 0.90m clearance contour 的直线最多以 4.0m 单个
`DriveOnHeading` 执行。长路径分成少量 4m 动作；门洞、墙边、转弯或验证不通过时自动
退回 2m continuous（0.70m contour），再退回 1.0m tight profile。终点仍需正常
stop/IDLE/终态回执，
不得把 full mode 描述成无限距离开环行走。

### L3-05：完整复位后重跑

Office V2 的正式复位是进程级冷启动，不是在运行中改 `qpos`：

```bash
./integration/stop-e2e.sh
./integration/start-e2e.sh --viewer --full-straight-drive
```

确认新 Plant boot、Robot Agent boot/session 和 Localization Epoch 后，新建会话并再次执行
L3-01。第二次 Mission 必须使用新身份和新 Evidence；旧 Mission 不能继续、未知物理动作
不能重放，报告结果仍满足同一验收条件。

### 可选基础运动回归

- 原地左右转向并回到稳定 IDLE；
- 直行与停止；
- 左右横移；
- 静态绕行；
- 单房间入口与内部到达。

这些用例也必须逐个显式审核，不能由通用 health script 自动猜目标。

## 6. 失败判定

以下任一情况不得标记 L3 成功：

- PWA 显示完成，但 Robot Agent/Nav2 没有相同 request 的终态回执；
- 巡检要求到房间后观察，却复用了到达前或上一 Mission 的画面；
- 双房任务缺少任一房间的导航、inside verify、scan 或最终报告节点；
- 报告含电脑/椅子，后续问答却因记忆投影丢失回答“未确认”；
- 候选有歧义但 PWA 没有展示 Evidence 和可选 ID；
- Plant、Robot Agent boot 或 Localization Epoch 改变后仍继续旧任务；
- `full_straight_drive` 绕过地图验证、终点停止或 fallback；
- 挥手结束后手臂未回落，或与导航并发占用执行边界；
- 人工拖拽、辅助带恢复、瞬移或旧日志被当作自然完成证据。

## 7. 发布门禁

- clean clone 的 L0、L1 通过；
- 目标主机 L2 通过且没有自动运动；
- 涉及本轮展示能力的 L3-01 至 L3-05 有可追溯证据；
- `integration/integration.lock.yaml`、SBOM、vendor/模型/场景哈希和 release notes 已更新；
- 已知问题如实记录，不把 skipped、hang 或单独通过写成全量 suite 通过；
- 远端审计分支验证后，`main` 与不可变 release tag 指向同一提交。

架构迁移的第二场景、断云和 Provider OOM 验收不属于当前 L3；见
[PRO 6000 Demo 架构迁移路线](../architecture/platform-roadmap.md)。

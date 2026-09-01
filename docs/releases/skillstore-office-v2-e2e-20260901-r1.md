# skillstore Office V2 E2E 2026-09-01 R1 发布说明

| 属性 | 值 |
| --- | --- |
| Release ID | `skillstore-office-v2-e2e-20260901-r1` |
| 目标 tag | `integration/office-v2-e2e-20260901-r1` |
| 发布状态 | Release candidate；L0–L3 和远端审计完成后创建不可变 tag |
| 上一稳定 tag / 回退点 | `integration/office-v2-e2e-20260827-r3` |

## 1. 组件来源

| 组件 | Source commit | 角色 |
| --- | --- | --- |
| H-IDE | `da1b83ab9793301250c6830ca23562f4a1f0b8d5` | PWA、Control API、Mission V2、Scene API |
| H-IDE Agent | `09c727678f249c11efadac48f727f7c8d103a22e` | 会话分类、planning-only Mission Architect |
| H-IDE Vision | `072a49f9741bed54805656c0358dde54fc998a0f` | Qwen3-VL-Plus、SAM2、RGB-D/TF 和 Evidence |
| Robot Agent | `f544f40c9c4ea7e1b6a35105918d12aa3a24900a` | 能力、租约、命令账本、ROS/观察边界 |
| H-IDE MuJoCo SONIC | `6c888673aaab2aa387b1dd02e2beac5a3dd4989a` | Office V2、Nav2、SONIC、MuJoCo、展示动作 |
| DAAAM（可选） | `ebd6e3b89763849eb7bce7f75e8d08550ea7344a` | post-Mission、observation-only importer |

H-IDE Vision 本轮沿用已冻结 source commit。FARM 不属于本发行版 runtime；它只出现在
后续 Shadow Provider 路线中。Skillstore 根 commit 在正式 tag 创建时由
`integration/integration.lock.yaml` 记录，不能预填一个尚未包含本发布说明的中间 commit。

## 2. 本轮功能

- PWA 初始页提供“巡检 Skill”和“语音助手 Skill”，机器人选择使用独立模态框；
- “今天巡检开放办公室和会议室”进入同一确定性复合 Mission，按房间完成进入、
  inside-region 验证、多方向观察和最终报告；
- 同一会话可依据已完成巡检报告回答电脑、椅子等历史问题，不用当前帧冒充历史观察；
- 普通语音助手回复可由浏览器中文朗读，并在启用仿真 profile 时请求一次 SONIC-owned wave；
- wave 只在 stationary/IDLE 接受，轨迹有平滑包络，结束后恢复基础手臂目标；
- `full_straight_drive` 为独立 opt-in profile：通过 0.90m clearance contour 的直线
  最多 4m 单 action，失败按 2m continuous（0.70m contour）→ 1m tight 回退；
- Robot Agent 对仿真可重试的 control-arm settling 失败不再永久 latch session，
  SONIC resident locomotion authority 保持可重新获取；
- Office V2 删除中央圆柱和可见吊灯，使用 checker 地板、增加可识别物件和三个
  无碰撞、无执行器的站立 G1 展示模型；
- MuJoCo 长步/连续巡检、报告 Evidence、PWA Mission 流式反馈和复位冷启动链保持可审计。

## 3. 本轮验证记录

### 已完成

H-IDE Agent：

```bash
cd components/h-ide-agent
npm test && npm run typecheck && npm run build
```

- Vitest：`16` 个 test file、`114` 个 test 全部通过；
- TypeScript typecheck 通过；
- Mastra production build 通过。

Robot Agent 定向回归（加载 ROS Jazzy 和本地 overlay）：

```bash
cd components/robotagent
source /opt/ros/jazzy/setup.bash
source tools/activate-local-nav2-jazzy.bash
source install/setup.bash
.venv/bin/python -m pytest -q \
  tests/test_safety_core.py tests/test_runtime.py \
  tests/test_motion_fault.py tests/test_coordinator.py
```

- `63 passed in 7.53s`。

### 正式发布前仍需

- clean clone L0、L1；
- 目标工作站 L2 readiness；
- [L3-01 至 L3-05](../acceptance/README.md) 的 Mission/Evidence 记录；
- 更新后的 integration lock、SBOM、模型和场景 digest；
- 远端审计分支验证并将 `main` fast-forward 到同一 root commit。

## 4. 已知问题和边界

### Robot Agent 扩大测试集的顺序性 hang

扩大执行：

```bash
.venv/bin/python -m pytest -q -m 'not soak and not integration'
```

整组运行在 `tests/test_agent_observation.py` 末尾附近停止输出，人工中断时结果为
`41 passed, 18 deselected in 201.90s`。疑似末项
`test_observation_watchdog_selectively_cancels_vln_and_strips_binding_params` 单独运行通过：
`1 passed in 1.86s`。因此本发布只声明 Robot Agent 定向回归 63/63 通过，不声明完整
non-soak suite 通过。该顺序/资源清理问题需单独排查，但没有证据表明上述 runtime/session
变更的定向用例失败。

### 产品边界

- Office V2 的房间 Polygon、入口、站点、扫描方向和路线仍来自场景源；换场地仍需开发者
  修改并重建，Semantic Scene Compiler 尚未实现；
- MuJoCo/Nav2 loader、strict planner 和 launcher 仍绑定 Office V2；
- DAAAM 只是 Mission 后 importer，FARM runtime 集成为 0；
- 当前物理 Mission 执行推进仍位于 H-IDE Control API/workstation，Field Executive 和
  Cognition Broker 尚未实现；
- wave 只属于 `g1-sim` 仿真，不是 Unitree 真机动作能力；
- `full_straight_drive` 是 opt-in profile，不会绕过地图验证、终点停止或 fallback；
- DeepSeek/DashScope 的欠费、限流或供应商模型更新仍会影响模型型任务可用性；
- 同机 Demo 不证明 Thor ARM64、真机 SLAM、功耗/散热或生产安全认证。

后续调整见 [PRO 6000 Demo 架构与代码迁移路线](../architecture/platform-roadmap.md)和
[完整平台架构文档](../../components/h-ide/docs/platform-architecture/README.md)。

## 5. 回退

回退点是不可变 tag：

```text
integration/office-v2-e2e-20260827-r3
```

先停止完整链路，再在单独 worktree 或干净克隆中检出旧 tag：

```bash
./integration/stop-e2e.sh
git switch --detach integration/office-v2-e2e-20260827-r3
./integration/bootstrap.sh
./integration/preflight.sh --level L2
./integration/start-e2e.sh --viewer
```

不要删除或复用新版本正在写入的运行数据库/Evidence 来伪造回退结果；生产式回退应使用
与 tag 对应的配置/volume snapshot。若只需关闭 `full_straight_drive`，可停止后省略该启动
参数，仍需整条运动链冷启动以生成一致的新 boot/session/epoch。

正式发布后，`main` 与 `integration/office-v2-e2e-20260901-r1` 必须指向同一 root commit；
标签创建后不得移动。

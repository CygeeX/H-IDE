# H-IDE × Robot Agent × Nav2 × SONIC × MuJoCo 联调基线

## 当前可运行链路（2026-08-19）

```text
浏览器聊天 / Mission V2 计划审核
  → H-IDE Control API
  → h-ide-agent（Mastra Supervisor，生成语义 Mission DAG）
  → 人工批准 PlanDraft / 每次 PlanRevision
  → g1-sim Robot Agent Gateway
  → Robot Agent（能力、身份、观测新鲜度和幂等校验）
  → Nav2 NavigateToPose / Spin
  → /cmd_vel
  → 速度守卫 + SONIC 重规划门控
  → SONIC 29-DoF 关节目标
  → MuJoCo Unitree G1 与单层办公室
  → 根位姿 + RGB-D + CameraInfo + TF
  → Observation Gateway → H-IDE Vision / 网页证据
```

导航目标和原地改变朝向都必须经过 Nav2。H-IDE 不发 `/cmd_vel`，SONIC
不接收地图目标，机器人固定头部不执行转头命令。VLM 只能提出语义或图像区域
假设；可信目标坐标来自 MuJoCo RGB-D/TF、场景事实或人工批准的相对位移解析。

## 启动与状态检查

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
scripts/start-workstation-demo.sh

# 修改过 H-IDE 前后端源码时：
scripts/start-workstation-demo.sh --rebuild

# 只检查，不改变运行状态：
scripts/workstation-demo-status.sh
```

启动脚本会幂等地管理 H-IDE Docker 服务、Nav2 容器、MuJoCo/SONIC、ROS
命令与观测桥、仿真 Robot Agent Gateway、`g1-sim` Robot Agent 和 Observation
Gateway。当前联调配置还会在宿主机 GPU 环境启动真实
`QwenVlmProvider + qwen3-vl-plus + SAM 2`，CPU 容器中的 Oracle 只保留在隔离端口
用于故障诊断。运行日志位于
`${XDG_RUNTIME_DIR:-/tmp}/h-ide-mujoco-demo-$UID`。

## 网页操作

1. 打开 `http://127.0.0.1:8080/`，右上角机器人选择 `g1-sim`。
2. 点击右上角“工具 → 仿真观测台”，或直接打开
   `http://127.0.0.1:8080/simulation`。
3. 选择最新的 `mujoco_sonic` 会话；页面应显示第一视角 RGB、实时根位姿、
   Nav2/SONIC 状态、速度和轨迹。桌面上的 MuJoCo 窗口是第三人称物理视图。
4. 回到聊天首页，输入“向前移动 0.3 米后停止”。页面只生成 Mission 草案。
5. 检查节点是 `navigation.navigate_to_pose`、目标机器人为 `g1-sim`，然后点击
   “确认计划并执行”。
6. Mission 卡片依次显示执行中、技能回执和已完成；同时观察第三人称行走、
   第一视角画面变化及仿真观测台轨迹更新。

推荐的非破坏性验证提示词：

- `只观察当前画面，描述桌子、显示器和黄色杯子，不移动底盘。`
- `只观察并定位办公场景中桌子右侧地面上唯一的小橙色球，忽略黄色圆柱和所有家具；禁止移动底盘，禁止旋转，禁止头部运动，不要生成任何导航节点。`
- `向前移动0.3米后停止。`
- `原地向左转30度后停止。`（仍由 Nav2 Spin 执行）
- `观察前方桌子，然后移动到离桌子安全距离的位置。`（用于视觉链，先审核）

## Oracle 与 VLM

- `oracle` 只在仿真中使用 MuJoCo 已知的物体、障碍和根位姿，适合验证通信、
  坐标、Nav2、连续状态机与安全闸门。它不等于 FoundationPose，也不应用于真机。
- `qwen` 使用 Qwen VLM 理解 MuJoCo 第一视角语义，再由本地 SAM 2、RGB-D/TF
  计算候选目标点。
  VLM 不拥有坐标、路径或执行权。无 DashScope 凭据时服务必须显式降为 fixture，
  不能把 fixture 结果标成真实 Qwen 验收。

当前网页联调运行真实 `qwen3-vl-plus`；会话必须选择 `Qwen · VLM`，而视觉证据中
的 `QwenVlmProvider / qwen3-vl-plus / sam2` 才是实际模型权威标识。Oracle 不再作为
当前视觉验收结果。

## 已验证结果

- 第一视角办公室 JPEG：640×480，深度：Z16，定位/TF 与观测会话持续可用。
- 真实 Qwen 只观察任务已通过：MuJoCo 第一视角中的唯一橙色球由
  `QwenVlmProvider / qwen3-vl-plus` 产生单一像素框，SAM 2 掩码与三帧深度/TF
  形成一个 verified `map` landmark；Mission 只有 `vision.observe` 且完成，未发送
  任何底盘动作。
- `g1-sim` V2 能力清单只发布导航和旋转，executor 为 Nav2。
- 人工审批后执行 `向前移动0.3米`：Mission 从 `approved → executing → completed`，
  Robot Agent 回执含真实 Nav2 goal ID；MuJoCo 根位姿和网页轨迹均发生变化。
- 即使审批等待超过位姿 2 秒 TTL，执行前也会重新发现 Robot Agent 位姿；
  审批时的 `mission_origin` 只用于审计和累计距离，不能冒充实时定位。
- 速度上限 0.25 m/s；提高至 0.35 m/s 的对照更不稳定，因此没有采用。
- 手臂目标经过导航专用包络和速率限制；Nav2 高频微调经过 SONIC 重规划门控。
- `/cmd_vel` 超时、取消或桥断开会立即锁存 MuJoCo 实测关节姿态；SONIC 的
  learned IDLE 只在内部保持推理历史，不再发送给关节执行器。
- Nav2 成功后，Robot Agent Gateway 不再把成功前或第一帧惯性位姿写成终点。
  只有根位姿线速度低于 0.02 m/s、角速度低于 0.05 rad/s 且连续稳定 0.5 秒，
  才以 `stable_after_terminal` 写入 H-IDE 事件账本；最长等待 4 秒，工作站在
  5 秒处还有显式降级上限。
- 0.2 米短程实测中，Nav2 终止后速度从约 0.13 m/s 衰减到稳定阈值，H-IDE
  在此期间保持“等待机身静止”；锁存终点距最终静止点约 1.3 cm，最终目标
  平面误差约 5 cm。此前 learned-IDLE 续走约 0.19 m 的问题已消除。
- 同一任务的摆臂峰值观测为：实测偏离约 0.72 rad、最大关节速度约
  1.85 rad/s、包络介入 2 个关节。画面呈步态反向摆臂而非持续失控，但这仍是
  仿真诊断数据，不代表真机手臂安全验收。

## 尚未解除的限制

- **Qwen视觉链已通过，但SONIC运动链仍冻结。** 2026-08-19 的受控0.2米用例中，
  `vision.observe` 成功识别唯一橙色球并形成verified几何候选，Robot Agent/Nav2
  随后开始相对导航，MuJoCo实际移动约0.083米；之后根速度异常达到约3.38 m/s，
  物理引擎报告时间回滚/数值不稳定。联调守卫立即取消Mission并将仿真恢复初始位姿。
  这不能记为导航成功；修复前网页验收只运行不含导航/旋转的视觉提示词。

- 当前 Nav2 `xy_goal_tolerance=0.15 m` 是稳定性优先的联调值。很短的 0.3 米
  指令会提前进入容差区；在正式精度验收前需要分距离标定，而不是盲目提高速度。
- MuJoCo 的垂直/横滚/俯仰 fall-arrest 是明确标记的仿真安全支撑；它不提供
  水平运动，不能代表真机平衡验收。
- 办公室场景与障碍目前是静态的，视觉避障和动态人流不在本轮范围内。
- FoundationPose 暂未进入此仿真主链；后续作为注册物体位姿 specialist 接入。

## 确定性后端删除门

确定性后端/FakeAdapter 目前只作为迁移与故障定位 fixture 保留。只有在以下验收
形成书面记录后才删除：真实 Qwen 视觉链、取消/超时恢复、十次连续重规划逐次
人工批准、静态绕障、重启恢复、长时间运行，以及网页证据与 ROS/Robot Agent
回执一致。删除时同时移除代码、配置、测试、文档和历史入口，不能留下第二条
执行路径。

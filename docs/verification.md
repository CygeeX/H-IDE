# 验证矩阵

| 层级 | 内容 | 命令 |
|---|---|---|
| 纯核心 | 协议、配置、账本、仲裁、租约、队列 | `.venv/bin/python -m pytest tests -m 'not integration and not soak'` |
| 传输 | 双模式 mTLS、CA/SAN/叶证书固定、帧、重连 | `.venv/bin/python -m pytest tests/test_transport.py` |
| ROS Jazzy | Service、Action、反馈/取消、Topic、心跳 | `.venv/bin/python -m pytest tests/test_ros_integration.py` |
| Humble | Python 3.10 与 ROS Humble 容器 | `tools/test_humble.sh` |
| 静态 | Ruff、mypy、ament/colcon | 见 README |
| soak | 30 分钟真实 Runtime、租约/重连/状态/有界队列 | 见下方命令 |
| RGB-D 核心 | 编码、CRC、严格元数据、容量 1 队列 | `.venv/bin/python -m pytest tests/test_observation_core.py` |
| gRPC/mTLS | 双向流、ACK、机器人证书 SAN、错误 SAN | `.venv/bin/python -m pytest tests/test_observation_grpc_integration.py tests/test_observation_mtls.py` |
| 观测 ROS | 强类型 demand/health QoS 与桥接 | `.venv/bin/python -m pytest tests/test_observation_ros_bridge.py` |
| 可视化 | 接收端 RGB/深度二进制端点与故障按钮 | `.venv/bin/python -m pytest tests/test_web_demo.py` |
| H-IDE 合约 | 上游 proto/lock、导航 Schema、真实 H-IDE Vision ingress | `.venv/bin/python -m pytest tests/test_h_ide_adaptation.py` |
| Nav2 | H-IDE map 目标、真实 NavigateToPose 类型 Mock、反馈/终态 | `.venv/bin/python -m pytest tests/test_nav2_integration.py` |

默认 90% 分支覆盖率门禁继续约束安全关键的 `robot_agent`（包含本地观测守卫）；
`robot_observation_gateway` 的 ROS/相机入口依赖真实时间戳、TF 和硬件，因此以核心单元、
gRPC/mTLS 集成、ROS bridge 和 D435i 真机验收四层覆盖，不用 mock 行数替代硬件证据。

验收记录应保存测试时间、ROS 发行版、Git commit、失败注入、峰值 RSS、队列最大深度、事件循环最大延迟和重复执行次数。任何 safe-stop 失败、命令重复下发、队列无界增长或退出阶段 ROS 销毁告警都视为失败。

## 2026-08-07 本地验收记录

验收快照：`d27b33e`，核心加固提交：`1d796a3`，分支：
`feature/robot-agent-v1`。本轮没有复制或启用真机服务。

| 门禁 | 结果 |
|---|---|
| Jazzy / Python 3.12 非 soak 全量 | 122 passed；1 soak deselected |
| 分支覆盖率 | 90.09%，门禁 90% |
| Ruff / strict mypy | 通过 |
| colcon build/test | 1 package；1 test，0 errors，0 failures |
| Humble / Python 3.10 容器 | 122 passed；1 soak skipped |
| 最终代码审查 | Blocker 0、Important 0、Minor 0，Ready |
| systemd / 配置 / doctor / Schema | 全部通过；systemd 仅报告主机无关的 AnyDesk legacy PIDFile 提示 |

正式命令：

```bash
env -u CYCLONEDDS_URI \
  ROBOT_AGENT_RUN_SOAK=1 ROBOT_AGENT_SOAK_SECONDS=1800 \
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  PYTHONPATH=src/robot_agent:${PYTHONPATH} \
  .venv/bin/python -m pytest -q -s tests/test_soak.py
```

该测试运行完整 `RobotAgentRuntime`，使用真实 WebSocket 协议/会话/续租/重连、
ROS 2 Service/Topic/心跳/safe-stop、SQLite 台账与运行期 retention。第二次正式运行
使用 10 ms tick，共 180,000 次迭代并通过：

| 指标 | 结果 | 门禁 |
|---|---:|---:|
| 实际时长 | 1800.281 s | ≥ 1800 s |
| 会话 / 请求 / 终态 | 6 / 8,749 / 8,749 | 所有请求有终态 |
| 成功 / ROS 调用 | 8,674 / 8,674 | 严格相等且成功吞吐 ≥ 3 Hz |
| 合法资源冲突拒绝 | 75 | owner 属于本轮且最终成功 |
| 起始 / 峰值 RSS | 94,519,296 / 105,746,432 B | 增长 < 64 MiB |
| RSS 增长 | 11,227,136 B | < 64 MiB |
| 队列峰值 | 2 / 64 | ≤ 64 |
| ROS 请求队列峰值 | 1 / 256 | ≤ 256 |
| 命令任务峰值 | 7 / 64 | ≤ 64 |
| 丢弃 | 0 | 0 |
| 最大事件循环延迟 | 20.510 ms | < 500 ms |
| 最大工作循环延迟 | 201.592 ms | 记录项 |
| ROS safe-stop / 心跳 | 7 / 7,221 | 发生且数量符合会话生命周期 |

测试以 5 Hz 开环向单一互斥资源提交命令。ROS Service 偶发超过 200 ms 时，后续
命令按设计得到 `REJECTED`；验收只接受 detail 可解析为本轮最终成功 owner 的资源冲突，
并逐 ID 校验线上状态与 SQLite 状态一致，因此不会把陈旧资源占用、重复下发或任意拒绝
当作成功。

幂等专项重复提交同一 `command_id` 和相同规范载荷时返回既有记录，没有再次执行；
同 ID 不同载荷被 `CommandIdConflict` 拒绝。重启恢复测试把所有遗留非终态记录置为
`ABORTED_AGENT_RESTART`，不会自动重放。

## 2026-08-12 可视化测试台增量验证

- Jazzy/Python 3.12：127 passed、1 soak deselected，分支覆盖率 90.09%。
- Humble/Python 3.10 固定镜像：127 passed、1 soak skipped。
- 可视化服务专项：5 passed；覆盖 HTTP 安全头、Agent 握手、控制/续租/命令桥接、
  Agent 离线拒绝，以及同源允许/异源拒绝。
- JavaScript 语法、全仓 Ruff、产品包和可视化服务 strict mypy、`git diff --check`
  全部通过。
- 使用隔离 `ROS_DOMAIN_ID=232` 完成真实链路：`SAFE_HOLD → ACTIVE`，
  `power.enable → SUCCEEDED → robot.mode=IDLE`；Action 进入 `RUNNING` 后取消为
  `CANCELLED`；暂停续租后回到 `SAFE_HOLD`，再次显式获取控制才恢复 `ACTIVE`。
- 一键启动栈退出时 Robot Agent 与 Mock Robot 均正常结束，无 ROS entity 销毁 traceback。

## 2026-08-12 D435i 观测网关增量验证

- Jazzy / Python 3.12 非 soak 全量：185 passed、1 soak deselected；控制面分支覆盖率
  90.18%，门禁 90%；Humble / Python 3.10 固定镜像：185 passed、1 soak skipped。
- ROS 2 Jazzy `colcon build/test`：3 packages；2 package smoke tests，0 errors、0 failures；
  全仓 Ruff、36 个产品 Python 文件 strict mypy、JavaScript 语法和 `git diff --check` 通过。
- 设备：Intel RealSense D435i，序列号 `261222077185`，固件 `5.15.1.55`，USB 3.2
  连接速率 5,000 Mbit/s。
- 官方 RealSense ROS 启用同步 RGB-D、同步器和深度对齐，实测约 28 Hz 原始同步输入。
- 使用严格本地 TF/定位 fixture 完成新一轮 12 帧端到端验收：12 accepted、0 rejected、
  最终 sequence 11、capture age 69.37 ms、JPEG magic `ffd8`、深度 PNG magic
  `89504e47`。
- 稳态遥测约 5.30 fps、4.74 Mbps、ACK RTT 约 12.66 ms、估算网络延迟约 0.87 ms、
  gRPC write blocked 约 0.212 ms，低于 20 Mbps 和 750 ms 门禁。`dropCount=44` 来自
  明确设计的 capacity-1 newest-only 丢旧帧（相机约 30 Hz、上报目标 5 Hz），不是积压。
- 修正网页真机启动拓扑后再次验收：相机 launch 使用 `start_gateway:=false`，网页进程
  是唯一 Gateway；网页二进制端点返回 640×480 JPEG 和深度 PNG，抽样时累计 324
  accepted、0 rejected，状态 `USABLE`，约 5.50 fps / 4.60 Mbps、估算网络 0.79 ms。
  彩色与深度图均人工检查为相机实拍内容，三个进程随后通过 SIGINT 正常退出。
- 合成数据端到端、mTLS 正向/错误机器人 SAN、强类型 ROS demand/health、Web RGB/深度
  二进制返回和故障动作均已有自动化覆盖。

## 2026-08-12 H-IDE 联调增量验证

- Observation proto、lock 和生成的 `observation_pb2.py` 与 H-IDE Vision 逐字节核对；
  gRPC stub 仅按 Python 包布局使用绝对包导入；Hello 使用 `major=1`、`minor=0`、descriptor SHA
  `d5055d5dbd81f2e098209bf9cba9db5739b74a3be62dae172958bd536648bb14`。
- H-IDE 导航 Schema 与本地逐字节相同；生产/本地配置均使用
  `nav2_msgs/action/NavigateToPose`，仅 `goalPose` 进入 ROS Goal。
- 自动化 Jazzy/Python 3.12 全量：213 passed、1 个 30 分钟 soak deselected，分支覆盖率
  90.54%；Humble/Python 3.10 真实 `nav2_msgs` 容器：211 passed、3 skipped（容器未挂载
  两个 H-IDE Vision 只读联调测试，另一个是默认关闭的 soak）。H-IDE Vision 实际
  ingress、真实 NavigateToPose 类型 Mock、boot/session/epoch/超时和 scene 可选均已覆盖。
- 隔离 `ROS_DOMAIN_ID=231` 真机验收：D435i LIVE RGB-D 被 H-IDE Vision 接收并生成
  `map` 目标；H-IDE Gateway 观察到 `RECEIVED → RUNNING/feedback → SUCCEEDED`。
  同 ID 重发返回同一 `rosGoalId`，Mock Nav2 只接收一次，ROS graph 无 `/cmd_vel`。
- 验收未连接真实 Nav2、未发布底盘速度；退出时 Robot Agent 再次调用 safe-stop，
  RealSense 节点干净关闭。完整证据和边界见
  [H-IDE 联调适配](h-ide-integration.md)。

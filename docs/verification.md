# 验证矩阵

| 层级 | 内容 | 命令 |
|---|---|---|
| 纯核心 | 协议、配置、账本、仲裁、租约、队列 | `.venv/bin/python -m pytest tests -m 'not integration and not soak'` |
| 传输 | 双模式 mTLS、CA/SAN/叶证书固定、帧、重连 | `.venv/bin/python -m pytest tests/test_transport.py` |
| ROS Jazzy | Service、Action、反馈/取消、Topic、心跳 | `.venv/bin/python -m pytest tests/test_ros_integration.py` |
| Humble | Python 3.10 与 ROS Humble 容器 | `tools/test_humble.sh` |
| 静态 | Ruff、mypy、ament/colcon | 见 README |
| soak | 30 分钟真实 Runtime、租约/重连/状态/有界队列 | 见下方命令 |

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

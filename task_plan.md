# Robot Agent v1 任务计划

## 目标

在本地实现并验证一个安全、受限、可审计的 Robot Agent：通过版本化 WebSocket 协议连接工作站，将逻辑高层命令映射到本地 ROS 2 Service/Action，聚合 Topic 状态，并在断线或租约超时后进入安全态。

## 当前阶段

- [x] 架构资料与 rosbridge 参考实现调研
- [x] 用户确认总体设计、协议边界与验收标准
- [x] 仓库与测试基础设施
- [x] 协议与严格配置模型
- [x] 状态归一化与输出模型
- [x] 命令台账、资源仲裁与租约状态机
- [x] client/server WebSocket 与 mTLS
- [x] ROS 2 Service/Action/Topic 适配与安全心跳
- [x] CLI、launch、systemd、Mock 工具和文档
- [x] Jazzy/Humble、故障、覆盖率与 soak 验证

## 已锁定决策

- ROS 2 Humble 为部署基线，本机 Jazzy 同时验证。
- Python 3.10/3.12 + rclpy；单进程，asyncio 与 ROS executor 分线程。
- WebSocket client/server 都支持，单次运行由配置二选一。
- 生产强制 mTLS；不暴露 rosbridge 通用操作，不允许远端指定 ROS 名称或类型。
- 命令使用本地注册表、参数 JSON Schema、资源集合、TTL、SQLite 幂等台账。
- 500 ms 续租、2 s 租约；断线锁存 SAFE_HOLD，恢复需显式 control.acquire。
- 本轮不接真实 SONIC/Unitree，不复制真机，不实现正式 BFF。

## 遇到的错误

| 错误 | 次数 | 处理 |
|---|---:|---|
| 初始工作区不是 Git 仓库，无法创建 worktree | 1 | 在当前空工作区初始化 `feature/robot-agent-v1` 分支 |
| 系统 Python 缺少 Pydantic/websockets/Ruff/mypy | 1 | 建立项目虚拟环境并锁定依赖，同时继承系统 rclpy |
| `uv venv` 首次自动选择 Python 3.11，无法继承 Jazzy 的 Python 3.12 rclpy | 1 | 用 `/usr/bin/python3.12 --system-site-packages --clear` 重建，并验证 rclpy/Pydantic/websockets 同时可导入 |
| 静态检查发现 `datetime.UTC` 不兼容 Humble 的 Python 3.10 | 1 | 全部改用 `timezone.utc`，并把 mypy 的 Python 版本固定为 3.10 |
| ROS 集成测试退出时偶发 `Destroyable`/ActionServer 销毁竞态 | 1 | 复现 10 轮并定位为 executor 尚在 `spin()` 时销毁 guard/entity；改成可唤醒 `spin_once` 循环，先停转并 join，再销毁实体，连续复验无告警 |
| Mock 工具命令覆盖了 ROS 环境的 `PYTHONPATH`，导致直接运行时找不到 `rclpy` | 1 | 文档与验证命令改为在项目路径后保留原有 `${PYTHONPATH}` |
| colcon 首次运行只发现包但未收集测试 | 1 | 增加包边界 smoke test 与 `tests_require`，并用项目 venv 驱动 `/usr/bin/colcon`；结果 1 test/0 failures |
| 开发机 `CYCLONEDDS_URI` 仅配置远端 peer，本地三个 ROS 进程互相不可见 | 1 | 本地联调取消该变量并设 `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`，写入 README |
| mandatory safe-stop 初始化失败发生在 runtime 清理范围之外，ROS executor 线程泄漏 | 1 | 将初始化纳入 `run()` 的 `try/finally`，增加初始化失败清理回归测试 |
| Docker daemon 未继承主机代理，拉取 Humble 镜像持续 EOF | 2 | 不重启正在承载其他容器的 daemon；用官方 registry 客户端经用户代理导出并 `docker load` |
| Humble 容器用 `pytest` 脚本启动时工作区根目录未进入 `sys.path` | 1 | 容器统一改用 `python3 -m pytest` |
| Python 3.10 的 `asyncio.TimeoutError` 尚未与内建 `TimeoutError` 合并 | 1 | 所有 `asyncio.wait_for` 路径显式捕获 `asyncio.TimeoutError`；修复命令超时、watchdog 与重连退避 |
| 最终设计审计发现保留期未生效、示例 Action 类型仅为测试依赖 | 1 | 只清理过期终态台账并保留非终态；将 `example_interfaces` 提升为运行依赖并加元数据测试 |
| 首次真实 1800 秒 soak 把合法互斥资源冲突视为失败 | 1 | 保留现场确认所有拒绝均引用最终成功 owner；验收改为逐 ID 校验线上/账本终态、owner 和 ROS 调用数，再完整重跑 1800 秒通过 |

## 验收摘要

- 验收快照：`d27b33e`，核心加固提交：`1d796a3`（`feature/robot-agent-v1`）。
- Jazzy/Python 3.12：122 项非 soak 测试通过，分支覆盖率 90.09%，Ruff 与 strict mypy 通过。
- Humble/Python 3.10：固定官方 Humble 镜像内 122 项通过，1 个长 soak 跳过。
- colcon：1 个包级 smoke test，0 errors/0 failures；最终审查 Blocker/Important/Minor 均为 0。
- 30 分钟真实 Runtime soak：180,000 次循环，8,749/8,749 条命令有终态，8,674 次成功且无重复 ROS 调用；RSS 增长 11,227,136 B、队列峰值 2/64、丢弃 0、事件循环最大延迟 20.510 ms。

# Robot Agent v1 进度

## 2026-08-07

- 完成两份架构资料与 rosbridge_suite 协议、capability、类型转换、WebSocket 队列和测试结构调研。
- 用户确认 Agent 内核闭环、双模式配置二选一、mTLS、受限注册表、Humble 主目标、资源分组、2 秒租约、显式重新授权和完整核心验收。
- 工作区无 Git 基线，记录 worktree 限制并初始化 `feature/robot-agent-v1` 分支。
- 发现系统 Python 缺少 Pydantic、websockets、Ruff 和 mypy；将使用带 system-site-packages 的项目虚拟环境，保留本机 rclpy。
- 写入 ament_python 元数据、依赖约束和首个协议测试；当前准备验证 RED。
- 首次 RED 因 `robot_agent.protocol` 不存在而在收集阶段失败，符合功能尚未创建但需进一步收敛为断言失败。
- `uv` 首次误选 3.11；已改为项目 `.venv` 使用系统 Python 3.12 并继承 Jazzy site-packages，依赖锁定在 `uv.lock`。
- 完成 `robot-agent.v1` 严格协议：14 种判别消息、64 KiB 文本帧限制、目标与时间窗校验、命令过期校验及 JSON Schema 导出。
- 完成严格 YAML 配置与本地命令注册表：重复键拒绝、mTLS 文件检查、明文仅显式 loopback、ROS 端点类型约束、命令参数 JSON Schema 与能力摘要。
- 协议/配置聚焦测试 26 项通过；Ruff 与严格 mypy 通过。静态检查发现并修复 `datetime.UTC` 的 Python 3.10 不兼容用法。
- 完成 SQLite WAL/FULL 命令台账、显式迁移版本、状态转移图、同 ID 幂等、载荷冲突拒绝和重启中止恢复。
- 完成资源组并发仲裁、租约/断线 `SAFE_HOLD` 状态机、命令协调器、超时/取消/安全预占，以及 5 Hz 变化/1 Hz 快照状态合并和有界优先队列。
- 安全核心阶段全量 43 项测试通过；Ruff 与 Python 3.10 严格 mypy 继续通过。

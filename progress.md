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
- 完成共用文本连接契约、WebSocket client 指数退避重连和 server 单控制会话策略；两种模式均强制 `robot-agent.v1` 子协议、64 KiB、关闭码和有界接收队列。
- 使用临时 CA/服务端/客户端证书完成真实 mTLS 集成验证，包括 Agent 首发应用握手、无客户端证书、错误 CA、错误 SAN、二进制、超大帧与重连后保持 `SAFE_HOLD`。
- 传输阶段全量 48 项测试通过；Ruff 与严格 mypy 继续通过。
- 完成 ROS 2 动态类型限制适配：异步 Service/Action、反馈、取消、显式 QoS/CallbackGroup、Topic 字段归一化与专用 executor 线程；未使用同步 ROS future 等待。
- 完成 250 ms 安全心跳，QoS 为 RELIABLE/VOLATILE/KEEP_LAST(1)、Deadline 500 ms、Lifespan 1 s；Mock ROS 图验证 Service、Fibonacci Action、状态 Topic、取消和安全停机。
- 系统化定位并修复 ROS executor/entity 退出竞态；10 轮复现和 5 轮组合回归均无销毁告警。
- 完成应用 runtime：Agent 首发 hello、welcome 策略锁定、显式 acquire、续租、命令路由、反馈/状态优先队列、租约过期安全停机、重启台账恢复和优雅 transport 停止。
- ROS/runtime 阶段全量 57 项测试通过；Ruff 与 Python 3.10 严格 mypy 通过。
- 完成 `run`、`validate-config`、只读 `doctor` CLI、JSON journald 日志、原生 systemd notify/watchdog、launch 和加固 Unit；`systemd-analyze verify` 返回 0。
- 补齐本地/生产配置、协议与配置 JSON Schema、Mock Robot、双模式 Mock BFF、Humble Docker 入口、部署/协议/验证文档。
- 扩充错误与生命周期分支测试；当前 88 项通过，核心分支覆盖率 92.45%，高于 90% 门禁。
- 修复 colcon 测试发现边界：新增已安装包 smoke test 与 `tests_require`；Jazzy colcon build/test 最终为 1 test、0 errors、0 failures。
- 完成本地三进程真实联调：Mock BFF 收到 `SAFE_HOLD → ACTIVE`，`power.enable` 经 ROS 2 Service 返回 `SUCCEEDED`；BFF 断开后 Mock Robot 再次收到 safe-stop，三个进程均正常退出。
- 修复 mandatory safe-stop 初始化失败时 runtime 未关闭 ROS/ledger 的生命周期漏洞，并让 Mock Robot 在 ROS SIGINT 后使用 `try_shutdown()` 干净退出。
- 在 Humble/Python 3.10 容器中定位并修复 `asyncio.TimeoutError` 兼容问题；最终固定官方 Humble 镜像内 122 项通过、1 个长 soak 按标记跳过。
- 完成最终设计审计：启用终态台账保留期清理、补齐随附 ROS 接口运行依赖、增加显式 Humble systemd 入口和可审计 Mock BFF 终态退出。
- 完成 WebSocket server 客户端叶证书 SHA-256 固定、运行期 retention、ROS owner 线程和有界请求队列，以及命令注册/取消/抢占与安全停机取消竞态加固。
- 最终 Jazzy 非 soak 全量为 122 项通过；Ruff、strict mypy 通过；分支覆盖率 90.09%；colcon 为 1 test/0 failures。
- 完成第二次 1800 秒真实 Runtime soak：180,000 次循环，8,749 条命令全部有终态，8,674 次 ROS 成功调用；RSS 94,519,296 → 105,746,432 B、队列峰值 2/64、丢弃 0、最大事件循环延迟 20.510 ms。
- 核心加固提交为 `1d796a3`，验收快照为 `d27b33e`；本轮仍未复制到真机，等待真实 SONIC/Unitree 接口清单和目标机变更窗口。

## 2026-08-12

- 新增本地可视化测试台：同一回环服务提供静态网页、真实 Robot Agent `/agent`
  接入端和浏览器 `/ui` 状态桥接。
- 页面可视化架构链路、控制权、租约倒计时、归一化 ROS 状态、命令生命周期和安全
  事件；支持动力使能、Action 执行/取消、暂停续租与重新获取控制。
- 增加明确标识的纯浏览器演示模式；它只模拟显示效果，不作为 ROS/Agent 验收证据。
- 一键入口在独立 `ROS_DOMAIN_ID=232` 启动 Mock Robot 与真实 Agent，子进程使用独立
  会话并由启动器统一回收，Ctrl-C 可干净退出。
- 增加回环监听、Origin 校验、CSP、64 KiB 帧限制、单 Agent、最多 8 个 UI 客户端、
  有界事件/命令历史和慢 UI 发送超时。
- 最终 Jazzy 与 Humble 均为 127 项通过；真实 UI→BFF→Agent→ROS 2 链路验证成功。

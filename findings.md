# Robot Agent v1 实施发现

## 本地环境

- 工作区初始为空，且不存在 `.codegraph/`。
- Ubuntu 24.04、x86_64、Python 3.12.3、ROS 2 Jazzy、colcon 和 Docker 可用。
- Docker daemon 没有继承主机代理，直接拉取 Docker Hub 会 EOF；为避免重启其承载的其他容器，本轮通过用户态官方 registry 客户端导入 Humble 镜像。镜像现已在本地缓存。
- 开发机默认 CycloneDDS 配置只声明远端 peer；本地多进程联调必须取消该配置并使用 `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`。

## 架构输入

- 浏览器只连接工作站；Robot Agent 是工作站与机器人内部 ROS/SONIC 接口之间的边界。
- 命令必须包含 ID、签发时间、有效期和目标机器人，并在工作站和 Agent 双重校验。
- 断线或租约过期必须在机器人本地停止；恢复后不得自动重放旧命令。
- 状态应归一化后上送，不能把 TF、PointCloud2、Marker 或完整 ROS 图跨 5G 暴露。

## rosbridge 可借鉴与排除

- 借鉴：协议信封、operation/capability 分层、严格 ROS 字段转换、QoS 校验、每连接有界写队列和 WebSocket/ROS 集成测试。
- 排除：远端 advertise/publish/subscribe/call_service/send_action_goal 通用能力、远端 ROS 类型选择和默认接受任意 Origin。
- Robot Agent 使用逻辑命令注册表；ROS 名称、类型、QoS 和字段映射只存在于机器人本地配置。

## 工程约束

- ROS Service/Action 调用全部异步；callback group 分离，禁止在回调中同步等待。
- 安全心跳采用 RELIABLE/VOLATILE/KEEP_LAST(1)，Deadline 500 ms、Lifespan 1 s。
- 普通状态可合并为最新值；命令结果和安全事件不得静默丢弃。
- Python 3.10 中应捕获 `asyncio.TimeoutError`，不能依赖 Python 3.11+ 将其别名到内建 `TimeoutError` 的行为。
- systemd 不会加载交互式 ROS 环境；目标入口必须显式 source Humble setup 后再 exec venv 内 Agent。
- `storage.retention_days` 同时定义 SQLite 增长上限和命令幂等窗口；仅允许清理超过窗口的终态命令。
- 5 Hz 开环提交到单一互斥资源时，ROS Service 偶发超过 200 ms 会按设计产生资源冲突拒绝；长稳验收必须验证冲突 owner 属于本轮且最终成功，并要求 ROS 调用数严格等于成功数，不能简单要求全部请求成功。
- asyncio 无法强杀永久吞掉 `CancelledError` 的协程；安全停机超时任务必须显式追踪、消费最终结果并阻止后续安全操作，避免静默恢复控制或无界累计。

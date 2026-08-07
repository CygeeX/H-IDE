# Robot Agent v1 实施发现

## 本地环境

- 工作区初始为空，且不存在 `.codegraph/`。
- Ubuntu 24.04、x86_64、Python 3.12.3、ROS 2 Jazzy、colcon 和 Docker 可用。
- 本地没有缓存 ROS 2 Humble 镜像，Humble 验证阶段需要拉取官方镜像。

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


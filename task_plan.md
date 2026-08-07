# Robot Agent v1 任务计划

## 目标

在本地实现并验证一个安全、受限、可审计的 Robot Agent：通过版本化 WebSocket 协议连接工作站，将逻辑高层命令映射到本地 ROS 2 Service/Action，聚合 Topic 状态，并在断线或租约超时后进入安全态。

## 当前阶段

- [x] 架构资料与 rosbridge 参考实现调研
- [x] 用户确认总体设计、协议边界与验收标准
- [x] 仓库与测试基础设施
- [x] 协议与严格配置模型
- [ ] 状态归一化与输出模型
- [ ] 命令台账、资源仲裁与租约状态机
- [ ] client/server WebSocket 与 mTLS
- [ ] ROS 2 Service/Action/Topic 适配与安全心跳
- [ ] CLI、launch、systemd、Mock 工具和文档
- [ ] Jazzy/Humble、故障、覆盖率与 soak 验证

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

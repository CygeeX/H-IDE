# Robot Agent v1

Robot Agent 是部署在机器人计算单元上的受限协议代理。它只把本地配置中声明的逻辑命令翻译为 ROS 2 Service/Action，并把指定 Topic 归一化为状态；它不是通用 rosbridge，也不会接受远端提供的 ROS 名称或类型。

当前实现完成的是 Agent 内核和本地验证环境，不包含真实 SONIC/Unitree 适配，也不会自动复制到真机。

## 安全边界

```text
BFF / 工作站
   │ robot-agent.v1 + mTLS + 2 s lease
   ▼
WebSocket client/server（运行时二选一）
   │ 严格消息模型、64 KiB、单控制会话
   ▼
会话状态机 ── SQLite 幂等台账 ── 资源组仲裁
   │ SAFE_HOLD / safe-stop              │
   └────────────────────────────────────▼
              受限 ROS 2 Service / Action / Topic
```

- 启动、断线或租约过期都会调用必配的本地 safe-stop，并锁存 `SAFE_HOLD`。
- 重连不会恢复控制；BFF 必须重新发送 `control.acquire`。
- 同一 `command_id` 和同一载荷返回已有状态；同 ID 不同载荷会被拒绝。
- 普通命令只能占用配置中的资源组；safe-stop 绕过锁并取消所有活动 Action。
- 生产配置必须使用 mTLS。server 模式还会固定客户端叶证书的 SHA-256
  指纹，不能只依赖同一 CA；明文只允许显式开启的 loopback 开发模式。

## 本地快速验证（ROS 2 Jazzy）

```bash
source /opt/ros/jazzy/setup.bash
uv venv --python /usr/bin/python3.12 --system-site-packages .venv
uv sync --python .venv/bin/python --all-groups
.venv/bin/python /usr/bin/colcon build --symlink-install
source install/setup.bash
```

若开发机预设了仅指向远端机器人的 `CYCLONEDDS_URI`，本地三个进程会互相发现不到。联调终端应先执行：

```bash
unset CYCLONEDDS_URI
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

依次打开三个终端：

```bash
# 1. Mock ROS graph
PYTHONPATH=src/robot_agent:${PYTHONPATH} .venv/bin/python tools/mock_robot.py

# 2. Mock BFF（监听本地 Agent client）
PYTHONPATH=src/robot_agent:${PYTHONPATH} .venv/bin/python tools/mock_bff.py --mode server

# 3. Agent
PYTHONPATH=src/robot_agent:${PYTHONPATH} .venv/bin/python -m robot_agent.cli run \
  --config src/robot_agent/config/robot_agent.local.yaml
```

Mock BFF 可发送一次命令：

```bash
PYTHONPATH=src/robot_agent:${PYTHONPATH} .venv/bin/python tools/mock_bff.py \
  --mode server --command-kind power.enable --params '{"data":true}' \
  --exit-on-command-terminal
```

配置检查与只读诊断：

```bash
PYTHONPATH=src/robot_agent:${PYTHONPATH} .venv/bin/python -m robot_agent.cli validate-config \
  --config src/robot_agent/config/robot_agent.local.yaml
PYTHONPATH=src/robot_agent:${PYTHONPATH} .venv/bin/python -m robot_agent.cli doctor \
  --config src/robot_agent/config/robot_agent.local.yaml
```

## 验证

```bash
PYTHONPATH=src/robot_agent:${PYTHONPATH} .venv/bin/python -m pytest -m 'not soak'
.venv/bin/ruff check .
.venv/bin/mypy src/robot_agent/robot_agent
.venv/bin/python /usr/bin/colcon build --symlink-install
.venv/bin/python /usr/bin/colcon test --event-handlers console_direct+
.venv/bin/python /usr/bin/colcon test-result --verbose
```

Humble 容器验证入口为 `tools/test_humble.sh`。30 分钟 soak 使用真实
WebSocket/ROS 2/SQLite 链路并带有 `soak` marker，详见[验证说明](docs/verification.md)。

## 配置与协议

- 本地可运行配置：[robot_agent.local.yaml](src/robot_agent/config/robot_agent.local.yaml)
- 生产 mTLS 模板：[robot_agent.production.example.yaml](src/robot_agent/config/robot_agent.production.example.yaml)
- 协议说明：[protocol-v1.md](docs/protocol-v1.md)
- 运维与部署：[operations.md](docs/operations.md)
- 生成的 JSON Schema：[schemas](src/robot_agent/schemas)

配置中的 `commands` 是唯一 ROS 出口。修改注册表后，能力摘要会变化，BFF 可据此识别部署能力。生产上线前必须把模板里的 ROS 名称、类型、参数 Schema、资源集合和 safe-stop 端点替换为真机经过评审的接口。

若生产使用 WebSocket server 模式，必须在 `tls.peer_certificate_sha256` 中填写
BFF 客户端叶证书 DER 的小写 SHA-256 指纹；证书轮换时应同步更新配置并重新验收。

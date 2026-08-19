# Robot Agent v1

Robot Agent 是部署在机器人计算单元上的受限协议代理。它只把本地配置中声明的逻辑命令翻译为 ROS 2 Service/Action，并把指定 Topic 归一化为状态；它不是通用 rosbridge，也不会接受远端提供的 ROS 名称或类型。

当前实现完成的是 Agent 内核和本地验证环境，不包含真实 SONIC/Unitree 适配，也不会自动复制到真机。

## 可视化体验

默认命令会启动测试网页和一条真实的合成 RGB-D 二进制链路；加上
`--launch-stack` 还会启动隔离 ROS Domain 中的 Mock Robot 与真实 Robot Agent：

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src/robot_agent:src/robot_observation_gateway:${PYTHONPATH} .venv/bin/python \
  tools/web_demo_server.py --launch-stack --open-browser
```

也可手动打开 <http://127.0.0.1:8765/>。页面可以获取控制权、调用
`power.enable`、执行/取消 `task.execute`，并暂停续租验证 Agent 在本地进入
`SAFE_HOLD`。如果暂时不启动 ROS/Agent，直接运行不带 `--launch-stack` 的命令，
网页同时显示 RGB、对齐深度、采集/编码/排队/ACK 与估算网络延迟、帧序号和丢帧，并可执行
绑定当前观测会话的导航测试命令。若要直接看已经接入的 D435i，先启动官方
RealSense ROS 节点和定位/TF，并确保相机 launch 使用 `start_gateway:=false`，再把参数
改为 `--observation-source realsense`；网页进程会启动唯一的本地 Gateway。
不启动 ROS/Agent 时仍可点击页面右上角“进入浏览器演示”查看控制面效果。详细说明见
[可视化测试台](docs/web-demo.md)。

## RGB-D 观测网关

`robot_observation_gateway` 是与主 Robot Agent 分进程部署的 sidecar。它订阅官方
RealSense ROS 输出的同步、对齐 RGB-D，在每个采集时间点查询 `map → camera` 和
`base_link → camera` TF，匹配不晚于采集时刻的定位协方差，然后用 JPEG 85 与无损
Zstd Z16 编码，通过机器人主动建立的 mTLS gRPC 双向流发往工作站。

数据面在各阶段使用容量 1 的 newest-only 槽位并且只允许一帧在途：网络变慢时丢弃旧
帧，不积压重放。只有工作站的 `ACK_ACCEPTED` 会刷新本地观测健康。Robot Agent 对
`requires_observation` 命令执行 500 ms 准入检查和 750 ms 看门狗；观测断流只取消这类
导航目标，原有控制 WebSocket、2 秒租约、safe-stop 与状态优先级不变。

协议和运行配置位于：

- [observation.proto](proto/robot_observation/v1/observation.proto)
- [本地 D435i 配置](src/robot_observation_gateway/config/robot_observation_gateway.local.yaml)
- [生产 mTLS 配置模板](src/robot_observation_gateway/config/robot_observation_gateway.production.example.yaml)
- [观测网关说明](docs/observation-gateway.md)
- [H-IDE 联调适配](docs/h-ide-integration.md)

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
/usr/bin/python3 /usr/bin/colcon build --symlink-install
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
PYTHONPATH=src/robot_agent:src/robot_observation_gateway:${PYTHONPATH} \
  .venv/bin/python -m pytest -m 'not soak'
.venv/bin/python -m pytest -m 'not soak' --cov=robot_agent --cov-fail-under=90
.venv/bin/ruff check .
.venv/bin/mypy
/usr/bin/python3 /usr/bin/colcon build --symlink-install
PYTHONPATH="$PWD/.venv/lib/python3.12/site-packages:${PYTHONPATH:-}" \
  /usr/bin/python3 /usr/bin/colcon test --event-handlers console_direct+
/usr/bin/python3 /usr/bin/colcon test-result --verbose
```

Humble 容器验证入口为 `tools/test_humble.sh`。30 分钟 soak 使用真实
WebSocket/ROS 2/SQLite 链路并带有 `soak` marker，详见[验证说明](docs/verification.md)。

## 配置与协议

- 本地可运行配置：[robot_agent.local.yaml](src/robot_agent/config/robot_agent.local.yaml)
- 生产 mTLS 模板：[robot_agent.production.example.yaml](src/robot_agent/config/robot_agent.production.example.yaml)
- 协议说明：[protocol-v1.md](docs/protocol-v1.md)
- 运维与部署：[operations.md](docs/operations.md)
- 可视化测试台：[web-demo.md](docs/web-demo.md)
- 生成的 JSON Schema：[schemas](src/robot_agent/schemas)

配置中的 `commands` 是唯一 ROS 出口。修改注册表后，能力摘要会变化，BFF 可据此识别部署能力。生产上线前必须把模板里的 ROS 名称、类型、参数 Schema、资源集合和 safe-stop 端点替换为真机经过评审的接口。

若生产使用 WebSocket server 模式，必须在 `tls.peer_certificate_sha256` 中填写
BFF 客户端叶证书 DER 的小写 SHA-256 指纹；证书轮换时应同步更新配置并重新验收。

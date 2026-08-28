# H-IDE 工作站端到端联调启动与关闭手册

## 文档目的

本文汇总 H-IDE Office V2 端到端联调的一键启动方式、底层逐项启动顺序、
状态检查、日志位置与安全关闭步骤，供开发者理解和复现以下链路：

```text
PWA
→ H-IDE Control API
→ Planning Agent
→ H-IDE Vision
→ Robot Agent Gateway
→ g1-sim Robot Agent
→ Nav2
→ SONIC
→ MuJoCo Office V3
```

适用仓库：

- `/home/phi/learn-projects/h-ide`
- `/home/phi/learn-projects/h-ide-agent`
- `/home/phi/learn-projects/h-ide-vision`
- `/home/phi/learn-projects/robotagent`
- `/home/phi/learn-projects/h-ide-mujoco-sonic`

DAAAM 当前不是常驻联调服务，只在离线导入语义记忆时运行工具，不属于本文的
启动链。

## 推荐方式：一键启动完整链路

在图形桌面的普通终端中执行：

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic

HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT=1 \
  ./scripts/start-workstation-demo.sh --viewer
```

参数说明：

| 参数 | 作用 |
| --- | --- |
| `--viewer` | 启动 MuJoCo 原生第三人称窗口和第一人称窗口 |
| `--rebuild` | 强制重建 H-IDE、Agent 等 Compose 镜像 |
| `--motion-core` | 仅用于隔离运动核心验收，不是端到端联调模式；必须与 `--viewer` 同时使用 |

修改过 H-IDE、Agent 或 Docker 内源码后执行：

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic

HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT=1 \
  ./scripts/start-workstation-demo.sh --rebuild --viewer
```

如果不需要桌面 MuJoCo 窗口，可以省略 `--viewer`：

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
./scripts/start-workstation-demo.sh
```

## 一键脚本管理的服务

| 顺序 | 服务 | 主要端口或接口 | 所属项目 |
| ---: | --- | --- | --- |
| 1 | Redis、PostgreSQL | Compose 内部网络 | `h-ide` |
| 2 | Control API、PWA、Caddy | `8080` | `h-ide` |
| 3 | Planning Agent | `4111` | `h-ide-agent` |
| 4 | Scene API、Scene Worker | `8090` | `h-ide` |
| 5 | 基础 Robot Agent Gateway | `8093` | `h-ide` |
| 6 | H-IDE Vision：Qwen3-VL-Plus + SAM 2 | HTTP `8092`、gRPC `50061` | `h-ide-vision` |
| 7 | MuJoCo Plant、RGB-D、TF | `8794` | `h-ide-mujoco-sonic` |
| 8 | SONIC | `8792` | `h-ide-mujoco-sonic` |
| 9 | Leader-demo gesture proxy | Docker bridge `8798`（仅 `g1-sim` wave） | `h-ide-mujoco-sonic` |
| 10 | MuJoCo 第三人称/第一人称 Viewer | X11 窗口 | `h-ide-mujoco-sonic` |
| 11 | ROS 速度、观测、Safe Stop 桥 | ROS Domain `232` | `h-ide-mujoco-sonic` |
| 12 | Nav2 | ROS 2 actions/lifecycle | `h-ide-mujoco-sonic` 镜像 |
| 13 | Strict Navigation Adapter | `/h_ide/strict/*` | `h-ide-mujoco-sonic` |
| 14 | 仿真 Robot Agent Gateway | `8094` | `h-ide` |
| 15 | Observation Gateway | Vision gRPC 上行 | `robotagent` |
| 16 | `g1-sim` Robot Agent | 诊断 `8796` | `robotagent` |
| 17 | Docker 诊断代理 | `8797` | `h-ide-mujoco-sonic` |
| 18 | SAFE_HOLD 启动门 | 只读交叉检查 | `h-ide-mujoco-sonic` |

PWA 已构建到 Control API 镜像中，不存在单独的生产 PWA 进程。

## 启动后的状态检查

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
./scripts/workstation-demo-status.sh
```

完整启动后应确认以下项目全部为 `ok`：

- H-IDE Web / Control API
- H-IDE Vision：`qwen3-vl-plus + sam2`
- MuJoCo Plant + RGB-D
- SONIC telemetry
- Robot Agent diagnostics
- Docker diagnostics proxy
- `g1-sim` Robot Agent Gateway
- Nav2 lifecycle：`active`

访问地址：

- PWA：`http://127.0.0.1:8080/`
- 仿真观测台：`http://127.0.0.1:8080/developer/simulation`

日志目录：

```bash
ls -1 /run/user/1000/h-ide-mujoco-demo-1000
tail -f /run/user/1000/h-ide-mujoco-demo-1000/*.log
```

查看脚本创建的用户级 transient units：

```bash
systemctl --user list-units 'h-ide-mujoco-demo-*' --all --no-pager
```

---

# 底层逐项启动命令

本节用于理解启动脚本。长驻命令若真的手工执行，必须分别占用独立终端；手动启动
还需要自行实现脚本中的超时、身份检查、就绪门和失败回滚，因此不用于正式验收。

尖括号形式的值代表本机配置或凭据，不应按字面填写，也不应把真实凭据记录在本文。

## 0. 公共路径与 ROS 环境

```bash
export HIDE_ROOT=/home/phi/learn-projects/h-ide
export SIM_ROOT=/home/phi/learn-projects/h-ide-mujoco-sonic
export ROBOT_ROOT=/home/phi/learn-projects/robotagent
export VISION_ROOT=/home/phi/learn-projects/h-ide-vision

export SIM_CONFIG="$SIM_ROOT/config/ros-headless.yaml"
export ROBOT_CONFIG="$ROBOT_ROOT/src/robot_agent/config/robot_agent.mujoco.yaml"
export OBS_CONFIG="$ROBOT_ROOT/src/robot_observation_gateway/config/robot_observation_gateway.mujoco.yaml"

export ROS_DOMAIN_ID=232
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'

export DISPLAY="${DISPLAY:-:0}"
export HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT=1
```

解析 Docker 可访问的宿主机地址：

```bash
export HIDE_DOCKER_OBSERVATION_BIND_HOST="$(
  "$SIM_ROOT/.venv/bin/python" \
    -m hide_mujoco_sonic.network \
    --requested auto
)"

test -n "$HIDE_DOCKER_OBSERVATION_BIND_HOST"
export SIMULATION_ROBOT_AGENT_DIAGNOSTICS_URL=http://host.docker.internal:8797
```

## 1. H-IDE Compose 工作站服务

```bash
cd "$HIDE_ROOT"

docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.simulation.yml \
  config --quiet

docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.simulation.yml \
  up -d
```

需要重建镜像时使用：

```bash
docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.simulation.yml \
  up -d --build
```

检查：

```bash
docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.simulation.yml \
  ps
```

## 2. H-IDE Vision

新终端：

```bash
cd "$VISION_ROOT"

export DASHSCOPE_API_KEY='<DASHSCOPE_API_KEY>'
export HIDE_VISION_AGENT_TOKEN='<HIDE_VISION_AGENT_TOKEN>'
export HIDE_VISION_CONTROL_TOKEN='<HIDE_VISION_CONTROL_TOKEN>'
export HIDE_VISION_SAM2_CHECKPOINT='<HIDE_VISION_SAM2_CHECKPOINT>'

export HIDE_VISION_HOST=0.0.0.0
export HIDE_VISION_HTTP_PORT=8092
export HIDE_VISION_GRPC_HOST=127.0.0.1
export HIDE_VISION_GRPC_PORT=50061
export HIDE_VISION_DATABASE_PATH="$VISION_ROOT/var/h-ide-vision-qwen.sqlite3"
export HIDE_VISION_EVIDENCE_DIR="$VISION_ROOT/var/evidence"
export HIDE_VISION_VLM_PROVIDER=qwen
export HIDE_VISION_QWEN_MODEL=qwen3-vl-plus
export HIDE_VISION_SEGMENTATION_PROVIDER=sam2
export HIDE_VISION_SAM2_DEVICE=cuda
export HIDE_VISION_ALLOW_INSECURE_LOOPBACK=true

"$VISION_ROOT/.venv/bin/h-ide-vision" serve
```

检查：

```bash
curl -fsS http://127.0.0.1:8092/healthz
```

必须确认 `releaseState=pure_vlm_product_profile`，且 Qwen3-VL-Plus 与 SAM 2 均
`available=true`。

## 3. SONIC 模型锁与 Office V2 场景

```bash
cd "$SIM_ROOT"

"$SIM_ROOT/.venv/bin/python" \
  "$SIM_ROOT/tools/verify_model_lock.py" \
  --lock "$SIM_ROOT/config/sonic-models.lock.json" \
  --config "$SIM_CONFIG"

"$SIM_ROOT/.venv/bin/python" \
  "$SIM_ROOT/tools/build_scene.py" \
  --check
```

发布并绑定 Office V2 场景：

```bash
cd "$HIDE_ROOT"

SCENE_API_TOKEN='<CONTROL_API_TOKEN>' \
SCENE_API_INTERNAL_TOKEN='<SCENE_API_INTERNAL_TOKEN>' \
SIMULATION_ROBOT_AGENT_ID=g1-sim \
  "$HIDE_ROOT/.venv/bin/python" \
  -m services.scene_api.office_v2_bootstrap \
  --base-url http://127.0.0.1:8090 \
  --timeout-seconds 900
```

## 4. Nav2 Office V2 镜像

```bash
cd "$SIM_ROOT"

export OFFICE_V2_SHA="$(
  "$SIM_ROOT/.venv/bin/python" -c \
    'import json; print(json.load(open("scenes/office_v2/generated/runtime_manifest.json"))["scene_source_sha256"])'
)"

test -n "$OFFICE_V2_SHA"

docker build \
  -t h-ide-nav2-sim:jazzy \
  --build-arg "OFFICE_V2_SOURCE_SHA256=$OFFICE_V2_SHA" \
  -f docker/nav2/Dockerfile \
  .
```

此时只构建镜像，不启动 Nav2。Nav2 必须等 MuJoCo 发布 `map → base_link` 后再启动。

## 5. MuJoCo Plant

新终端：

```bash
cd "$SIM_ROOT"

export HIDE_MUJOCO_ONSCREEN_VIEWER=1
export HIDE_MUJOCO_DEFER_ONSCREEN_VIEWER=1
export DISPLAY="${DISPLAY:-:0}"

"$SIM_ROOT/.venv/bin/python" -u \
  -m hide_mujoco_sonic plant \
  --config "$SIM_CONFIG"
```

检查：

```bash
curl -fsS http://127.0.0.1:8794/healthz
```

## 6. SONIC

新终端：

```bash
cd "$SIM_ROOT"

export HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT=1

"$SIM_ROOT/.venv/bin/python" -u \
  -m hide_mujoco_sonic sonic \
  --config "$SIM_CONFIG"
```

检查：

```bash
curl -fsS http://127.0.0.1:8792/healthz
curl -fsS http://127.0.0.1:8792/telemetry
```

启动脚本不会因为 8792 已监听就继续；它还要求模型完成加载、权威 50 Hz 仿真时钟
稳定、learned IDLE 正常且 `motion_ready=true`。

## 7. MuJoCo Viewer

SONIC ready 后，启动脚本验证 Plant PID、所有者与命令行，再发送：

```bash
kill -USR1 '<经过身份验证的 MuJoCo Plant PID>'
```

该信号请求 Plant 创建第三人称原生 Viewer。不要绕过启动脚本直接向不明 PID 发送
信号。

第一人称 Viewer 是独立进程：

```bash
cd "$SIM_ROOT"
"$SIM_ROOT/.venv/bin/python" "$SIM_ROOT/tools/live_first_person_viewer.py"
```

## 8. ROS `/cmd_vel` 桥

新终端：

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=232
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
export PYTHONPATH="$SIM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SIM_ROOT"
python3 -u -m hide_mujoco_sonic ros-bridge --config "$SIM_CONFIG"
```

数据方向：`Nav2 /cmd_vel → SONIC 速度目标`。

## 9. ROS 观测桥

新终端：

```bash
source /opt/ros/jazzy/setup.bash
source "$ROBOT_ROOT/tools/activate-local-realsense-jazzy.bash"

export ROS_DOMAIN_ID=232
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
export PYTHONPATH="$SIM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SIM_ROOT"
python3 -u -m hide_mujoco_sonic ros-observation-bridge --config "$SIM_CONFIG"
```

数据方向：`MuJoCo RGB-D / CameraInfo / TF / 根位姿 → ROS 2`。

## 10. ROS Safe Stop 服务

新终端：

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=232
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
export PYTHONPATH="$SIM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SIM_ROOT"
python3 -u -m hide_mujoco_sonic safe-stop-service --config "$SIM_CONFIG"
```

## 11. Nav2

当前容器已经存在但停止时：

```bash
docker start h-ide-nav2-sim
```

首次创建时：

```bash
docker run -d \
  --name h-ide-nav2-sim \
  --network host \
  --restart unless-stopped \
  -e ROS_DOMAIN_ID=232 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e "CYCLONEDDS_URI=$CYCLONEDDS_URI" \
  h-ide-nav2-sim:jazzy
```

检查全部生命周期节点：

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=232
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'

ros2 lifecycle get /map_server
ros2 lifecycle get /planner_server
ros2 lifecycle get /controller_server
ros2 lifecycle get /behavior_server
ros2 lifecycle get /bt_navigator
ros2 lifecycle get /waypoint_follower
```

所有节点都必须为 `active [3]`。

## 12. Strict Navigation Adapter

新终端：

```bash
source /opt/ros/jazzy/setup.bash
source "$ROBOT_ROOT/tools/activate-local-nav2-jazzy.bash"

export ROS_DOMAIN_ID=232
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
export PYTHONPATH="$SIM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python3 -u "$SIM_ROOT/tools/strict_navigation_adapter.py" \
  --command-timeout-seconds 2400
```

它提供：

- `/h_ide/strict/navigate_to_pose`
- `/h_ide/strict/spin`

普通 `FollowPath` 输出继续保持隔离。

## 13. 仿真 Robot Agent Gateway

新终端：

```bash
cd "$HIDE_ROOT"

export ROBOT_AGENT_GATEWAY_BACKEND=websocket_inbound
export ROBOT_AGENT_ID=g1-sim
export ROBOT_AGENT_ALLOW_PLAINTEXT_LOOPBACK=true
export ROBOT_AGENT_GATEWAY_TOKEN='<SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN>'
export ROBOT_AGENT_OBSERVATION_PREFLIGHT_TIMEOUT_SECONDS=2
export ROBOT_AGENT_OBSERVATION_PREFLIGHT_MAX_AGE_MS=500

unset ROBOT_AGENT_CAPABILITIES_DIGEST
unset ROBOT_AGENT_CAPABILITIES

"$HIDE_ROOT/.venv/bin/uvicorn" \
  services.robot_agent_gateway.main:app \
  --host 0.0.0.0 \
  --port 8094 \
  --workers 1
```

这是仿真专用 Gateway `8094`，不同于 Compose 中的基础 Gateway `8093`。

## 14. Observation Gateway

新终端：

```bash
source /opt/ros/jazzy/setup.bash
source "$ROBOT_ROOT/install/setup.bash"
source "$ROBOT_ROOT/tools/activate-local-realsense-jazzy.bash"

export ROS_DOMAIN_ID=232
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
export PYTHONPATH="$ROBOT_ROOT/src/robot_observation_gateway${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROBOT_ROOT"

"$ROBOT_ROOT/.venv/bin/python" -u \
  -m robot_observation_gateway.cli \
  --config "$OBS_CONFIG"
```

数据方向：`ROS RGB-D / TF → H-IDE Vision gRPC 50061`。

## 15. `g1-sim` Robot Agent

新终端：

```bash
source /opt/ros/jazzy/setup.bash
source "$ROBOT_ROOT/tools/activate-local-nav2-jazzy.bash"
source "$ROBOT_ROOT/install/setup.bash"

export ROS_DOMAIN_ID=232
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
export PYTHONPATH="$ROBOT_ROOT/src/robot_agent${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROBOT_ROOT"

"$ROBOT_ROOT/.venv/bin/python" -u \
  -m robot_agent.cli run \
  --config "$ROBOT_CONFIG"
```

Robot Agent 负责能力发布、控制租约、幂等、位姿/观测新鲜度、Nav2 Goal/Cancel 和
SONIC 常驻控制边界，并在 `8796` 提供本地只读诊断。

## 16. Docker Robot Agent 诊断代理

新终端：

```bash
cd "$SIM_ROOT"

"$SIM_ROOT/.venv/bin/python" -u \
  -m hide_mujoco_sonic.robot_agent_diagnostics_proxy \
  --bind-host "$HIDE_DOCKER_OBSERVATION_BIND_HOST"
```

它把本机 `8796` 的诊断约束性代理到 Docker 可访问的 `8797`。

## 17. 最终 SAFE_HOLD 启动门

```bash
cd "$SIM_ROOT"

"$SIM_ROOT/.venv/bin/python" \
  "$SIM_ROOT/tools/startup_safe_hold_gate.py" \
  --deadline-seconds 30 \
  --interval-seconds 0.5 \
  --http-timeout-seconds 1
```

该检查要求连续两次确认：

- Plant 存活、数值有限并保持静止；
- SONIC 存活且 learned IDLE 正常；
- Nav2 就绪；
- Robot Agent 无活动命令；
- 控制权、safe-stop 和 boot/session 身份相互一致。

只有该门通过，完整端到端启动才算成功。

## 可选：离线 Whisper

默认流式语音使用 DashScope，不要求本地 Whisper。只有显式启用离线 WAV 降级时才
启动：

```bash
cd /home/phi/learn-projects/h-ide

docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.simulation.yml \
  --profile speech up -d whisper
```

---

# 安全关闭完整链路

关闭前先在 PWA 取消正在执行的 Mission，并等待终止回执。不要在活动 Mission 中直接
杀死 MuJoCo、SONIC 或 Robot Agent。

启动脚本使用固定名称的用户级 transient units。按依赖逆序停止：

```bash
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"

for HIDE_STOP_UNIT in \
  robot-agent-diagnostics-proxy \
  leader-demo-gesture-proxy \
  robot-agent-g1-sim \
  observation-gateway \
  sim-robot-agent-gateway \
  strict-navigation-adapter; do
  systemctl --user stop \
    "h-ide-mujoco-demo-${HIDE_STOP_UNIT}.service" 2>/dev/null || true
done

docker stop --timeout 10 h-ide-nav2-sim 2>/dev/null || true

for HIDE_STOP_UNIT in \
  ros-safe-stop \
  ros-observation-bridge \
  ros-cmd-vel-bridge \
  first-person-viewer \
  sonic-driver \
  mujoco-plant \
  h-ide-vision-qwen; do
  systemctl --user stop \
    "h-ide-mujoco-demo-${HIDE_STOP_UNIT}.service" 2>/dev/null || true
done

cd /home/phi/learn-projects/h-ide

docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.simulation.yml \
  down --timeout 30
```

如果单独启用了 Whisper，但它没有随 Compose 在超时窗口内退出：

```bash
docker stop --timeout 10 robot-action-console-whisper-1
```

关闭命令不带 `-v`，因此以下数据继续保留：

- Control API 数据库；
- Planning Agent 数据；
- Scene 数据；
- PostgreSQL / Redis 数据；
- Vision 数据与证据；
- 模型缓存和运行日志。

## 关闭后的确认

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
./scripts/workstation-demo-status.sh
```

所有联调服务都应显示 `unavailable`，并且：

```bash
docker ps --format '{{.Names}}\t{{.Status}}'
systemctl --user list-units 'h-ide-mujoco-demo-*' --all --no-pager
```

不应再显示运行中的 H-IDE、Vision、MuJoCo、SONIC、Robot Agent 或 Nav2 服务。
与本项目无关的容器，例如 `cli-proxy-api`，不在本文的启动或关闭范围内。

## 为什么正式联调必须使用一键脚本

一键脚本不仅是命令集合，还负责：

- 检查模型 lock、场景 SHA 和生成产物；
- 防止复用残缺的 MuJoCo/SONIC/Nav2 拓扑；
- 校验 X11 Viewer 的 Xauthority 和进程身份；
- 等待 Vision、Plant、SONIC、TF、Nav2 和 Strict Navigation 的真实 readiness；
- 保证 Robot Agent 最后接入，避免冷启动阶段提前获取控制租约；
- 对 Plant、SONIC、Robot Agent 做连续两次 SAFE_HOLD 交叉确认；
- 将所有长驻进程放入可审计的用户级 systemd transient units；
- 把每个组件的日志写入统一运行目录。

因此，逐项命令适合学习和故障定位，不应作为正式验收启动方式。

## 相关文件

- 启动脚本：`/home/phi/learn-projects/h-ide-mujoco-sonic/scripts/start-workstation-demo.sh`
- 状态脚本：`/home/phi/learn-projects/h-ide-mujoco-sonic/scripts/workstation-demo-status.sh`
- 仿真配置：`/home/phi/learn-projects/h-ide-mujoco-sonic/config/ros-headless.yaml`
- Robot Agent 配置：`/home/phi/learn-projects/robotagent/src/robot_agent/config/robot_agent.mujoco.yaml`
- Observation Gateway 配置：`/home/phi/learn-projects/robotagent/src/robot_observation_gateway/config/robot_observation_gateway.mujoco.yaml`
- H-IDE Compose：`/home/phi/learn-projects/h-ide/infra/docker-compose.yml`
- 仿真 Compose Overlay：`/home/phi/learn-projects/h-ide/infra/docker-compose.simulation.yml`

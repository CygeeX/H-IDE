# 联调运维手册

## 启动

推荐入口只有两个：

```bash
./integration/start-e2e.sh --viewer  # 开发者桌面
./integration/start-e2e.sh           # 无 onscreen viewer
```

首次构建或依赖变更后：

```bash
./integration/start-e2e.sh --rebuild --viewer
```

`start-e2e.sh` 只负责从 monorepo 根导出可靠路径；真正启动顺序、模型锁、Office V2
场景哈希、ROS/TF、SONIC 50 Hz motion gate 和 Robot Agent safe-hold gate 都由
`components/h-ide-mujoco-sonic/scripts/start-workstation-demo.sh` 执行。

## 状态与端口

```bash
SKILLSTORE_ROOT="$PWD" \
  ./components/h-ide-mujoco-sonic/scripts/workstation-demo-status.sh
```

| 服务 | 监听/用途 |
| --- | --- |
| PWA/Gateway | `127.0.0.1:8080` |
| Scene API | `127.0.0.1:8090` |
| Vision HTTP | `127.0.0.1:8092` |
| Planning Agent | `127.0.0.1:4111` |
| Simulation Robot Agent Gateway | `127.0.0.1:8094` |
| SONIC telemetry | `127.0.0.1:8792` |
| MuJoCo Plant/RGB-D | `127.0.0.1:8794` |
| Robot Agent diagnostics | `127.0.0.1:8796` |

ROS 仿真使用 domain 232；Unitree LowState/LowCmd 使用 domain 230。不要把 live
domain 231 混入当前终端。

## 日志

Host worker 日志：

```bash
runtime_root="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/h-ide-mujoco-demo-$(id -u)"
ls -l "$runtime_root"
tail -f "$runtime_root/mujoco-plant.log"
tail -f "$runtime_root/sonic-driver.log"
tail -f "$runtime_root/strict-navigation-adapter.log"
tail -f "$runtime_root/robot-agent-g1-sim.log"
```

Compose 日志：

```bash
docker compose --project-name skillstore-e2e \
  --env-file .env \
  -f components/h-ide/infra/docker-compose.yml \
  -f components/h-ide/infra/docker-compose.simulation.yml \
  ps

docker compose --project-name skillstore-e2e \
  --env-file .env \
  -f components/h-ide/infra/docker-compose.yml \
  -f components/h-ide/infra/docker-compose.simulation.yml \
  logs --tail 200 planning-agent control-api
```

## 停止和冷启动

```bash
./integration/stop-e2e.sh
./integration/preflight.sh --level L2
./integration/start-e2e.sh --viewer
```

停止脚本不删除 volume 或证据。部分 systemd worker、旧 Nav2 容器或旧 Robot
Agent 仍存活时，启动器会拒绝复用，这是 boot/session/epoch 一致性要求，不是
需要放宽的导航安全阈值。

## 恢复初始点

完整复位采用“停止整条运动链并重新启动”，因为 Office V2 的 `reset_policy` 固定为
`explicit_process_restart`：

```bash
./integration/stop-e2e.sh
./integration/start-e2e.sh --viewer
```

不要在 Plant 运行时直接改 qpos、单独重启 SONIC 或复用旧 Nav2；这会造成 odom、
localization epoch、控制历史和 Mission 状态不一致。

## 常见故障定位

### PWA 显示 planning 500/502

先检查 `planning-agent` 的 readiness 和 `DEEPSEEK_API_KEY`。供应商欠费/限流与
内部 token 不匹配会产生相似的外观；必须看服务日志的具体 upstream error，
不要把失败请求绕成模板执行或编译器结果。

### Mission 卡在“准备执行”

按顺序检查：Robot Agent capability/lease → fresh observation/epoch → strict Nav2
actions → Nav2 lifecycle → SONIC motion gate。不能只看 Plant 窗口是否存在。

### MuJoCo viewer 消失但进程仍在

检查 Plant `/state` 的 `viewerWorker`、Xauthority 所有者和 Plant log。viewer 是
Plant 的只读展示 worker；窗口状态不等于物理仿真 authority。

### 模型锁失败

运行：

```bash
cd components/h-ide-mujoco-sonic
.venv/bin/python tools/verify_model_lock.py \
  --lock config/sonic-models.lock.json \
  --config config/ros-headless.yaml --json
```

不要手改哈希以让它通过。重新获取锁定 revision，或按发布流程生成一个新的审计版。

## 变更和回退

当前 `main` 和 `integration/office-v2-e2e-20260827-r2` 标签代表同一冻结版。实验变更
从 feature 分支开始；合入前必须通过 L0/L1，涉及 live motion 时保存 L2 证据。
回退时检出冻结标签并重新执行 bootstrap/preflight，不要把旧 worktree 的 `.env`
或数据库提交进来。

# Office V2 联调运维手册

本文适用于 release candidate `skillstore-office-v2-e2e-20260901-r1`。命令从仓库根
目录运行，并使用 `ROOT="$(pwd -P)"` 派生本机路径。

## 1. 启动 profile

默认冻结 profile：

```bash
./integration/start-e2e.sh --viewer
```

领导演示推荐 profile：

```bash
./integration/start-e2e.sh --rebuild --viewer --full-straight-drive
```

无显示器时去掉 `--viewer`。`--rebuild` 只在首次运行或代码/依赖/场景改变后需要；运行
中的旧容器不会自动加载新源码。

`--full-straight-drive` 不是无限制直行：

- 只接纳地图额外膨胀后仍证明无碰撞的直线段；
- 当前单个 `DriveOnHeading` 上限 4 m，长路径仍会分成少量动作；
- 保持 SONIC 官方前进速度 0.40 m/s；
- 转向、终点、取消、故障和不满足条件的区段仍进入正常停止/复核边界；
- 无法建立宽轮廓证明时自动回退到 2 m continuous profile，再回退到默认短段 profile。

当前 Office V2 已移除走廊中央圆柱，验收和故障说明不再使用“绕中央柱子”。

## 2. 双 Skill 演示

挥手是 simulation-only 可选能力。录制前在本地 `.env` 中配置：

```dotenv
HIDE_LEADER_DEMO_GESTURES_ENABLED=true
HIDE_LEADER_DEMO_GESTURE_PROXY_TOKEN=<operator-generated-random-secret>
```

不要将真实令牌写进命令、文档或 Git。启动后按以下顺序：

1. 新建会话，在 Skill 上方选择“宇树 Unitree”，确认绑定 `g1-sim`。
2. 选择“巡检 Skill”，输入：`今天巡检开放办公室和会议室`。
3. 审核 Mission，确认机器人依次进入房间、停稳、多方向观察并输出按房间报告。
4. 切换“语音助手 Skill”，输入：`刚刚有看到电脑吗？`。
5. 确认答案来自上轮报告/证据，浏览器朗读文本，机器人在空闲时挥手后放下手臂。

挥手请求只在 G1 空闲且没有行走/停止边界时接纳；它不创建第二个 LowCmd 发布者，不
直接写 MuJoCo `qpos`。语音成功而挥手被拒绝时，普通问答本身仍可成功。

## 3. 状态、端口与 profile 证明

```bash
ROOT="$(pwd -P)"
SKILLSTORE_ROOT="$ROOT" \
  ./components/h-ide-mujoco-sonic/scripts/workstation-demo-status.sh
```

| 服务 | 监听/用途 |
| --- | --- |
| PWA / Gateway | `127.0.0.1:8080` |
| Scene API | `127.0.0.1:8090` |
| Vision HTTP | `127.0.0.1:8092` |
| Planning Agent | `127.0.0.1:4111` |
| Simulation Robot Agent Gateway | `127.0.0.1:8094` |
| SONIC telemetry | `127.0.0.1:8792` |
| MuJoCo Plant / RGB-D | `127.0.0.1:8794` |
| Robot Agent diagnostics | `127.0.0.1:8796` |
| Container diagnostics proxy | `8797`（仅受控主机桥接） |
| Leader-demo gesture proxy | `8798`（仅受控主机桥接） |

ROS 仿真 domain 为 232，Unitree LowState/LowCmd domain 为 230；不要把 live domain
231 混入当前终端。

确认 full-straight profile 时检查 strict adapter ROS health，而不是只观察窗口：

```bash
ROOT="$(pwd -P)"
bash -lc "source /opt/ros/jazzy/setup.bash && \
  source '$ROOT/components/robotagent/tools/activate-local-nav2-jazzy.bash' && \
  export ROS_DOMAIN_ID=232 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
  ros2 service call /h_ide/strict_navigation/health std_srvs/srv/Trigger '{}'"
```

`workstation-demo-status.sh` 还会检查 Vision、Plant、SONIC、Robot Agent、容器诊断与
gesture proxy。进程存活只表示 health；Mission 成功仍需终态回执和证据。

## 4. 日志

Host worker 日志：

```bash
runtime_root="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/h-ide-mujoco-demo-$(id -u)"
ls -l "$runtime_root"
tail -f "$runtime_root/mujoco-plant.log"
tail -f "$runtime_root/sonic-driver.log"
tail -f "$runtime_root/strict-navigation-adapter.log"
tail -f "$runtime_root/robot-agent-g1-sim.log"
tail -f "$runtime_root/leader-demo-gesture-proxy.log"
```

Compose 日志：

```bash
ROOT="$(pwd -P)"
docker compose --project-name skillstore-e2e \
  --env-file "$ROOT/.env" \
  -f "$ROOT/components/h-ide/infra/docker-compose.yml" \
  -f "$ROOT/components/h-ide/infra/docker-compose.simulation.yml" \
  logs --tail 200 planning-agent control-api robot-agent-gateway
```

日志和 Evidence 可能包含用户文本、机器人状态或图像，不得上传到公开 issue 或 Git。

## 5. 停止、冷启动与初始点复位

```bash
./integration/stop-e2e.sh
./integration/preflight.sh --level L2
./integration/start-e2e.sh --viewer --full-straight-drive
```

Office V2 `reset_policy` 是 `explicit_process_restart`。完整冷启动会统一更新 Plant boot、
SONIC 控制历史、Robot Agent session、Localization Epoch 和 Nav2 TF，是当前唯一受支持
的回初始点方式。

不要在运行时直接修改 `qpos`，也不要单独重启 SONIC 或继续旧 Mission。停止脚本不会
删除数据库 volume、Evidence 或外部模型；需要诊断时应先保留日志，再冷启动。

## 6. 首个管理员初始化

如果尚无账户，普通用户可从 PWA 注册。首个 ADMIN 必须在 Control API 数据库 schema
已创建、Control API 已停止时运行一次性离线 CLI：

```bash
ROOT="$(pwd -P)"
COMPOSE=(docker compose --project-name skillstore-e2e \
  --env-file "$ROOT/.env" \
  -f "$ROOT/components/h-ide/infra/docker-compose.yml" \
  -f "$ROOT/components/h-ide/infra/docker-compose.simulation.yml")

"${COMPOSE[@]}" stop control-api
"${COMPOSE[@]}" run --rm --no-deps control-api \
  h-ide-admin-bootstrap \
  --database /data/control.db \
  --username administrator \
  --display-name "Administrator" \
  --site-id default
"${COMPOSE[@]}" up -d control-api gateway
```

密码通过交互式隐藏输入读取。数据库不存在、schema 不完整、已有任何用户或审计写入失败
都会拒绝并回滚。不要使用旧数据库或把管理员密码放进 `.env`。

## 7. 常见故障

### Planning 500/502 或“服务不可用”

先检查 `planning-agent` readiness、内部 token 是否一致，再查看 DeepSeek 上游错误。欠费、
限流、超时和内部鉴权失败外观相近；不得把上游失败偷偷替换为另一条固定运动模板。

### Mission 卡在“准备执行”

按顺序检查 Robot Agent capability/lease、fresh observation/epoch、strict adapter、Nav2
lifecycle、TF 和 SONIC motion gate。Plant viewer 存在不代表这条链已经 ready。

### 巡检提前报告或漏掉观察

检查 Mission 是否包含每个房间的进入、内部验证、扫描和最终报告节点；报告 Evidence
必须晚于对应房间到达。不要将入口站点、当前帧描述或上一轮画面当作本轮室内巡检。

### 历史问答说没有电脑/椅子

检查上一轮报告是否保存规范化 label、房间、EvidenceRef 和 turn/Mission 关联。语音助手
只能依据已经完成并投影到会话记忆的事实回答，不应从当前帧或模型常识补写结果。

### 语音正常但没有挥手

检查 `.env` 的 gesture enable、Control API 到 `8798/healthz` 的主机桥接、SONIC
telemetry 的 gesture state，以及当前是否正在行走或停止。不要让第二个进程直接发布
LowCmd 作为修复。

### MuJoCo viewer 消失

查看 Plant `/state` 的 `viewerWorker` 与日志，区分只读 viewer 退出和 Plant authority
退出。时间跳变和渲染问题应记录诊断；不能用“窗口消失”推断机器人已倒下。

### 模型锁失败

```bash
ROOT="$(pwd -P)"
(cd "$ROOT/components/h-ide-mujoco-sonic" && \
  .venv/bin/python tools/verify_model_lock.py \
  --lock config/sonic-models.lock.json \
  --config config/ros-headless.yaml --json)
```

不要修改哈希来放行。重新获取锁定 revision，或走新的 release candidate 流程。

## 8. 回退

候选通过后，`integration/office-v2-e2e-20260901-r1` 是本版本的不可变回退入口。回退前
停止完整链路，在干净 worktree 检出标签，重新 bootstrap/preflight，再冷启动；不要把旧
`.env`、数据库或运行缓存提交到回退分支。

目标云—站点—边缘架构仍是 Proposed，不改变本候选当前的同机运维事实。参见
[平台架构](../../components/h-ide/docs/platform-architecture/README.md)。

# 快速开始：Office V2 端到端联调

本文面向第一次复刻 `skillstore-office-v2-e2e-20260901-r1` 的开发者。所有命令均从
仓库根目录执行，并通过 `ROOT="$(pwd -P)"` 派生路径，不依赖任何原开发者目录。

目标标签 `integration/office-v2-e2e-20260901-r1` 只有在 clean-clone 和 L0–L3 发布门禁通过后
才创建；标签发布前，审计人员使用 `release/office-v2-e2e-20260901` 分支。

## 1. 本候选提供什么

当前链路为：

```text
PWA → Control API / Conversation V2 / Mission V2
    → h-ide-agent → h-ide-vision
    → Robot Agent → Nav2 → SONIC → MuJoCo Office V2
```

可演示两项 Current Skill：

1. `inspection.office`：中文单房间/多房间巡检，进入房间后多方向观察并按房间报告。
2. `assistant.voice`：普通问答和巡检历史问答，浏览器中文语音朗读；仿真可选挥手。

`H-IDE Field Executive`、`Cognition Broker`、FARM/DAAAM Shadow Provider、统一 World
Model 和 Semantic Scene Compiler 属于 Proposed 架构，不在本候选的运行声明内。参见
[平台架构说明](../../components/h-ide/docs/platform-architecture/README.md)。

## 2. 主机前置条件

- Ubuntu 24.04；Ubuntu 22.04 仅用于容器或协议级验证。
- Python 3.12：H-IDE、MuJoCo/SONIC、Robot Agent Jazzy 环境。
- Python 3.11：`h-ide-vision`。
- Node.js `>=22.13.0`、npm、`uv`、Docker Engine 与 Compose v2。
- ROS 2 Jazzy、Nav2、TF2、`rmw_cyclonedds_cpp`、colcon。
- NVIDIA 驱动/CUDA、`nvidia-smi`、`git-lfs`；完整 L2 还需要与锁文件相符的模型。
- 可用的 `systemctl --user`；仅 `--viewer` 需要 X11 `DISPLAY`/`XAUTHORITY`。

不要仅凭“有 NVIDIA GPU”宣称复现成功。验收记录应保存 GPU SKU、显存、driver、CUDA、
ROS、Docker、Node 与 Python 版本；目标 PRO 6000 也必须记录准确 SKU。

## 3. 克隆并固定版本

发布后推荐：

```bash
git clone http://192.168.50.189:37085/zhongwenjun/skillstore.git
cd skillstore
git checkout --detach integration/office-v2-e2e-20260901-r1
ROOT="$(pwd -P)"
git status --short
```

发布审计阶段改为：

```bash
git checkout release/office-v2-e2e-20260901
ROOT="$(pwd -P)"
```

预期工作树为空。公开仓库匿名 clone 不需要凭据；内网 HTTP 不加密，写权限凭据不得
写入 remote URL 或 shell history。

## 4. 创建隔离环境

`bootstrap.sh` 默认只创建本地 `.env`、内部随机令牌和外部资产链接。它不会替你创建
全部 Python/ROS 环境，因此首次复刻还需执行：

```bash
./integration/bootstrap.sh --install-agent --provision-ros-prefixes

python3.12 -m venv components/h-ide/.venv
components/h-ide/.venv/bin/pip install -e 'components/h-ide[test,daaam]'

python3.12 -m venv components/h-ide-mujoco-sonic/.venv
components/h-ide-mujoco-sonic/.venv/bin/pip install \
  -e 'components/h-ide-mujoco-sonic[runtime,test]'

(cd components/h-ide-vision && uv sync --locked --extra dev)

(cd components/robotagent && \
  uv venv --python /usr/bin/python3.12 --system-site-packages .venv && \
  uv sync --python .venv/bin/python --all-groups && \
  bash -lc 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /usr/bin/colcon build --symlink-install')

(cd components/h-ide/apps/control-pwa && npm ci)
```

launcher 要求以下两个 operator-local ROS prefix：

```text
components/robotagent/.local/ros-jazzy-nav2/root/opt/ros/jazzy
components/robotagent/.local/ros-jazzy-realsense/root/opt/ros/jazzy
```

它们不在 Git 中，也不能从另一名开发者的工作区复制。`bootstrap` 的
`--provision-ros-prefixes` 会读取
`manifests/ros/jazzy-user-prefix-amd64.json`，下载精确 ROS deb、核对 SHA-256，并在无 sudo
情况下解包到上述目录。也可以单独运行：

```bash
./integration/provision-ros-prefixes.sh
```

L2 preflight 会验证两个 prefix 的 manifest 和必需 ROS package；不得注释掉 activation
script 绕过门禁。

如果某一步失败，不要借用仓库外的 `.venv`、`node_modules` 或 ROS `install` 目录。完整
clean-clone 过程见[复刻文档](../reproduction/README.md)。

## 5. 配置供应商凭据与演示选项

根目录 `.env` 必须是当前用户拥有的普通文件且权限为 `0600`：

```bash
chmod 600 "$ROOT/.env"
```

至少配置：

```dotenv
DEEPSEEK_API_KEY=<operator-provided>
DASHSCOPE_API_KEY=<operator-provided>
HIDE_VISION_SAM2_CHECKPOINT=<absolute-operator-cache-path>
```

当前文本 profile 为 DeepSeek `deepseek/deepseek-v4-flash`，视觉 profile 为
`qwen3-vl-plus` + SAM2。云模型可能欠费、限流或由供应商更新；固定 model ID 不等于
冻结外部服务本身。

要展示仿真挥手，还需在 `.env` 中启用并设置一个本地随机代理令牌；不要复制文档中的
固定示例或把真实值提交到 Git：

```dotenv
HIDE_LEADER_DEMO_GESTURES_ENABLED=true
HIDE_LEADER_DEMO_GESTURE_PROXY_TOKEN=<operator-generated-random-secret>
```

## 6. 获取外部资产

```bash
./integration/bootstrap.sh --fetch-groot-source
SKILLSTORE_ACCEPT_G1_MODEL_TERMS=1 \
  ./integration/bootstrap.sh --fetch-g1-model
./integration/bootstrap.sh --fetch-sonic-model
```

- G1 网格、GR00T/SONIC 源和 SONIC 权重按 manifest 固定 revision，由操作者获取。
- G1 网格再分发状态未确认，不随公开 Git 仓库分发。
- SAM2 checkpoint 由操作者单独取得，L2 会核验 SHA-256。
- DAAAM 研究环境、数据集和权重不随本候选分发，也不是默认启动依赖。

具体来源见 `manifests/vendor/`、`integration/integration.lock.yaml` 和
[安全文档](../security/README.md)。

## 7. 分级预检与启动

```bash
./integration/preflight.sh --level L0
./integration/preflight.sh --level L1
./integration/preflight.sh --level L2
./integration/start-e2e.sh --viewer --full-straight-drive
```

`--full-straight-drive` 只合并地图证明安全的直线段，单动作当前最多 4 m；转向、门口、
终点、取消和故障仍进入正常边界，不满足条件时自动回退到较短 profile。省略该参数即可
使用默认冻结 profile。

启动后：

- PWA：`http://127.0.0.1:8080/`
- 仿真观测：`http://127.0.0.1:8080/simulation`
- 状态汇总：`./components/h-ide-mujoco-sonic/scripts/workstation-demo-status.sh`

## 8. 初始化首个管理员

普通用户可以在 PWA 注册；需要管理员/开发者功能时，首个管理员必须通过离线 CLI 创建，
不能把密码放进 `.env`、命令参数或仓库。先让 Control API 创建数据库 schema，再停止它：

```bash
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

CLI 会通过隐藏输入读取并确认密码。它只允许在 `users` 表为空时成功一次；详细失败关闭
规则见
[`offline-admin-bootstrap.md`](../../components/h-ide/docs/operations/offline-admin-bootstrap.md)。

## 9. 推荐演示顺序

1. 点击“新任务”，在 Skill 上方选择“宇树 Unitree”，确认后端为 `g1-sim`。
2. 选择“巡检 Skill”，输入：`今天巡检开放办公室和会议室`。
3. 审核并执行 Mission；确认机器人依次进入两个房间，停稳、多方向观察，最后输出
   Office V2 定向巡检报告。
4. 切换“语音助手 Skill”，输入：`刚刚有看到电脑吗？`。
5. 确认回答引用上一轮报告中的房间和证据，并由浏览器朗读；机器人空闲时完成一次挥手
   并放下手臂。

不要用直接调用 Nav2 action 代替该验收，否则会跳过自然语言、Mission 审核、会话记忆、
视觉证据和 Robot Agent 权限边界。

## 10. 停止和复位

```bash
./integration/stop-e2e.sh
./integration/start-e2e.sh --viewer --full-straight-drive
```

Office V2 的完整复位策略是冷启动整条运动链；不要在 Plant 运行时直接修改 `qpos` 或
单独重启 SONIC。更多监控和故障定位见[运维手册](../operations/README.md)。

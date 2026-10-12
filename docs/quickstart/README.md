# 快速开始：Office V2 端到端联调

本文面向第一次复刻 `skillstore-office-v2-e2e-20260901-r1` 的开发者。所有命令均从
仓库根目录执行，并通过 `ROOT="$(pwd -P)"` 派生路径，不依赖任何原开发者目录。

目标标签 `integration/office-v2-e2e-20260901-r1` 只有在 clean-clone 和 L0–L3 发布门禁通过后
才创建；标签发布前，审计人员使用 `release/office-v2-e2e-20260901` 分支。

## 1. 本冻结 Demo 提供什么

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
Model 和 Semantic Scene Compiler 属于 Proposed 架构，不在本冻结 Demo 的运行声明内。参见
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

工作站 topology 使用固定 Compose/container/systemd 名称。在开始另一份 clone 前，必须先从
当前活动 clone 停止服务：

```bash
./integration/stop-e2e.sh
```

两个 clone 不能并行运行本 profile。发布和 clean-clone 审计禁止 shallow/partial clone；
不要使用 `--depth`、`--shallow-*` 或 `--filter`。

发布后推荐：

```bash
git clone http://192.168.50.189:37085/zhongwenjun/skillstore.git
cd skillstore
git checkout --detach integration/office-v2-e2e-20260901-r1
ROOT="$(pwd -P)"
git status --short
test "$(git rev-parse --is-shallow-repository)" = false
```

发布审计阶段改为：

```bash
git checkout release/office-v2-e2e-20260901
ROOT="$(pwd -P)"
```

预期工作树为空。公开仓库匿名 clone 不需要凭据；内网 HTTP 不加密，写权限凭据不得
写入 remote URL 或 shell history。

## 4. 创建隔离环境

`bootstrap.sh` 创建本地 `.env`、内部随机令牌、外部资产链接和锁定 ROS prefix；
`setup-environments.sh` 根据发布 manifest 创建全部 Python/Node/ROS 环境：

```bash
./integration/bootstrap.sh --provision-ros-prefixes
./integration/setup-environments.sh
```

普通开发者默认使用自己账号下的持久缓存。只有发布 clean audit 才在上述两条命令之前
创建并导出一次性隔离根，并在整个审计 shell 中保持该值：

```bash
audit_external_root="$(mktemp -d)"
chmod 700 "$audit_external_root"
export SKILLSTORE_EXTERNAL_ROOT="$audit_external_root"
```

该值不应复制进 Git。全新 external root 是 release clean audit 的硬性要求，但普通开发者
无需每次重新下载缓存；两者仍不得跨 clone 并行使用同一固定运行 topology。当前 SONIC lock 的路径
契约仍使用 Hugging Face 默认 cache；该 cache 可以复用，但启动前会逐文件校验大小与
SHA-256，`SONIC_HF_CACHE` 在本发行版必须留空。

仿真环境严格安装 `torch 2.11.0+cu128`、`torchvision 0.26.0+cu128` 和
`cyclonedds 11.0.1`；Unitree 上游旧 CycloneDDS 要求的兼容覆盖及原因记录在
`manifests/python/simulation-runtime.json`。不要改用浮动 PyPI CUDA 13 wheel。

launcher 要求以下两个 operator-local ROS prefix：

```text
components/robotagent/.local/ros-jazzy-nav2/root/opt/ros/jazzy
components/robotagent/.local/ros-jazzy-realsense/root/opt/ros/jazzy
```

它们不在 Git 中，也不能从另一名开发者的工作区复制。`bootstrap` 的
`--provision-ros-prefixes` 会读取
`manifests/ros/jazzy-user-prefix-amd64.json`，从清单锁定的 ROS buildfarm 不可变快照下载
精确 ROS deb、核对 SHA-256，并在无 sudo 情况下解包到上述目录。也可以单独运行：

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
`qwen3-vl-plus` + `sam2.1_hiera_tiny.pt`。云模型可能欠费、限流或由供应商更新；固定 model ID 不等于
冻结外部服务本身。

PWA/浏览器使用同源 server-side `hide_session` Cookie 与 double-submit CSRF；写请求
携带 `x-csrf-token`。`CONTROL_API_TOKEN` 由 bootstrap 生成，但只保留给 legacy
loopback/internal compatibility，不能通过 `VITE_*`、URL 或页面脚本交给浏览器。

要展示仿真挥手，还需在 `.env` 中启用并设置一个本地随机代理令牌；不要复制文档中的
固定示例或把真实值提交到 Git：

```dotenv
HIDE_LEADER_DEMO_GESTURES_ENABLED=true
HIDE_LEADER_DEMO_GESTURE_PROXY_TOKEN=<operator-generated-random-secret>
```

## 6. 获取外部资产

使用仓库内的 operator-fetch 脚本安装本发行固定的 SAM 2.1 Hiera Tiny checkpoint：

```bash
sam2_dir="${SKILLSTORE_EXTERNAL_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/skillstore}/sam2"
install -d -m 700 "$sam2_dir"
./integration/fetch-sam2-checkpoint.sh \
  --target "$sam2_dir/sam2.1_hiera_tiny.pt" \
  --env-file "$ROOT/.env"
```

脚本从 `manifests/vendor/sam2-checkpoint.json` 读取官方 URL、固定文件名、byte size 和
SHA-256，在同目录私有临时文件中完成 HTTPS 下载和校验，再进行原子 no-clobber 安装；
已有正确文件只验证复用。它只安全更新 `.env` 中的 checkpoint 路径，不会 source 或导出
其中的其他秘密。Checkpoint 保持 operator-fetch，不提交 Git。

```bash
./integration/bootstrap.sh --fetch-groot-source
SKILLSTORE_ACCEPT_G1_MODEL_TERMS=1 \
  ./integration/bootstrap.sh --fetch-g1-model
./integration/bootstrap.sh --fetch-sonic-model
```

- G1 网格、GR00T/SONIC 源和 SONIC 权重按 manifest 固定 revision，由操作者获取。
- G1 网格再分发状态未确认，不随公开 Git 仓库分发。
- SAM2 checkpoint 必须使用上述 `sam2.1_hiera_tiny.pt` 官方文件，L2 会再次核验 SHA-256。
- DAAAM 研究环境、数据集和权重不随本发行版分发，也不是默认启动依赖。

具体来源见 `manifests/vendor/`、`integration/integration.lock.yaml` 和
[安全文档](../security/README.md)。

## 7. 分级预检与启动

```bash
./integration/preflight.sh --level L0
./integration/preflight.sh --level L1
./integration/preflight.sh --level L2
./integration/stop-e2e.sh
./integration/start-e2e.sh --rebuild --viewer --full-straight-drive
```

clean audit 必须传 `--rebuild`，并在启动前再次执行 `stop-e2e.sh`，防止固定名称的旧容器或
systemd unit 被误当成当前 clone 的服务。

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

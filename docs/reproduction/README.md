# 从干净克隆复刻 Office V2 联调

本文区分“能够 clone”与“能够复现端到端行为”。目标版本为
`skillstore-office-v2-e2e-20260901-r1`，目标不可变标签为
`integration/office-v2-e2e-20260901-r1`。

## 1. 建立真正干净的工作区

发布后：

```bash
git clone http://192.168.50.189:37085/zhongwenjun/skillstore.git skillstore-clean
cd skillstore-clean
git checkout --detach integration/office-v2-e2e-20260901-r1
ROOT="$(pwd -P)"
git status --short
git show --no-patch --decorate HEAD
./integration/acceptance.sh --level L0
```

发布审计时改为检出 `release/office-v2-e2e-20260901`。候选必须满足：

- 工作树为空；
- 当前树无 gitlink/submodule、`.env`、数据库、日志、Evidence、模型或权重；
- 不依赖仓库外的 sibling project、旧 `.venv`、旧 `node_modules` 或 ROS build；
- 关键 runtime selector 不含开发者绝对路径；
- L0 通过。

匿名 clone 不需要 token。因为仓库地址是内网 HTTP，写权限凭据只允许通过本机
credential helper/askpass 使用，不能拼进 URL。

## 2. 记录主机 manifest

复现报告至少记录以下只读输出，不把主机序列号、用户名或凭据提交到公开仓库：

```bash
uname -a
lsb_release -ds
nvidia-smi
docker version
docker compose version
bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 doctor --report'
node --version
npm --version
python3.12 --version
python3.11 --version
uv --version
```

“PRO 6000”必须记录准确产品代际、显存、driver 和 CUDA；不能用计划采购型号代替实际
验收主机。

## 3. 创建隔离依赖环境

```bash
./integration/provision-ros-prefixes.sh
./integration/setup-environments.sh
```

安装脚本从 `manifests/python/simulation-runtime.json` 固定 PyTorch CUDA 12.8、
TorchVision、CycloneDDS、LeRobot 与 Unitree bridge 组合；它不会使用 sibling repo、系统中
已有的虚拟环境或浮动 CUDA wheel。

预期存在：

```text
components/robotagent/install/setup.bash
components/robotagent/tools/activate-local-nav2-jazzy.bash
components/robotagent/tools/activate-local-realsense-jazzy.bash
```

两个 activation script 还要求 operator-local prefix：

```text
components/robotagent/.local/ros-jazzy-nav2/root/opt/ros/jazzy
components/robotagent/.local/ros-jazzy-realsense/root/opt/ros/jazzy
```

这些 prefix 由 `integration/provision-ros-prefixes.sh` 根据
`manifests/ros/jazzy-user-prefix-amd64.json` 从零生成：脚本固定 package version、核对 deb
SHA-256，并在无 sudo 情况下解包。禁止复制原工作站 `.local` 或注释掉 launcher 的
activation gate。

不同 Python minor 版本不能共用 `.venv`。不得把生成的环境、ROS `build/install/log`、
PWA `dist` 或 `node_modules` 提交到发行仓库。

## 4. 创建秘密与获取外部资产

```bash
./integration/bootstrap.sh --provision-ros-prefixes
./integration/setup-environments.sh
chmod 600 "$ROOT/.env"
```

填写 DeepSeek、DashScope、SAM2 checkpoint 和本地演示选项。内部 service token 由
bootstrap 随机生成；不要复制原开发者 `.env` 或数据库。

```bash
./integration/bootstrap.sh --fetch-groot-source
SKILLSTORE_ACCEPT_G1_MODEL_TERMS=1 \
  ./integration/bootstrap.sh --fetch-g1-model
./integration/bootstrap.sh --fetch-sonic-model
```

脚本必须检出 manifest 中的完整 revision 并验证哈希，不允许下载浮动 `main`。G1 网格、
SONIC/GR00T 权重、SAM2 checkpoint 与 DAAAM 研究资产不进入 Git。

当前模型与外部资产边界：

| 类别 | Current 依赖 | 复现方式 |
| --- | --- | --- |
| 文本 | DeepSeek `deepseek/deepseek-v4-flash` | 操作者提供 API 凭据 |
| 视觉语义 | Qwen `qwen3-vl-plus` | 操作者提供 DashScope 凭据 |
| 分割 | SAM2 锁定 checkpoint | 外置文件 + SHA-256 |
| 运动 | GEAR-SONIC 锁定 bundle | operator-fetch + model lock |
| G1 场景模型 | Unitree G1 MuJoCo checkout + reviewed overlay | 接受上游条款后 operator-fetch |
| DAAAM | 默认关闭的异步 importer | 完整 research stack 不在复现主路径 |

## 5. 静态和组件门禁

```bash
./integration/preflight.sh --level L0
./integration/preflight.sh --level L1
./integration/acceptance.sh --level L1

(cd "$ROOT/components/h-ide/apps/control-pwa" && npm test && npm run build)
(cd "$ROOT/components/h-ide-agent" && npm test && npm run typecheck && npm run build)
```

L1 必须从这个 clean clone 的环境运行。使用另一份源目录中已经构建的镜像、测试缓存或
生成场景，不能作为复现证据。

## 6. 启动、管理员初始化和 L2 readiness

```bash
./integration/preflight.sh --level L2
./integration/start-e2e.sh --rebuild --viewer --full-straight-drive
./integration/acceptance.sh --level L2
```

如果需要首个 ADMIN，在服务已创建 schema 后停止 Control API，并通过一次性 CLI 初始化：

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

密码通过交互式隐藏输入提供；数据库、密码和审计运行状态都不属于发行源码。

## 7. 有行为的端到端验收

通用 L2 health 不会自动改变机器人状态。要证明当前候选的联调效果，操作者必须显式执行
并保存 Mission ID、Robot Agent 回执、Plant/SONIC session、最终 pose 和 EvidenceRef：

1. 单房间：`今天巡检开放办公室`。
2. 双房间：`今天巡检开放办公室和会议室`。
3. 确认每个房间都有进入、内部停稳、多方向观察和报告节点。
4. 同一会话切换语音助手，询问：`刚刚有看到电脑吗？`。
5. 确认历史回答与上轮报告一致，浏览器语音可用，空闲机器人挥手后恢复正常姿态。
6. 确认 full-straight 只在经地图证明的直线段生效，转向/终点仍正确停稳。
7. 冷启动复位后至少再运行一次，排除旧数据库、旧 Mission 或缓存造成的假成功。

当前场景没有走廊中央圆柱，不使用旧的柱子绕行用例。不得用直接 Nav2 action、人工移动
`qpos` 或预填运行数据库替代 PWA 端到端验收。

## 8. 可复现边界

- 供应商可能更新同一云模型 ID、限流、欠费或改变服务可用区；代码无法冻结其在线状态。
- GPU driver、CUDA、ROS deb 和基础容器镜像来自外部供应链；正式 Release 应附镜像 digest、
  主机 manifest、SBOM 与漏洞/许可证报告。
- G1 网格和模型权重受独立条款约束，不能因源码仓库公开就重新分发。
- PWA 账户、会话、Mission 数据库、视觉 Evidence 和运行日志必须在新环境重新创建。
- 当前 Office V2 场景目录与可信站点仍是固定 Demo 回退基线；FARM/DAAAM、World Model、
  Semantic Scene Compiler 和第二场景无硬编码 onboarding 属于 Proposed 里程碑。
- 旧 LeRobot subtree 历史可能包含上游测试二进制路径；当前树门禁不等于完整历史审计，
  正式公开发布前仍要执行 full-history secret/license scan。

只有完成本页的 clean clone、依赖、L0/L1、L2 readiness 和 L3 显式行为验收，才能声称另一名
开发者复现了联调；仅成功启动 PWA 或 MuJoCo viewer 不够。

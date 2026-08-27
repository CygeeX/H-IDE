# 从干净克隆复刻联调

本文把“代码可克隆”和“联调可复现”分开验证。仅在 README 中列出缺失依赖并不
足够；冻结版通过相对路径、机器可读 manifest、哈希验证和分级验收把这些依赖变成
显式输入。

## 1. 验证克隆身份

```bash
git clone http://192.168.50.189:37085/zhongwenjun/skillstore.git
cd skillstore
git status --short
git tag --verify integration/office-v2-e2e-20260827-r1 2>/dev/null || \
  git show --no-patch integration/office-v2-e2e-20260827-r1
./integration/acceptance.sh --level L0
```

预期：工作树为空、无 gitlink/submodule、L0 通过。`third_party/lerobot` 和
`unitree_sdk2_python` 应为普通可读源码；G1 网格和权重应仍为空/外置。

## 2. 创建各组件隔离环境

不同组件的 Python 版本不同，不能共用一个 `.venv`：

```bash
# H-IDE / Mission / Scene（Python 3.12）
python3.12 -m venv components/h-ide/.venv
components/h-ide/.venv/bin/pip install -e 'components/h-ide[test,daaam]'

# MuJoCo / SONIC（Python 3.12）
python3.12 -m venv components/h-ide-mujoco-sonic/.venv
components/h-ide-mujoco-sonic/.venv/bin/pip install \
  -e 'components/h-ide-mujoco-sonic[runtime,test]'

# Vision（项目锁固定 Python 3.11）
(cd components/h-ide-vision && uv sync --locked --extra dev)

# Robot Agent
(cd components/robotagent && uv sync --locked --group dev)

# Planning Agent
(cd components/h-ide-agent && npm ci)
```

Robot Agent 的 ROS 2 interfaces/overlay 仍需按其 README 构建，并使以下文件存在：

```text
components/robotagent/install/setup.bash
components/robotagent/tools/activate-local-nav2-jazzy.bash
components/robotagent/tools/activate-local-realsense-jazzy.bash
```

## 3. 创建秘密和本机依赖

```bash
./integration/bootstrap.sh --install-agent
chmod 600 .env
```

填写 DeepSeek、DashScope、SAM2 路径。内部 token 由 bootstrap 随机生成；同一个
`.env` 通过相对 symlink 给 H-IDE/Compose/host launcher 使用，避免多份 token 漂移。

获取外部资产：

```bash
./integration/bootstrap.sh --fetch-groot-source
SKILLSTORE_ACCEPT_G1_MODEL_TERMS=1 ./integration/bootstrap.sh --fetch-g1-model
./integration/bootstrap.sh --fetch-sonic-model
```

脚本会检出 manifest 中的完整 Git SHA，并对 G1 应用已记录 SHA-256 的 Office
overlay patch。它不会下载任意 `main` 最新版。

## 4. 主机 ROS/Nav2 依赖

完整 L2 以 Ubuntu 24.04 + ROS 2 Jazzy 为基线。至少需要：

- `ros-jazzy-ros-base`、Nav2、`rmw-cyclonedds-cpp`、TF2 和常用 message/action。
- 能运行 host-network 容器的 Docker Engine/Compose v2。
- 当前用户可用的 `systemctl --user`；桌面模式有 X11 session。
- NVIDIA driver/CUDA 与项目依赖兼容，`nvidia-smi` 正常。

具体 ROS package 以 `components/h-ide-mujoco-sonic/docker/nav2/Dockerfile` 和
Robot Agent 的 activation scripts 为准。L1 可以在没有 GPU/ROS 的构建机运行，
但不能据此宣称运动链复现成功。

## 5. 从静态到动态验收

```bash
./integration/preflight.sh --level L0
./integration/preflight.sh --level L1
./integration/acceptance.sh --level L1
./integration/preflight.sh --level L2
./integration/start-e2e.sh --viewer
./integration/acceptance.sh --level L2
```

需要实际运动时，按 `docs/acceptance/README.md` 逐个运行有名称的 case。通用脚本
不会自动猜路线，也不会因“这是仿真”而绕过 Mission/Robot Agent 权限边界。

## 6. 可复现性仍有哪些边界

- API 服务端模型可能由供应商更新、限流或欠费；仓库固定请求模型 ID和参数，
  但不能冻结第三方在线服务的运行状态。
- GPU driver、CUDA、ROS deb 和 Docker base image 由主机/镜像源提供；正式发布应
  额外记录 SBOM 和镜像 registry digest。
- G1 网格和 SONIC 权重受各自条款约束，不能通过公开 Git 镜像分发。
- PWA 账号、会话历史、Mission 数据库和视觉证据属于运行状态，必须由 bootstrap/
  管理命令重新创建，不能从开发者数据库复制。

这些边界会影响复现，因此不能只写一句“请自行安装”。本发行版对它们提供锁定
revision、哈希、明确安装命令和 fail-fast 检查；只有供应商账户/硬件环境需要
开发者自己提供。

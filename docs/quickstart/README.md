# 快速开始：复现冻结版端到端联调

本文是新开发者从零开始复刻当前 H-IDE Office V2 联调的唯一推荐入口。
命令均以仓库根目录为当前目录，不依赖 `/home/phi` 或其他开发者的本机路径。

## 1. 你将得到什么

冻结版链路如下：

```text
PWA → H-IDE Control API/Mission V2 → h-ide-agent（只规划）
                                      ↘ h-ide-vision（Qwen + SAM2）
                         → Robot Agent → Nav2 → ROS 2 bridge
                         → SONIC → MuJoCo Plant（Office V2）
```

LLM 只负责语言意图和计划草案；它没有运动权限。运动命令必须经过 Mission
审核/授权、Robot Agent 能力边界和 Nav2/SONIC 执行链。DAAAM 默认不启动，
只在 Mission 完成后提供区域级语义先验。

## 2. 主机前置条件

- Ubuntu 24.04（Ubuntu 22.04 只支持容器/协议级验证，不宣称完整复现）。
- Python 3.12：`h-ide`、`h-ide-agent` 的相关服务、MuJoCo/SONIC。
- Python 3.11：`h-ide-vision`（项目声明 `<3.12`）。
- Python 3.10 或更高：`robotagent`。
- Node.js `>=22.13.0`、npm、Docker Engine + Compose v2。
- ROS 2 Jazzy、`rmw_cyclonedds_cpp`、Nav2；L2 需要 NVIDIA 驱动/CUDA、
  `nvidia-smi`、`git-lfs` 和 systemd user manager。
- 只有选择 `--viewer` 时才需要可用的 X11 `DISPLAY`/`XAUTHORITY`。

## 3. 克隆和本地配置

```bash
git clone http://192.168.50.189:37085/zhongwenjun/skillstore.git
cd skillstore
git checkout main
./integration/bootstrap.sh
```

`bootstrap.sh` 会创建权限为 `0600` 的根目录 `.env`，生成空缺的内部随机令牌，
并建立 `components/h-ide/.env → ../../.env` 相对链接。它不会覆盖已有凭据，
不会把密钥写入 Git，也不会静默下载大型模型。

编辑 `.env`，至少填写：

```dotenv
DEEPSEEK_API_KEY=<文本规划凭据>
DASHSCOPE_API_KEY=<Qwen/SAM2 服务凭据>
HIDE_VISION_SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_tiny.pt
```

当前冻结版文本模型固定为 `deepseek/deepseek-v4-flash`，视觉模型固定为
`qwen3-vl-plus`；不要把供应商凭据放在 URL、Compose 文件或提交信息中。

## 4. 获取许可受限的外部资产

源码和权重分开处理：

```bash
# NVIDIA GR00T/SONIC 部署源码（官方 profile 的来源证明）
./integration/bootstrap.sh --fetch-groot-source

# G1 MuJoCo 网格和 XML；先阅读 manifests/vendor/g1-model.json 的条款
SKILLSTORE_ACCEPT_G1_MODEL_TERMS=1 \
  ./integration/bootstrap.sh --fetch-g1-model

# 约 865 MB 的 GEAR-SONIC ONNX bundle
./integration/bootstrap.sh --fetch-sonic-model

# 可选：安装规划 Agent 的 Node 依赖
./integration/bootstrap.sh --install-agent
```

默认下载位置是 `${XDG_CACHE_HOME:-$HOME/.cache}/skillstore`。可在 `.env` 中用
`SKILLSTORE_EXTERNAL_ROOT`、`HIDE_G1_MODEL_ROOT`、`SONIC_HF_CACHE` 改变位置。
G1 网格没有在仓库中再分发；SONIC 权重也不进 Git。每个来源、提交、哈希和
许可证状态见 `manifests/vendor/` 与 `integration/integration.lock.yaml`。

SAM2 checkpoint 需要由操作者单独取得；L2 会核验其 SHA-256。当前冻结哈希为：
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。

## 5. 分级预检

```bash
./integration/preflight.sh --level L0   # 不需要模型、ROS 或 Docker
./integration/preflight.sh --level L1   # 需要 Docker/场景构建环境
./integration/preflight.sh --level L2   # 完整主机、外部资产、GPU/ROS
```

预检失败时先修复报告中的第一项，不要用 `--force` 绕过。L0/L1/L2 的定义和
验收证据见 [`../acceptance/README.md`](../acceptance/README.md)。

## 6. 启动完整联调

```bash
./integration/start-e2e.sh --viewer
```

无图形环境使用：

```bash
./integration/start-e2e.sh
```

启动器按以下顺序等待就绪：Compose 控制面 → Qwen/SAM2 → Office V2 场景校验
→ MuJoCo Plant → SONIC 50 Hz motion gate → ROS ingress → map→base_link TF
→ Nav2 lifecycle → strict navigation adapter → Robot Agent/observation gateway。
它不会因为某个旧服务“看起来还活着”就复用半条链路。

启动后：

- PWA：`http://127.0.0.1:8080/`
- 仿真观测：`http://127.0.0.1:8080/simulation`
- 状态汇总：`./components/h-ide-mujoco-sonic/scripts/workstation-demo-status.sh`

## 7. 停止与重新开始

```bash
./integration/stop-e2e.sh
./integration/start-e2e.sh --viewer
```

停止脚本只停止带有 `skillstore-e2e` 项目名的 Compose 服务和冻结脚本创建的
systemd user 单元；不会删除数据库卷、日志、模型或外部 checkout。需要查看
失败原因时，先看 `XDG_RUNTIME_DIR/h-ide-mujoco-demo-$(id -u)/*.log`，再看
`docker compose ... logs`。

## 8. 最小 PWA 验收顺序

1. 登录并确认机器人状态为 `g1-sim`、场景为已发布的 Office V2。
2. 先发送纯观察请求，确认视觉凭证和对象事实来自 Qwen/SAM2。
3. 发送一个短距离 `走到开放办公室门口`，审核 Mission 草案后执行。
4. 再执行原地转向、横移和静态柱子绕行；每一步等待终态对账。
5. 最后执行办公室巡检，确认 PWA 收到所有房间/路上观察信息及 Mission 终态。

不要用“直接调用 Nav2 action”替代 PWA 验收；那会跳过本项目要验证的授权、
视觉 grounding、流式状态和 Robot Agent 边界。

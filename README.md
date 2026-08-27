# Skillstore · H-IDE 端到端联调发行仓库

这是 H-IDE Assistant V2、Mission V2、Vision、Robot Agent、Nav2、SONIC 和
MuJoCo Office V2 的单仓库联调发行版。它把各组件的冻结源码、跨项目版本锁、
配置模板、启动脚本和验收门禁放在同一个 Git 版本中。

> 当前 `main` 对应已验证的联调冻结版。真实 API key、数据库、日志、视觉证据、
> 模型权重和本机缓存永远不提交到 Git。

## 快速入口

1. 阅读 [`docs/quickstart/README.md`](docs/quickstart/README.md)。
2. 运行 `integration/preflight.sh`，确认主机和外部资产满足前置条件。
3. 复制 `integration/.env.simulation.example` 为本地秘密文件并填入供应商凭据。
4. 运行 `integration/bootstrap.sh` 准备依赖和生成场景产物。
5. 运行 `integration/start-e2e.sh --viewer` 启动完整联调。
6. 运行 `integration/acceptance.sh --level L1` 或配置 GPU/模型后运行 `--level L2`。

```bash
git clone http://192.168.50.189:37085/zhongwenjun/skillstore.git
cd skillstore
./integration/bootstrap.sh
# 编辑 .env；不要把它加入 Git
./integration/preflight.sh --level L0
./integration/start-e2e.sh --viewer
```

HTTP 地址是本项目约定的内网入口；Git 写入凭据必须通过本机 credential
helper 提供，不能写进 URL 或脚本。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| `components/h-ide` | Control API、PWA、Scene API 和 Mission 服务 |
| `components/h-ide-agent` | Assistant/Mission planning-only Agent |
| `components/h-ide-vision` | Qwen3-VL-Plus + SAM2 视觉服务 |
| `components/robotagent` | Robot Agent 与观察网关 |
| `components/h-ide-mujoco-sonic` | MuJoCo Plant、SONIC、ROS 桥和 Nav2 适配 |
| `optional/DAAAM` | 异步语义记忆集成说明与上游 provenance |
| `third_party` | 经过许可审查的依赖补丁/来源说明 |
| `integration` | 统一配置、bootstrap、启动、检查和验收入口 |
| `manifests` | 组件、场景、模型、镜像、SBOM 和许可证锁 |
| `docs` | 架构、组件、运维、复刻和验收文档 |

## 重要边界

- LLM 只能提供语义分类；它不能直接派发运动或提供可执行坐标。
- 场景可信站点来自已发布的 Office V2 Scene；对象接近点必须由新鲜
  Qwen/SAM2 + RGB-D/TF 几何生成并验证。
- Nav2 负责路径，Robot Agent/SONIC/MuJoCo 负责执行和反馈。
- DAAAM 是 post-Mission、observation-only 的区域级先验，不是实时运动服务。
- 每个物理 Mission 都必须经过草案审核、授权、执行和终态对账。

## 文档索引

- [快速开始与账号初始化](docs/quickstart/README.md)
- [组件与调用链](docs/architecture/README.md)
- [各组件开发说明](docs/components/README.md)
- [运行、停止、复位和监控](docs/operations/README.md)
- [从干净克隆复刻](docs/reproduction/README.md)
- [L0/L1/L2 验收清单](docs/acceptance/README.md)
- [秘密、许可证和第三方边界](docs/security/README.md)
- [跨项目锁定清单](integration/integration.lock.yaml)

发布边界：代码仓库是公开可读的，但 G1 网格、SONIC/GR00T 权重、API 凭据、账号
数据库和运行证据仍按各自条款与本机权限处理，不会因为 Git 可读就自动随仓库分发。

# 组件开发者索引

## `components/h-ide`

控制面、PWA、Mission V2、Scene API 和开发者诊断。关键入口：

- `services/control_api/`：认证、会话、Mission、事件和 Robot Agent BFF。
- `services/scene_api/`：Office V2 发布场景、绑定和资产元数据。
- `infra/docker-compose.yml`：控制面 Compose 拓扑。
- `tests/test_end_to_end.py`、`tests/test_visual_mission_runtime.py`：控制面合同。

本组件永远不应把浏览器直接暴露给 ROS、DDS、ZMQ 或机器人端口。运行时数据库和
视觉证据在 `.env` 指定的本地/Compose volume 中，不属于 Git 发行物。

## `components/h-ide-agent`

独立 Node/Mastra 规划服务。关键入口：

- `src/http/routes.ts`：内部 planning/conversation API。
- `src/contracts.ts`：PlanProposal、PlanFragment、Conversation V2 schema。
- `src/agents/`：Supervisor、motion/vision worker 和 structuring 边界。

它只拥有 `MASTRA_INTERNAL_TOKEN` 和模型供应商凭据，不拥有 ROS/Robot Agent 凭据。
`npm run typecheck && npm test && npm run build` 是提交前最低门禁。

## `components/h-ide-vision`

独立 Python 3.11 host-GPU 服务。Qwen3-VL-Plus 是语义 authority，SAM2 只负责
分割/几何支持。关键入口：

- `src/h_ide_vision/`：HTTP/gRPC ingress、provider boundary、geometry。
- `tools/observation_fixture_sender.py`：协议 fixture，不是生产语义 oracle。
- `tests/contracts/`、`tests/unit/`：协议和 provider contract。

Vision 不发 Nav2 命令；失败时 H-IDE 必须显示“无法解析/需澄清”，不能静默变成
当前画面观察或可信站点。

## `components/robotagent`

机器人侧协议、能力/租约、观察网关和 ROS 适配。关键入口：

- `src/robot_agent/`：命令接收、能力校验、回执和取消。
- `src/robot_observation_gateway/`：RGB-D/TF 快照和 freshness/identity gate。
- `tools/activate-local-*-jazzy.bash`：本机 ROS overlay 激活。

Robot Agent 是物理执行边界；任何新能力都必须同时更新 capability manifest、
schema、测试和 integration lock。

## `components/h-ide-mujoco-sonic`

MuJoCo Office V2、SONIC driver、ROS bridge、Nav2 strict adapter 和 diagnostics。
关键入口：

- `src/hide_mujoco_sonic/cli.py`：Plant/SONIC/bridge 命令。
- `config/*.yaml`：相对路径 runtime selector。
- `config/sonic-models.lock.json` + `tools/verify_model_lock.py`：模型/源完整性。
- `tools/build_scene.py`：单源场景产物构建/检查。
- `tools/strict_navigation_adapter.py`：Nav2 action 到已验收原语的适配。
- `scripts/start-workstation-demo.sh`：有序启动和 ready gates。

SONIC 的控制历史、参考 epoch、idle/safe-stop 和 Plant clock 是一个整体；不要
单独替换某个“看起来更安全”的阈值而跳过 `tests/test_gait_runtime_integration.py`
与 motion acceptance。

## `third_party`

`third_party/lerobot` 和 `third_party/unitree_sdk2_python` 是有 provenance 的源快照，
不是可随意升级的 vendor 目录。升级时必须：更新 manifest、内容哈希、锁文件、
许可证记录、回归测试，并在临时 clean clone 验证。

## `optional/DAAAM`

这里只记录上游提交和 H-IDE importer 合同，不复制研究环境、受限数据集或权重。
要改动 importer，先阅读组件内 `docs/integration/daaam-semantic-memory.md`，再运行
DAAAM 相关单元测试；不得把它加入实时运动依赖图。

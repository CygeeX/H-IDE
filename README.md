# Robot Action Console

当前主任务流为 [Mission V2 连续 Supervisor–Worker 架构](docs/architecture/mission-v2-continuous-supervisor-worker.md)。真实 Robot Agent / D435i / Mock Nav2 的整链步骤见 [Mission V2 联调与验收清单](docs/integration/mission-v2-real-validation.md)。

Robot Action Console 是工作站上的机器人行动控制平面。它不是机器人侧
Robot Agent，也不是 Agent 依赖框架；它把手动控制、已确认的 Agent 计划和
接口映射管理统一收敛到一个可审计的 `ActionRuntime`。

```text
React PWA (浏览器/移动端)
          │ HTTPS + WebSocket（同源 gateway）
          ▼
Caddy ─────┬── Control API ── SQLite repository/event log
           │       ├── Mastra Supervisor–Worker（planning only）
           │       ├── whisper.cpp（optional speech-to-text）
           │       └── ActionRuntime ── Robot Agent / HTTP / MQTT / ROS 2
           └── Scene API/assets ── Scene Worker
                                    PCL → PDAL → Open3D + PotreeConverter
```

## 能力边界

- 手动按钮和 Agent 确认后的多步骤计划都通过同一个 `ActionRuntime`，带幂等
  键、有效期、机器人租约、回执和事件时间线。
- Agent 只能创建 Mission `draft`；只有用户明确确认才会发送技能。某一步返回
  `accepted/running` 时，Mission 使用同一命令 ID 对账并持续推进，不会重复
  dispatch；进程重启后无法确认的物理状态会显式挂起，绝不自动重放。
- 命令到机器人侧接口的关系由不可变的 `MappingVersion` 描述。流程是
  `draft → validate → probe → activate`，激活使用 SQLite 指针 CAS；可以通过
  Web/API 切换 v1/v2 并回滚，不需要修改 PWA 代码。并发的旧草稿/探测会收到
  `409`，同一版本的顺序能力重探测使用探测快照 CAS。
- 场景、视频、语音和规划服务不在运动安全路径中。它们故障时，
  `stop` 与 `cancel` 仍由控制 API 直接处理。
- PCD 支持断点分片上传、自动清理/降采样、二维占用图/高度图、Potree 三维浏览、
  identity/similarity 两种 `scene → map` 标定、语义区域和安全入口标注。标定前
  生成未验证预览，标定后生成与 revision 绑定的正式地图；只有已发布且绑定到当前
  机器人的紧凑地图摘要会进入 Agent，原始点云和资产 URL 不进入模型。
- Mission Executive 只选择语义地点、视觉目标、观察或已发布技能；Qwen 坐标只
  能作为假设。PCD 场景检查或 RGB-D + TF 确定性工具生成一个可信 `map` 目标，
  H-IDE 在签发前复核授权边界。工作站不规划路径，避障与可达性属于 Robot
  Agent/Nav2；包括原地改变朝向在内的底盘运动都必须经过 Nav2。
- 物理急停、机器人本地安全监控和现场人员始终优先；远程按钮不能解除物理
  急停。

当前 runnable profile 使用 SQLite 和 FakeAdapter，真实 Robot Agent、ROS 2、
HTTP、MQTT driver 是显式适配器边界。SQLite 可替换为 PostgreSQL repository，
Redis 用于未来事件扇出，但二者目前不是命令执行的第二条路径。

## 本地运行

需要 Python 3.12、Node.js 20+（构建 PWA）和仓库内的 `.venv`。第一次使用可执行：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

启动 API（开发默认使用内存库）：

```bash
CONTROL_API_TOKEN=dev-token \
CONTROL_DB_PATH=./data/control.db \
.venv/bin/uvicorn services.control_api.main:app --host 127.0.0.1 --port 8080
```

启动 PWA 开发服务器：

```bash
cd apps/control-pwa
npm ci
npm run dev
```

开发服务器会把同源 `/api`、事件 WebSocket 和 `/scene-assets` 代理到
`VITE_DEV_PROXY_TARGET`（默认 `http://127.0.0.1:8080`），因此 5173 页面无需
CORS 配置。若开发时 Control API 和 Scene API 分别运行（例如 18080/8090），再设置
服务端专用的 `HIDE_DEV_SCENE_API_TARGET` 和 `HIDE_DEV_SCENE_API_TOKEN`；Vite 会按
与 Caddy 相同的路径规则分流，浏览器仍只持有 Control API token。完整 PCD 功能需要
可用的 Scene API/Worker；只启动 Control API 时其余控制界面仍可使用。也可以设置 `VITE_CONTROL_API_URL` 和
`VITE_CONTROL_API_TOKEN`。生产构建完成后，如果
`apps/control-pwa/dist` 存在，控制 API 会挂载静态 bundle。

当前 Supervisor–Worker、MissionEnvelope、视觉事实、语音审核与外部技能接入见
[`docs/architecture/mission-v2-continuous-supervisor-worker.md`](docs/architecture/mission-v2-continuous-supervisor-worker.md)。
旧 V1 Assistant 的迁移背景仍记录在
[`docs/architecture/assistant-supervisor-worker.md`](docs/architecture/assistant-supervisor-worker.md)。
独立 Mastra 服务位于相邻的 `/home/phi/learn-projects/h-ide-agent`。PCD 场景的
导入、标定、标注和空间规划闭环见
[`docs/architecture/pcd-scene-spatial-agent.md`](docs/architecture/pcd-scene-spatial-agent.md)。

## API 快速示例

所有 REST 请求都需要 `Authorization: Bearer $CONTROL_API_TOKEN`；事件 WebSocket
可使用 `?token=...&after=<eventId>`。交互式 OpenAPI、Swagger 和 ReDoc 默认关闭。

```bash
curl -H 'Authorization: Bearer dev-token' http://127.0.0.1:8080/api/v1/robots

curl -X POST http://127.0.0.1:8080/api/v1/commands \
  -H 'Authorization: Bearer dev-token' \
  -H 'Content-Type: application/json' \
  -d '{"robotId":"robot-1","commandKey":"move.forward",
       "commandVersion":"1","transport":"fake",
       "payload":{},"idempotencyKey":"demo-1"}'
```

常用接口：

| 用途 | 路径 |
|---|---|
| Mission 草案/审核 | `POST /api/v2/missions/drafts`、`POST .../drafts/{draftId}/{approve\|reject}` |
| Mission 持续执行/取消 | `POST /api/v2/missions/{missionId}/advance`、`POST .../{missionId}/cancel` |
| Mission 补充/修订 | `POST /api/v2/missions/{missionId}/messages`、`POST .../revisions/{revisionId}/approve` |
| 机器人/能力 | `GET /api/v1/robots`、`GET /api/v1/robots/{robotId}/capabilities` |
| 手动命令 | `POST /api/v1/commands` |
| 查询/取消 | `GET /api/v1/commands/{commandId}`、`POST /api/v1/commands/{commandId}/cancel` |
| Agent 对话/任务 | `POST /api/v1/assistant/conversations`、`POST .../{conversationId}/messages` |
| Agent 轮次/审核 | `GET /api/v1/assistant/turns/{taskId}`、`POST /api/v1/assistant/drafts/{draftId}/{confirm\|revise\|reject}` |
| 语音转写 | `POST /api/v1/speech/transcriptions` |
| 场景导入/转换 | `POST /api/v1/scene-uploads`、`PATCH .../{uploadId}`、`GET /api/v1/scene-jobs/{jobId}` |
| 场景重新处理 | `POST /api/v1/scenes/{sceneId}/reprocess`（需要保留原始 PCD） |
| 场景版本/绑定 | `GET /api/v1/scenes`、`PUT .../calibration`、`PUT .../annotations`、`POST .../publish`、`PUT /api/v1/robots/{robotId}/scene-binding` |
| 映射生命周期 | `POST /api/v1/mappings/drafts`、`.../{mappingId}/{version}/{validate|probe|activate}` |
| 回滚/事件 | `POST /api/v1/mappings/{commandKey}/rollback`、`GET /api/v1/events` |

## 测试与构建

```bash
PYTHONWARNINGS=error PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -rA
.venv/bin/python -m compileall -q packages services tests
git diff --check

cd apps/control-pwa
npm ci
npm test -- --run
npm run build
```

端到端验收在 [`tests/test_end_to_end.py`](tests/test_end_to_end.py)，弱网和急停
运行约束在 [`docs/operations/safety-and-weak-network.md`](docs/operations/safety-and-weak-network.md)。

## 部署

- Docker：复制并填写 `.env.example`，执行
  `docker compose --env-file .env -f infra/docker-compose.yml up --build`；可选的
  本地语音服务使用 `--profile speech`。
  `/data/control.db` 由 `control-data` volume 持久化；PostgreSQL/Redis 服务是
  后续 repository/event-fanout 的迁移边界，`planning-data` 保存 Mastra 任务和
  memory，`scene-data` 保存原始 PCD、转换产物和场景元数据。浏览器始终通过
  `CONTROL_API_PORT` 的同源网关访问；Scene API 的 `SCENE_API_PORT` 只绑定回环，
  供可选的 host-network ROS 2 profile 使用。
- systemd：安装 [`infra/systemd/robot-action-console.service`](infra/systemd/robot-action-console.service)
  并在 EnvironmentFile 中设置 token、数据库路径和适配器端点。
- ROS 2：仅在确实需要访问主机 DDS 图时启用 compose 的 `ros2-host` profile；浏览器
  不接触 ROS 话题、DDS、ZMQ 或机器人端口。

生产环境至少应使用 HTTPS/WireGuard、长随机 Bearer token、控制库与场景库备份。
手机安装 PWA、浏览器录音以及浏览器端 PCD 校验依赖安全上下文，因此 LAN 真机
演示也应提供受信任的 HTTPS。部署前请按安全文档验证断网、重启、租约过期和
物理急停流程。

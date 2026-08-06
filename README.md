# Robot Action Console

Robot Action Console 是工作站上的机器人行动控制平面。它不是机器人侧
Robot Agent，也不是 Agent 依赖框架；它把手动控制、已确认的 Agent 计划和
接口映射管理统一收敛到一个可审计的 `ActionRuntime`。

```text
React PWA (浏览器/移动端)
          │ HTTPS + WebSocket（Bearer token）
          ▼
FastAPI BFF ── SQLite repository/event log
          │ immutable CommandRequest + MappingVersion
          ▼
ActionRuntime ── adapter registry ── Robot Agent / HTTP / MQTT / ROS 2

scene-api + scene-worker       video-gateway (ZMQ → WebRTC)
        （隔离、可失败）                （隔离、可失败）
```

## 能力边界

- 手动按钮和 Agent 确认后的多步骤计划都通过同一个 `ActionRuntime`，带幂等
  键、有效期、机器人租约、回执和事件时间线。
- Agent 只能创建 `draft`；只有用户明确确认才会发送命令。某一步返回
  `accepted/running` 等非终态时，计划变为 `unknown`，不会自动继续或重放。
- 命令到机器人侧接口的关系由不可变的 `MappingVersion` 描述。流程是
  `draft → validate → probe → activate`，激活使用 SQLite 指针 CAS；可以通过
  Web/API 切换 v1/v2 并回滚，不需要修改 PWA 代码。并发的旧草稿/探测会收到
  `409`，同一版本的顺序能力重探测使用探测快照 CAS。
- 场景、视频和未来的语音/规划服务不在运动安全路径中。它们故障时，
  `stop` 与 `cancel` 仍由控制 API 直接处理。
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

PWA 默认请求同源 `/api/v1`，也可以设置 `VITE_CONTROL_API_URL` 和
`VITE_CONTROL_API_TOKEN`。生产构建完成后，如果
`apps/control-pwa/dist` 存在，控制 API 会挂载静态 bundle。

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
| 机器人/能力 | `GET /api/v1/robots`、`GET /api/v1/robots/{robotId}/capabilities` |
| 手动命令 | `POST /api/v1/commands` |
| 查询/取消 | `GET /api/v1/commands/{commandId}`、`POST /api/v1/commands/{commandId}/cancel` |
| Agent 草案 | `POST /api/v1/assistant/drafts`、`POST /api/v1/assistant/drafts/{draftId}/confirm` |
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

- Docker：`docker compose -f infra/docker-compose.yml up --build control-api`。
  `/data/control.db` 由 `control-data` volume 持久化；PostgreSQL/Redis 服务是
  后续 repository/event-fanout 的迁移边界。
- systemd：安装 [`infra/systemd/robot-action-console.service`](infra/systemd/robot-action-console.service)
  并在 EnvironmentFile 中设置 token、数据库路径和适配器端点。
- ROS 2：仅在确实需要访问主机 DDS 图时启用 compose 的 `ros2-host` profile；浏览器
  不接触 ROS 话题、DDS、ZMQ 或机器人端口。

生产环境至少应使用 HTTPS/WireGuard、长随机 Bearer token、文件库备份和受限的
`CONTROL_DB_PATH`。部署前请按安全文档验证断网、重启、租约过期和物理急停流程。

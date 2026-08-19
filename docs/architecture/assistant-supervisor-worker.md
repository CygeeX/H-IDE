# H-IDE 内置调度 Agent 与 Supervisor–Worker 集成

## 已实现的边界

本次实现把“内置 Agent”定义为工作站上的规划调度能力，而不是机器人控制器。
H-IDE 仍是对话、机器人上下文、人工审核和执行的唯一控制平面；独立的
[`h-ide-agent`](../../../h-ide-agent) 使用 Mastra 完成 Supervisor–Worker 规划。

```text
PCD 导入/地图选点    文本输入          浏览器语音
Scene API/Worker       │          16 kHz mono WAV
      │ sceneRef       │                 │
      └───────────────┴──────► H-IDE Control API
                                  │ 语音先经本地 whisper.cpp，
                                  │ 转写文本由用户编辑/明确提交
                                  ▼
                      TaskRequest + frozen context
                  robot / capability / catalog / scene
                                  │ internal bearer HTTP
                                  ▼
                     Mastra Supervisor（只做路由）
                         ┌────────┼────────┐
                         ▼        ▼        ▼
                       Motion  Manipulation  外部 A2A Worker
                         └────────┼────────┘
                                  ▼
                         Structuring Agent
                                  ▼
                H-IDE 二次校验并保存不可变 PlanDraft
                                  ▼
                    人工确认 / 修改 / 拒绝 / 取消
                                  │ 只有“确认”进入执行边界
                                  ▼
                 ActionRuntime → Adapter → Robot Agent/ROS 2
```

规划服务没有 `ActionRuntime`、ROS 2、Robot Agent、HTTP/MQTT adapter 或凭据，
因此不能绕过人工审核执行动作。手动控制和 `stop` 也不经过 Mastra 或 Whisper；
规划服务失效时它们仍可独立工作。

## 三种输入如何统一

### PCD

PCD 已实现为独立的异步场景流水线：浏览器断点分片上传，Scene Worker 用固定的
PCL → PDAL → Open3D/PotreeConverter 链路生成清理后的分级点云、未验证二维预览和
标定后的正式占用/高度地图。PWA 负责二维/三维查看、两种 map 标定、语义区域/
安全入口和单点选择。场景版本发布后不可变，并且必须显式绑定到当前机器人。

H-IDE 只把绑定 revision 的 `sceneRef`、map 边界、语义区域、正式地图摘要、当前
机器人位姿和可选点加入 `TaskRequest.sceneContext`。PCD 二进制、栅格、转换资产、
服务器路径和标定矩阵均不进入规划 envelope。完整流程见
[`pcd-scene-spatial-agent.md`](pcd-scene-spatial-agent.md)。

### 文本

用户输入自然语言并点击“发送给调度 Agent”。H-IDE 创建或复用一个机器人对话，
生成统一 `TaskRequest`，冻结提交时的状态和能力后异步调用规划服务。

### 语音

浏览器录音后在本地重采样为 16 kHz、单声道、16-bit PCM WAV；H-IDE 对格式、
10 MiB 大小和 60 秒时长做校验，再调用工作站内的 whisper.cpp `/inference`。
转写结果只回填文本框，不会自动触发规划。用户可以修改文字并明确提交，此时
`modality=voice` 且携带 `transcriptionId`，之后与文本走同一条链路。

## TaskRequest 与冻结上下文

H-IDE 是协议主方。每个规划轮次发送一个严格、拒绝额外字段的 envelope：

- `TaskRequest`：`taskId`、`conversationId`、`robotId`、操作者、`text|voice`、
  文本、locale、可选转写 ID 和修订来源；
- 最近 20 条规范化对话历史；
- 当前 `RobotState` 和 `CapabilitySnapshot`；
- 只包含高层逻辑命令的 `commandCatalog`；
- 可选的已发布、已绑定 `SceneContext`；
- 对以上机器人与场景上下文计算的 `contextVersion`。

映射 ID/版本、ROS topic/service/action 名、机器人代理地址和凭据不会进入规划
envelope。完整 Python 契约在 `packages/control_core/assistant_models.py`，对应的
Zod 契约在独立服务的 `src/contracts.ts`。

## Supervisor–Worker 规划规则

Mastra Supervisor 对一个轮次只能返回三类决策：

1. `route`：选择一个或多个专属 Worker；
2. `needs_input`：信息不足，返回一条追问；
3. `cannot_plan`：命令目录或机器人能力不足，列出缺失能力。

当前本地 Worker 有移动和操作两类；外部专属 Agent 通过 Mastra A2A 注册。选中的
Worker 并行生成 `PlanFragment`，结构化 Agent 再把所有 fragment 合为一个
`PlanProposal`。系统实行全有或全无：未知 Worker、超时、非法 JSON、Agent ID
冒用、漏掉任何贡献、未知命令、缺少能力、上下文版本不一致或有环依赖，都会让
轮次失败，绝不降级为一个看似完整的部分计划。

移动 Worker 的空间任务还有一层约束：模型判断静态场景是否适合辅助任务，并只能
选择 `relative`、`semantic` 或 `selected_point` 意图；运行时再调用
`resolve_spatial_target` 确定性工具，将这个意图转换为可审核候选坐标。模型不自行
计算坐标，工具不替模型决策，也不把 PCD 当作实时占用、可达性或安全结论；工作站
不生成路径。

Mastra memory 按 `conversationId` 隔离，用于模型连续性；H-IDE SQLite 中的消息
历史仍是权威记录，每次请求都会重新发送。两个数据库不能替代彼此。

## 从 PlanProposal 到执行

H-IDE 收到结果后会用自己的 Pydantic/JSON Schema 再验证一次，并把 proposal
拆成两部分持久化：

- `PlanDraft`：只含 ActionRuntime 需要的 `stepId`、逻辑 `commandKey`、payload、
  依赖和五分钟有效期；
- review metadata：摘要、理由、假设、警告、每个 Worker 的贡献和
  `contributedBy`，供人工审核。

用户可以：

- 确认：草案经 CAS 从 `draft` 进入 `confirming`，然后才逐步交给
  `ActionRuntime`；
- 修改：原草案保留，反馈形成带 `revisionOfDraftId` 的新规划轮次；
- 拒绝：记录原因并把草案置为 `rejected`；
- 取消：中止仍在排队/运行的规划轮次。

确认前会先解析全部逻辑命令、活动映射、机器人目标绑定和安全状态，防止执行到
中途才发现未知接口；还会重读当前能力快照，任何已撤销的命令都会在零下发状态下
拒绝确认。确认采用每步稳定幂等键；重启时正在规划或确认的工作会置为 `unknown`，
不会静默重放。

空间命令还会在草案创建时和确认时重新执行确定性解析。确认会重读当前绑定 revision
和机器人位姿；场景改变或相对规划时的位姿漂移超过配置阈值时，在任何 adapter
dispatch 之前返回冲突，要求用户重新规划。

## API 与状态

浏览器只访问 H-IDE `/api/v1`：

| 用途 | API |
|---|---|
| 对话 | `POST/GET /assistant/conversations` |
| 对话详情 | `GET /assistant/conversations/{conversationId}` |
| 提交文本/语音任务 | `POST /assistant/conversations/{conversationId}/messages` |
| 查询/取消规划 | `GET /assistant/turns/{taskId}`、`POST .../cancel` |
| 审核 | `POST /assistant/drafts/{draftId}/{confirm|revise|reject}` |
| 转写 | `POST /speech/transcriptions`（multipart WAV） |

H-IDE 与 Mastra 之间只使用 `/internal/v1/planning-turns`，两边共享独立的
`MASTRA_INTERNAL_TOKEN`。Mastra 生成的 Agent、memory 和 Studio/API 路由也受同一
token 保护，只有 `/healthz` 和 `/readyz` 可匿名访问；该端口不得转发给浏览器。

主要轮次状态是 `queued → running → needs_input|cannot_plan|draft_ready|failed`；
草案状态是 `draft → confirming → confirmed|failed|partial|unknown`，或由审核进入
`revision_requested|superseded|rejected`。前端对规划使用独立 busy 状态，不会
因为模型调用锁住手动控制和停止。

## 运行配置

开发时分别启动：

```bash
# /home/phi/learn-projects/h-ide-agent
npm ci
MASTRA_INTERNAL_TOKEN=... SUPERVISOR_MODEL=openai/gpt-5-mini \
STRUCTURING_MODEL=openai/gpt-5-mini OPENAI_API_KEY=... npm run dev

# /home/phi/learn-projects/h-ide
CONTROL_API_TOKEN=... CONTROL_DB_PATH=./data/control.db \
MASTRA_PLANNING_URL=http://127.0.0.1:4111 MASTRA_INTERNAL_TOKEN=... \
WHISPER_ENDPOINT=http://127.0.0.1:8081/inference \
.venv/bin/uvicorn services.control_api.main:app --host 127.0.0.1 --port 8080
```

Docker Compose 已包含 `planning-agent`，Whisper 使用可选 `speech` profile。模型
文件不会写入 Git。使用默认 `ggml-small.bin` 时，可在 H-IDE 仓库根目录准备：

```bash
mkdir -p models
curl --fail --location \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin \
  --output models/ggml-small.bin
```

也可以自行下载其他 whisper.cpp `ggml` 模型，并同步修改
`WHISPER_MODEL_DIR/WHISPER_MODEL_NAME`。模型就绪后运行：

```bash
docker compose --env-file .env -f infra/docker-compose.yml up --build
docker compose --env-file .env -f infra/docker-compose.yml --profile speech up --build
```

配置模板见根目录 `.env.example` 和独立项目的 `.env.example`。生产环境应使用
不同于浏览器 API token、至少 32 字符的长随机内部 token（可用
`openssl rand -hex 32` 生成），仅绑定回环/内部网络，并为外部 A2A origin 配置
显式 allowlist。

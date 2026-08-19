# H-IDE Agent

`h-ide-agent` 是独立于 H-IDE 的 Mastra 规划服务。它实现
Supervisor–Worker 编排，但没有任何机器人执行工具：输入是 H-IDE 冻结后的
机器人状态、能力快照和逻辑命令目录，输出只可能是追问、无法规划，或一个待人
工审核的 `PlanProposal`。

Assistant V2 另有一个前置对话路由边界。它先区分普通聊天、通用问答、纯视觉
观察、机器人行动、场景教学和开发者管理请求。普通聊天/问答会直接自然回复，
绝不创建计划；涉及机器人时只返回口语化过渡和是否应进入 Mission/PlanDraft，
本服务仍没有执行权限。明显问候和有界四则运算走本地快速路径，其余请求交给
同一个 Mastra/Qwen provider，并以低温、关闭 thinking 的结构化输出分类。

```text
H-IDE TaskRequest + frozen context
                │ internal bearer HTTP
                ▼
Mastra Supervisor ──► 1..N local / A2A specialist workers
                │
                ▼
          Structuring Agent
                │ strict Zod validation
                ▼
           PlanProposal only
```

Supervisor 负责选择需要的专属 Agent；当前内置 `motion-worker`、
`manipulation-worker` 和 `vision-worker`。所有被选择的动作 Worker 必须成功返回合法 `PlanFragment`，
最终合并结果必须包含每个贡献、使用同一 `robotId/contextVersion`、形成无环依赖
图，并且只能引用 H-IDE 提供的命令和能力。任一条件不满足，整个规划轮次失败，
不会产生可确认草案。

空间移动任务使用内置的 Mastra `resolve_spatial_target` 工具。H-IDE 把当前已发布、
已绑定场景的紧凑 `SceneContext` 放入本轮 `RequestContext`；`motion-worker` 只选择
相对位移、语义位置或浏览器选点意图，由纯函数计算 ROS `map` 目标，并原样返回
`navigation.navigate_to_pose` payload。模型看不到原始 PCD 或 Potree 资产，不自行
计算坐标，也不规划路径。H-IDE 会在生成草案和用户确认时用 canonical 场景事实再
校验，因此该工具不是执行权限。

`vision-worker` 用于“走到当前摄像头看到的黄色箱子旁”等视觉目标任务。它只让
语言模型抽取目标描述、目标类别和用途，随后由确定性代码写入机器人启动 ID、
定位 epoch、传感器、距离/速度上限以及 VLM/算法/ApproachPolicy 版本，输出
`VisualMissionSpec` 供人工审核。审核后的每个阶段由
`POST /internal/v1/visual-stages/resolve` 调用独立 `h-ide-vision`，只向 H-IDE 返回
不可变 `visualResolutionId` 和歧义说明；坐标由控制面重新读取并校验。此服务没有
路径规划、速度控制或 ROS/机器人凭据。

为了避免 Supervisor 在没有 PCD 绑定时误向用户索要坐标，服务会对一个很窄的 V1
意图做确定性预路由：明确要求从当前画面寻找/识别/靠近单个静态物体时直接选择
`vision-worker`。涉及厨房等命名场所、明确坐标、抓取、探索或跟随目标的请求仍交给
Supervisor 判断，不会被该规则截获。

## 本地运行

需要 Node.js 22.13+：

```bash
npm ci
cp .env.example .env
# 填写模型供应商密钥以及 MASTRA_INTERNAL_TOKEN
npm run dev
```

普通阿里云百炼中国区示例：三个模型变量均填写
`alibaba-cn/qwen3.8-max`，凭证填写到 `DASHSCOPE_API_KEY`。该模型只负责文本调度和
结构化规划；视觉服务仍单独使用 `qwen3-vl-plus` 理解 RGB 图像。

生产构建和校验：

```bash
npm run typecheck
npm test
npm run build
node .mastra/output/index.mjs
```

`GET /healthz` 只表示进程存活；`GET /readyz` 会在模型变量、对应供应商凭据或
内部 token 缺失、少于 32 字符、仍为占位值时返回 `503`。规划任务和 Mastra memory 分别持久化到
`PLANNING_DATABASE_URL` 与 `MASTRA_MEMORY_URL`。服务重启时未完成任务会被标记
为 `unknown`，不会自动重放。

## H-IDE 内部协议

除 `/healthz`、`/readyz` 外，规划接口以及 Mastra 自动生成的 Agent、memory、
Studio/API 路由都需要 `Authorization: Bearer $MASTRA_INTERNAL_TOKEN`：

| 用途 | 接口 |
|---|---|
| 创建或幂等重取轮次 | `POST /internal/v1/planning-turns` |
| 查询状态/增量事件 | `GET /internal/v1/planning-turns/{taskId}?afterEvent=0` |
| 取消轮次 | `POST /internal/v1/planning-turns/{taskId}/cancel` |
| 解析已审核视觉阶段 | `POST /internal/v1/visual-stages/resolve` |
| Assistant V2 对话路由 | `POST /internal/v2/conversation-turns` |

同一 `taskId` 和完全相同的 envelope 会返回原任务；同一 ID 携带不同内容会
返回 `409`。输入输出契约位于 `src/contracts.ts`，路由位于
`src/http/routes.ts`。

对话路由的完整请求/响应契约、意图边界和调用示例见
[`docs/conversation-v2.md`](docs/conversation-v2.md)。它复用
`SUPERVISOR_MODEL` 及相应供应商凭据；不需要额外模型环境变量。调用方必须只在
`shouldCreateMission=true` 时进入既有 Mission V2/PlanDraft 流程，不能把本接口的
文字回复当作执行授权。

## 接入外部专属 Agent

外部 Agent 使用 Mastra A2A 客户端接入。配置示例：

```dotenv
A2A_ALLOWED_ORIGINS=https://inspection.internal,https://navigation.internal
A2A_WORKERS_JSON=[{"id":"inspection-worker","name":"Inspection","description":"Plans inspection tasks","url":"https://inspection.internal/a2a/inspection","token":"worker-secret","timeoutMs":45000}]
```

外部 Worker 必须返回一个 JSON `PlanFragment`，其中 `agentId` 与注册 ID 一致，
且只能引用 envelope 内的逻辑命令。来源必须在 `A2A_ALLOWED_ORIGINS` 中；除
`localhost/127.0.0.1` 外强制 HTTPS；agent card 指向其他 origin 或任何 HTTP
重定向都会被拒绝；不自动重试非幂等的远端调用。建议为不同 Worker 使用不同
短期 token，并在网络层只允许规划服务访问。

## 部署

构建镜像：

```bash
docker build -t h-ide-agent:local .
docker run --rm --env-file .env -p 127.0.0.1:4111:4111 \
  -v h-ide-agent-data:/app/data h-ide-agent:local
```

systemd 模板位于 `infra/systemd/h-ide-agent.service`。生产环境应只在回环地址或
工作站内部网络监听；浏览器、Robot Agent 和 ROS 2 图均不应直接访问本服务。

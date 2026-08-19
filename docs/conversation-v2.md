# Assistant V2 conversation routing

`POST /internal/v2/conversation-turns` 是 H-IDE 控制 API 与 Mastra 之间的内部同步
分类边界。浏览器不得直连。请求必须携带与其他内部接口相同的
`Authorization: Bearer $MASTRA_INTERNAL_TOKEN`。

## Request

```json
{
  "schemaVersion": "h-ide.conversation-turn/2",
  "conversationId": "conversation-42",
  "robotId": "g1-sim",
  "locale": "zh-CN",
  "text": "只观察当前画面里有什么，不要移动",
  "history": [
    { "role": "user", "content": "你能看到什么？" },
    { "role": "assistant", "content": "我可以发起一次视觉观察。" }
  ],
  "memoryContext": {
    "schemaVersion": "h-ide.conversation-memory-context/1",
    "providedBy": "H_IDE_SERVER",
    "authority": "CONVERSATIONAL_ONLY",
    "resourceId": "authenticated-user-42",
    "revision": 7,
    "durableSummary": {
      "summaryId": "summary-42",
      "revision": 3,
      "text": "用户正在验证办公室仿真，并偏好简短的中文说明。",
      "coveredThroughTurn": 40,
      "generatedAt": "2026-08-19T08:00:00.000Z",
      "method": "SERVER_COMPACTION",
      "contentSha256": "d347d9bca43742ca4df939a4c6f47eed6d00a4c1da9c621dc9cfa3c97525af1b"
    },
    "relevantMemories": [
      {
        "memoryId": "preference-brief-zh",
        "kind": "USER_PREFERENCE",
        "text": "用户偏好简短的中文说明。",
        "recordedAt": "2026-08-19T07:30:00.000Z",
        "relevanceScore": 0.96
      }
    ]
  }
}
```

`schemaVersion`、`locale` 和 `history` 可省略，默认分别为
`h-ide.conversation-turn/2`、`zh-CN` 和空数组。`robotId`、`summary` 可省略。
最近历史最多 20 条、合计最多 40000 字符；摘要最多 12000 字符。旧调用方仍可发送
字符串 `summary`，新调用方应使用 `memoryContext.durableSummary`。二者同时存在时，
结构化且覆盖范围/修订号更新的摘要优先。

`memoryContext` 只能由经过认证的 H-IDE 服务端组装，不能从浏览器请求原样透传。
`providedBy` 和 `authority` 是固定字面值；摘要 SHA-256 必须与正文一致。相关记忆最多
8 条、正文合计最多 12000 字符。允许的记忆类型只有用户偏好、对话事实、既有决定、
当前话题和对话摘录，契约中没有“空间真值”类型。

## 长程记忆与审计

服务复用 `MASTRA_MEMORY_URL` 对应的现有 Mastra `LibSQLStore`，按已认证
`resourceId + conversationId` 隔离会话。它显式保存每轮用户/助手消息，但**不会**把
Mastra Memory 自动挂到 Router Agent 上：自动挂载会把取回内容提升到系统上下文，
与本项目的权限边界不符。

每次普通模型路由只接收三层有界数据：

1. 最近 20 条消息，用于当前语句和代词衔接；
2. 可持久化摘要，用于更早的长程话题；
3. 最多 8 条服务端相关记忆或本地词法检索到的旧对话摘录。

达到 30 轮后，服务每累计 10 个可覆盖轮次做一次确定性摘录压缩，保留最近 10 轮
不压缩。原始消息不会删除。线程 metadata 会保存摘要修订号、覆盖轮次、正文摘要、
每次选中的 memory ID，以及压缩源消息的 SHA-256。若有任何轮次缺失，服务不会虚报
覆盖范围。这样可以从 Mastra storage 复核“哪些历史进入了本轮、哪些消息进入了
摘要”，而不是依赖不可解释的隐式 LLM memory。当前检索采用有界的 `LEXICAL_V1`；
它不需要额外 embedding API，也不会让记忆服务阻断基础聊天。问候、感谢、告别和
有界四则运算仍完全绕过模型与记忆检索，只做短小的持久化追加。

## 信任边界

无论内容来自 H-IDE 服务端、历史数据库还是用户文本，送给模型时都被序列化在
`<UNTRUSTED_CONVERSATIONAL_CONTEXT authority="CONVERSATIONAL_ONLY">` 数据块中。
Router 的 system policy 明确拒绝其中的 prompt injection、工具调用、权限提升和
绕过审核指令。

会话记忆只能帮助理解“那里”指前文提到的“会议桌”、用户偏好哪种回答方式，或
之前决定先观察再行动。它不能证明会议桌的坐标、当前是否可见、道路是否空闲、目标
是否可达或动作是否安全。`ROBOT_ACTION` 仍只产生进入 Mission V2/PlanDraft 的语义
意图；坐标与执行条件必须由 typed SceneMemory、当前视觉/定位、Robot Agent 和 Nav2
重新解析并验证。记忆里的 `map(8,9)` 即使写得非常肯定，也只是引用文本。

## Response

```json
{
  "schemaVersion": "h-ide.conversation-result/2",
  "conversationId": "conversation-42",
  "robotId": "g1-sim",
  "intent": "ROBOT_OBSERVE",
  "assistantMessage": "好的，我会先整理一项仅观察当前环境、不会移动机器人的视觉任务。",
  "needsRobot": true,
  "shouldCreateMission": false
}
```

意图与权限含义固定如下：

| intent | 含义 | needsRobot | 可创建 Mission |
|---|---|---:|---:|
| `SOCIAL_CHAT` | 问候、感谢、闲聊 | false | false |
| `GENERAL_QA` | 不依赖实时传感器的知识/计算 | false | false |
| `ROBOT_OBSERVE` | 使用当前传感器，仅观察且不移动 | true | 否，进入只读 VisualTask |
| `ROBOT_ACTION` | 任意物理动作，包括底盘移动或旋转 | true | 是 |
| `SCENE_TEACH` | 命名、纠正或记忆当前场景 | true | 否，进入场景记忆流程 |
| `ADMIN_DEV` | 开发/管理诊断或配置请求 | false | false |

当模型需要追问时，响应增加 `clarification`，并强制
`shouldCreateMission=false`。H-IDE 应先展示追问，把用户回复作为新的 conversation
turn 发送；不得创建半完整 Mission。

`assistantMessage` 对普通聊天/问答是直接答案；对机器人相关意图只是口语化过渡，
不会包含坐标、路径、速度、ROS 接口或执行命令。`ROBOT_ACTION` 必须继续走既有
Mission V2 → PlanDraft → 权限/人工审核流程；`ROBOT_OBSERVE` 进入无运动的视觉任务，
不能隐式加入转向或导航。

## Model and latency profile

路由 Agent 复用 `SUPERVISOR_MODEL` 及其供应商凭据。Alibaba/Qwen 调用设置
`enableThinking=false`，DeepSeek 调用设置 disabled thinking，并统一使用
`temperature=0`。明显问候、感谢、告别和有界四则运算不访问模型、也不读取历史；其他输入才走
模型及现有瞬态网络重试。provider-facing schema 保持根 `type: object`，且不产生
DashScope 不接受的 `propertyNames`/`patternProperties`。

HTTP 状态：缺少内部认证为 `401`，请求契约错误为 `422`，模型/供应商最终失败为
`502`。本接口不启动规划、不调用 Robot Agent、不读相机，也不写场景记忆。

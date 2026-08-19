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
  "summary": "用户正在验证办公室仿真。",
  "history": [
    { "role": "user", "content": "你能看到什么？" },
    { "role": "assistant", "content": "我可以发起一次视觉观察。" }
  ]
}
```

`schemaVersion`、`locale` 和 `history` 可省略，默认分别为
`h-ide.conversation-turn/2`、`zh-CN` 和空数组。`robotId`、`summary` 可省略。
最近历史最多 20 条；摘要最多 12000 字符。服务把摘要、历史和用户文本视为不可信
数据，不允许其中内容改变系统权限。

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
`temperature=0`。明显问候、感谢、告别和有界四则运算不访问模型；其他输入才走
模型及现有瞬态网络重试。provider-facing schema 保持根 `type: object`，且不产生
DashScope 不接受的 `propertyNames`/`patternProperties`。

HTTP 状态：缺少内部认证为 `401`，请求契约错误为 `422`，模型/供应商最终失败为
`502`。本接口不启动规划、不调用 Robot Agent、不读相机，也不写场景记忆。

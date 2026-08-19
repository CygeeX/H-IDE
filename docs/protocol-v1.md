# `robot-agent.v1` 协议

每个 WebSocket 消息都是 UTF-8 JSON 文本帧，最大 65,536 字节，子协议固定为 `robot-agent.v1`。所有模型都拒绝未知字段和隐式类型转换。

公共信封：

```json
{
  "version": "1.0",
  "type": "command.request",
  "message_id": "UUID",
  "robot_id": "g1-dev",
  "session_id": "UUID",
  "sent_at": "2026-08-07T12:00:00Z",
  "payload": {}
}
```

## 连接顺序

1. WebSocket 完成 mTLS 和子协议协商。
2. 无论 Agent 是 client 还是 server，Agent 都先发 `session.hello`。
3. BFF 返回 `session.welcome`，租约参数必须与 Agent 本地配置一致，初始状态必须是 `SAFE_HOLD`。
4. BFF 发 `control.acquire`，Agent 回 `control.status=ACTIVE`。
5. BFF 每 500 ms 发送单调递增的 `lease.renew`；Agent 回 `lease.status`。
6. 最后一个已拥有命令到达终态后，BFF 发送 `control.release`。Agent 先回报
   `SAFE_HOLD/requested`，物理安全停止确认后再回报同一新 epoch 的
   `SAFE_HOLD/confirmed`；BFF 随后停止续租。
7. 2 秒内没有有效续租、连接断开或本地故障时，Agent 立即 safe-stop 并锁存 `SAFE_HOLD`。

`SAFE_HOLD` 只表示控制权已撤销，不能单独证明机器人已经物理停止。新增的兼容字段
`authority_revoked`、`safe_stop_state=requested|confirmed|failed` 和单调递增的
`safe_stop_epoch` 给出真实边界。只有同一 epoch 的本地 safe-stop 成功后才会发布
`confirmed` 并把取消中的命令写成终态；失败时保持禁止重新授权。旧客户端可以继续读取
原字段，但安全决策应同时检查这三个字段。

## 消息族

| 方向 | 类型 | 用途 |
|---|---|---|
| Agent → BFF | `session.hello` | 版本、boot ID、传输模式和能力摘要 |
| BFF → Agent | `session.welcome` | 分配 session ID 并确认本地租约策略 |
| 双向 | `session.goodbye` | 有原因的正常结束 |
| BFF → Agent | `control.acquire` | 显式获取控制权 |
| BFF → Agent | `control.release` | 无活动命令时显式归还控制权并确认物理安全停止 |
| Agent → BFF | `control.status` | `ACTIVE` 或锁存安全态 |
| BFF → Agent | `lease.renew` | 单调序号续租 |
| Agent → BFF | `lease.status` | 序号和到期时刻 |
| BFF → Agent | `command.request` | 逻辑命令、TTL 和受 Schema 约束的参数 |
| BFF → Agent | `command.cancel` | 取消一个活动 Action/等待任务 |
| Agent → BFF | `command.status` | 运行、反馈或终态 |
| Agent → BFF | `state.snapshot` | 每秒完整归一化状态 |
| Agent → BFF | `state.update` | 最多 5 Hz 的最新变化 |
| Agent → BFF | `safety.event` | 不与普通状态合并的高优先级安全事件 |
| Agent → BFF | `error` | 有界错误码和关联 message ID |

命令终态包括 `SUCCEEDED`、`FAILED`、`CANCELLED`、`REJECTED`、`TIMED_OUT` 和 `ABORTED_AGENT_RESTART`。断线后不会重放旧命令；BFF 可以用相同命令 ID 和完全相同的载荷查询已记录状态。
safe-stop 请求、确认或失败按 epoch 独立写入同一 SQLite 台账，便于把命令取消终态与
对应的物理停止凭据关联起来。

完整机器可读定义见 `src/robot_agent/schemas/robot-agent.v1.schema.json`。

## 与 RGB-D 数据面的关系

本协议仍然禁止图片和任意二进制 ROS 数据进入 WebSocket。RGB-D 使用独立的
`robot.observation.v1` Protobuf/gRPC 数据面，定义见
`proto/robot_observation/v1/observation.proto`。两者唯一的本地协作点是强类型 ROS
`ObservationDemand` / `ObservationHealth`：WebSocket 命令可以声明
`requires_observation` 并绑定 observation session、robot boot 和 localization epoch；
scene/PCD 仅为可选辅助信息，不能作为执行前置条件。命令不能借此选择 ROS 话题、
编码器或 TF frame。工作站发现观测超时时仍通过本协议的 `command.cancel` 取消导航；
机器人本地观测看门狗是独立后备，不依赖工作站取消能够到达。

## H-IDE 导航命令

`navigation.navigate_to_pose` 的参数严格服从随包保存的
`src/robot_agent/schemas/navigation.navigate_to_pose.schema.json`。Robot Agent 验证
boot/session/epoch 和观测新鲜度后，只把 `goalPose` 转成
`nav2_msgs/action/NavigateToPose` 的 `PoseStamped`，frame 固定为 `map`，
`behavior_tree` 留空。它不规划路径，也不把 `missionId`、`stageId`、
`visualResolutionId`、限速或其他审计字段写入 ROS Goal；这些字段只进入日志和 SQLite
命令账本。

命令登记后持续返回 `RUNNING`（包括 Nav2 feedback），随后返回
`SUCCEEDED`、`FAILED` 或 `CANCELLED` 等终态。相同 `command_id` 与相同规范载荷重发时，
返回账本中的当前或最终状态；相同 ID 携带不同载荷会被拒绝，不会执行第二次。

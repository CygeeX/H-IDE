# `robot-agent.v1` 协议

每个 WebSocket 消息都是 UTF-8 JSON 文本帧，最大 65,536 字节，子协议固定为 `robot-agent.v1`。所有模型都拒绝未知字段和隐式类型转换。

公共信封：

```json
{
  "version": "1.0",
  "type": "command.request",
  "message_id": "UUID",
  "robot_id": "g1-production",
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
6. 2 秒内没有有效续租、连接断开或本地故障时，Agent 立即 safe-stop 并锁存 `SAFE_HOLD`。

## 消息族

| 方向 | 类型 | 用途 |
|---|---|---|
| Agent → BFF | `session.hello` | 版本、boot ID、传输模式和能力摘要 |
| BFF → Agent | `session.welcome` | 分配 session ID 并确认本地租约策略 |
| 双向 | `session.goodbye` | 有原因的正常结束 |
| BFF → Agent | `control.acquire` | 显式获取控制权 |
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

完整机器可读定义见 `src/robot_agent/schemas/robot-agent.v1.schema.json`。

# 安全与弱网运行约束

本文档是控制台上线前的操作约束。它描述控制面能保证的事情，也明确哪些事情
必须由机器人本体和现场人员完成。

## 安全层级

1. 物理急停、机器人本地安全控制器和现场人员最高优先级。
2. 机器人侧 Robot Agent/驱动器的限位、碰撞和看门狗位于远程控制之下。
3. 工作站 `ActionRuntime` 负责命令校验、机器人运动租约、幂等、取消、事件和
   映射版本。
4. PWA、Agent 规划、场景和视频只是请求/观察界面，不能解除上层安全屏障。

远程 `stop` 不是物理急停的替代品；现场出现危险时先按本体急停，再处理网络和
软件状态。恢复运动前必须由现场人员确认设备、工作区和安全模式。

## 租约、有效期和重放策略

- 每个运动命令绑定一个机器人租约，并带 `expiresAt`。租约到期后 watchdog 将
  回执置为 `expired`，尝试通过 adapter cancel，并设置安全 barrier。
- 同一 `idempotencyKey` 只会产生一次 adapter dispatch；重试返回原始持久化回执。
  进程重启后 `accepted/running` 命令会保留事实并标记
  `runtime ownership unavailable … reconciliation required`，新进程不会猜测或
  自动重放。现场/运维必须通过机器人侧查询、日志和人工检查完成 reconciliation。
- Agent 计划的确认是 CAS 状态机。任何 `accepted/running/pending` 的步骤都会使
  计划进入 `unknown`，依赖步骤不会继续，第二次 confirm 被拒绝。
- 结构校验失败（例如 `expiresAt` 已在过去）的请求在进入 runtime 前直接返回
  `4xx`，不会占用幂等键；客户端应修正请求后使用新的明确幂等键。已被 runtime
  接受的过期/重复消息则永远不能重新 dispatch。

## 断网、VPN/5G 和适配器重启

控制 API 与机器人连接状态必须分开观测。对非 `SAFETY` 命令，只有持久化状态满足
`connected == true`、`status ∈ {online, degraded}` 且 `safetyMode ∈ {normal,
degraded}` 才放行；缺失、类型错误或未知状态一律 fail-closed。`stop` 保持在独立的
安全路径上。

建议的 WireGuard/5G 操作规则：

- VPN 握手、最近心跳、Robot Agent 会话和控制 API WebSocket 分别报警；不要只看
  浏览器“在线”图标。
- 发生丢包、切网或 adapter 重启时，停止发送运动重试；等待租约/回执收敛，按
  reconciliation 流程确认机器人实际状态。
- 重连只恢复事件游标（`after=eventId`）和状态读取，不恢复未证实的动作。
- 将 SQLite 文件放在持久卷并定期备份；重启演练必须验证命令、mapping pointer、
  audit 和 Agent 草案状态都能恢复。
- 场景转换、视频帧、ASR/规划服务的超时或崩溃只能显示 stale/degraded UI，不能
  拦截或替代 `stop`/`cancel`。

## 映射变更安全流程

1. 在 Interface Center 创建新 `mappingId/version` 草案。
2. `validate` 检查声明式字段、transport、target 和禁止的任意 URL/命令字段。
3. `probe` 做能力探测/dry-run；同一版本的顺序重探测使用前一次 probe 快照的
   乐观 CAS，若另一进程先发布结果则返回 `409`，需要重新读取后再试。
4. `activate` 通过 SQLite `BEGIN IMMEDIATE` 和 active-pointer CAS 原子提交；旧
   版本变为 `superseded`。协议/语义变化应发布新版本或新 adapter driver，不在
   已激活版本中偷偷改 payload。
5. 观察事件和真实机器人行为，异常时按版本回滚；回滚同样受 CAS 和审计约束。

启动恢复只接受“数据库行状态为 `active` 且 probe 成功”的 active pointer。指向
`draft`、`validated`、未探测或损坏行的 pointer 会被忽略并回退到最近安全 active
版本/seed，不会把草案提升为可执行版本。

## 弱网/故障注入验收

在接入真机前，至少完成以下矩阵；所有测试都应保存 command receipt、event cursor、
adapter 日志和机器人侧日志：

| 场景 | 注入方式 | 预期 |
|---|---|---|
| 延迟/丢包 | Toxiproxy、`tc netem` 或受控 WireGuard/5G 切换 | 状态变 stale/degraded；不自动重放运动命令 |
| adapter 重启 | 停止/重启 Robot Agent driver | 未证实的 accepted/running 进入 reconciliation；`stop` 仍可发 |
| 租约过期 | 缩短测试 lease 或阻断回执 | receipt=`expired`，尝试 cancel，安全 barrier 生效 |
| 重复/乱序 | 重发相同幂等键、延迟旧事件 | 只一次 dispatch；PWA 按 eventId 去重并可从 cursor 重放 |
| 场景/视频故障 | 停 worker、丢 ZMQ 帧、阻断 WebRTC | 画面/场景 stale，stop/cancel 成功路径不受影响 |
| 急停 | 真机物理 E-stop | 远程动作被拒绝；必须现场复位并重新确认，软件不得代替复位 |

`tests/test_end_to_end.py` 覆盖其中的可自动化部分：统一 runtime、v1/v2 切换与
回滚、重复/过期、重启/VPN 失败不重放、任务故障下 stop/cancel、严格安全状态和
损坏 pointer 恢复。真机急停、DDS/5G 断链和实际 PDAL/WebRTC 仍需在部署环境做
现场演练。

## 事件与审计检查

生命周期审计与 mapping/pointer 写入同一 SQLite 事务；事务失败时不发布“已激活”
假事件。运维排查时按 `traceId`、`commandId`、`mappingId/version` 和 event cursor
交叉核对，不要仅凭浏览器提示判断动作是否发生。

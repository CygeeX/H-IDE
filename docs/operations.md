# Robot Agent 运维与部署

## 文件布局

建议在真机计算单元使用：

- 程序与虚拟环境：`/opt/robot-agent`
- ROS 环境入口：`/opt/robot-agent/bin/robot-agent-entrypoint`
- 严格配置：`/etc/robot-agent/agent.yaml`
- CA、Agent 证书和私钥：`/etc/robot-agent/tls/`
- SQLite 台账：`/var/lib/robot-agent/ledger.sqlite3`
- systemd Unit：`/etc/systemd/system/robot-agent.service`

观测 sidecar 独立使用：

- 程序：同一只读 `/opt/robot-agent` 发布目录
- 配置与证书：`/etc/robot-observation-gateway/`（不得与控制 Agent 共用私钥）
- systemd Unit：`/etc/systemd/system/robot-observation-gateway.service`
- 运行用户：`robot-observation-gateway`，需授予相机访问组，但不授予控制设备权限

私钥应仅对 `robot-agent` 用户可读。不要把开发用明文配置复制到真机。

## 上线前门禁

1. 运行 `validate-config`，确认命令数量和能力摘要。
2. 运行 `doctor`，所有 ROS 类型和 safe-stop 类型必须是 `PASS`。
3. 在 Mock 或安全工位验证每个 Service/Action 参数和资源集合。
4. 验证错误 CA、错误 SAN 和无客户端证书不能建立连接；server 模式还要验证
   非固定客户端叶证书即使由同一 CA 签发也会被拒绝。
5. 主动断网与停止续租，确认 2 秒内机器人进入安全态。
6. 杀死 Agent 进程，由独立安全监视器确认 500 ms Deadline / 1 s Lifespan 心跳丢失。
7. 重启 Agent，确认遗留非终态命令变为 `ABORTED_AGENT_RESTART`，且仍需新 `control.acquire`。
8. 验证 D435i 是 USB 3.x、同步/对齐话题稳定，采集时刻 TF 与定位严格缺失时会丢帧。
9. 验证错误 Observation Gateway 客户端 SAN、错误工作站证书名和无客户端证书均失败。
10. 暂停 observation ACK，确认 750 ms 后只取消 `requires_observation` 导航；取消不确认时
    进入 `FAILED / CANCEL_NOT_ACKNOWLEDGED` 并 safe-stop。
11. 在隔离 ROS Domain 中只连接 `tools/mock_robot.py` 的真实
    `nav2_msgs/action/NavigateToPose` Mock，确认验收期间不存在真实 Nav2 server 或
    `/cmd_vel` 发布者。

## systemd

Unit 位于 `deploy/systemd/robot-agent.service`，使用 `Type=notify`、`WatchdogSec=5`、自动重启和只读系统保护。入口脚本会显式加载 `/opt/ros/humble/setup.bash`，避免 systemd 非交互环境缺少 ROS 变量与动态库路径。安装前按目标机修改 `User` 与设备访问策略，然后：

```bash
sudo systemd-analyze verify deploy/systemd/robot-agent.service
sudo install -D -m 0755 deploy/robot-agent-entrypoint /opt/robot-agent/bin/robot-agent-entrypoint
sudo install -m 0644 deploy/systemd/robot-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now robot-agent
journalctl -u robot-agent -f -o cat
```

Observation Gateway unit 没有 `BindsTo=robot-agent.service`，任一进程重启不会连带杀死
另一进程。控制 Agent 的 CPU 权重更高；观测进程达到内存/CPU 限额时优先牺牲图像吞吐：

```bash
sudo install -D -m 0755 deploy/robot-observation-gateway-entrypoint \
  /opt/robot-agent/bin/robot-observation-gateway-entrypoint
sudo install -m 0644 deploy/systemd/robot-observation-gateway.service \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now robot-observation-gateway
journalctl -u robot-observation-gateway -f -o cat
```

日志是一行一个 JSON 对象，可按 `robot_id`、`session_id`、`command_id` 和 `event_code` 索引。`doctor` 是只读命令，不会创建台账或调用 ROS 接口。

## 只读运行诊断 API

`diagnostics.enabled` 缺省为 `false`；生产模板也保持关闭。启用时 `host` 必须是字面量
loopback IP，不能配置为 `0.0.0.0`、LAN 地址或主机名。该端口没有控制接口、CORS 或图像
输出，不应通过反向代理直接暴露给浏览器；H-IDE 开发者页应由同机后端读取并按自己的
登录/RBAC 规则转发必要字段。

四个 GET 端点为：

- `/healthz`：只证明 HTTP 进程仍存活，不读取 ROS、会话或传感器状态。
- `/readyz`：聚合可执行就绪度；任一必需检查失败即返回 503。
- `/v1/diagnostics`：始终返回当前检查明细，未就绪时仍为 200，便于定位故障。
- `/v1/capabilities`：返回经过删减的能力、执行器和传感器元数据，不返回配置文件、TLS
  路径/密钥、命令参数 Schema、原始状态或图像。

`/readyz` 不把“进程启动成功”误报为可控制。它要求 runtime 完成初始化、工作站 transport
握手存在、控制状态为 `ACTIVE`、ROS executor 可接受请求、所有配置的 Nav2 Action server
可发现、每个配置的 `localization.*` 字段都在 `localization_max_age_ms` 内更新，以及所有
`requires_observation` 能力具有不超过 `admission_max_age_ms` 的 `USABLE` accepted ACK。
因此启动后尚未取得控制权或视觉尚未出首帧时返回 503 是预期行为。

本地配置显式使用 `127.0.0.1:8795`，MuJoCo 配置使用 `127.0.0.1:8796`，避免两套 Agent
同时开发时冲突。生产若部署同机 H-IDE 代理，可在评审后设置：

```yaml
diagnostics:
  enabled: true
  host: 127.0.0.1
  port: 8795
  request_timeout_ms: 1000
  probe_timeout_ms: 500
  localization_max_age_ms: 1000
```

`storage.retention_days` 是命令幂等窗口：Agent 启动时先把遗留非终态命令标记为
`ABORTED_AGENT_RESTART`，随后在启动阶段和运行期间周期性删除超过该窗口的终态
命令及其转移历史；非终态记录永不被保留期清理。

WebSocket server 模式必须配置 `tls.peer_certificate_sha256`，其值是获准 BFF 客户端
叶证书 DER 的小写 SHA-256：

```bash
openssl x509 -in bff-client.crt -outform DER | sha256sum
```

证书轮换必须把新证书与新指纹作为同一变更发布；不能用“同一 CA 签发”替代叶证书
身份固定。

## 故障处置

- `SAFE_STOP_FAILED`：禁止重新授权；先在本地确认安全链路和 ROS safe-stop 服务。
- `COMMAND_ID_CONFLICT`：BFF 复用了 ID 但修改了载荷；生成新 ID 或恢复原载荷。
- `ABORTED_AGENT_RESTART`：Agent 重启时命令未到终态；人工确认机器人状态后发送新命令，绝不自动重放。
- 反复断线：检查证书有效期/SAN、时钟、DDS/CPU 压力与 BFF 500 ms 续租节拍。
- `ACK_STALE`：检查工作站解码/ACK、gRPC 链路和 750 ms 在途超时；不要扩大队列。
- `CAMERA_UNAVAILABLE`：检查 USB 3.x、RealSense ROS 节点与同步 RGB-D 话题。
- `TF_MISSING` / `LOCALIZATION_MISSING`：修复采集时刻变换或定位源，不允许降级到最新位姿。
- 台账完整性：停服务后备份数据库，并运行 `sqlite3 ledger.sqlite3 'PRAGMA integrity_check;'`。

复制到真机、创建用户、安装证书和启用服务不属于本轮本地开发动作，必须在目标机变更窗口内单独执行。

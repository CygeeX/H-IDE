# Robot Agent 运维与部署

## 文件布局

建议在真机计算单元使用：

- 程序与虚拟环境：`/opt/robot-agent`
- ROS 环境入口：`/opt/robot-agent/bin/robot-agent-entrypoint`
- 严格配置：`/etc/robot-agent/agent.yaml`
- CA、Agent 证书和私钥：`/etc/robot-agent/tls/`
- SQLite 台账：`/var/lib/robot-agent/ledger.sqlite3`
- systemd Unit：`/etc/systemd/system/robot-agent.service`

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

日志是一行一个 JSON 对象，可按 `robot_id`、`session_id`、`command_id` 和 `event_code` 索引。`doctor` 是只读命令，不会创建台账或调用 ROS 接口。

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
- 台账完整性：停服务后备份数据库，并运行 `sqlite3 ledger.sqlite3 'PRAGMA integrity_check;'`。

复制到真机、创建用户、安装证书和启用服务不属于本轮本地开发动作，必须在目标机变更窗口内单独执行。

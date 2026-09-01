# 贡献与发布流程

## 分支

- `main` 只接受已验证的冻结集成版本。
- 日常改动从 `feature/<scope>` 或 `fix/<scope>` 分支开始。
- 运动控制、场景、模型锁、协议或权限变更必须在提交说明中写明影响的组件和
  L0/L1/L2/L3 验收等级。

## 提交前

```bash
git diff --check
./integration/acceptance.sh --level L0
./integration/acceptance.sh --level L1
```

涉及真实模型/ROS 时完成 L2 readiness；涉及实际 Mission/运动时，在操作者明确审核后
完成 L3 并保存去敏的结构化证据。不要把 `.env`、数据库、日志、原始 Evidence、模型、
LFS 测试数据或本机绝对路径加入提交。

## 依赖变更

更新 `third_party` 或外部模型时必须同时更新：

1. `manifests/vendor/*.json`；
2. `integration/integration.lock.yaml`；
3. `THIRD_PARTY_NOTICES.md`；
4. 对应内容/二进制 SHA-256；
5. clean-clone 复现记录和回退方案。

没有 `.gitmodules` 的 gitlink 不允许合入。无法公开分发的源码/权重必须采用
operator-fetch，并明确许可证确认步骤。

## 发布

发布者先在远端推送临时审计分支，再做完整历史的普通 clone。禁止使用 `--depth`、
`--shallow-*` 或 `--filter`；L0 的 Git 历史风险清单必须能看到全部可达对象。

clean audit 还必须：

1. 从当前活动 clone 执行 `integration/stop-e2e.sh`，停止固定的 `skillstore-e2e` Compose、
   `h-ide-mujoco-demo-*` systemd units 和 `h-ide-nav2-sim`；
2. 为新 clone 创建全新的 `SKILLSTORE_EXTERNAL_ROOT`，不复用另一 clone 的 G1、GR00T、
   SAM2、虚拟环境、ROS prefix 或运行数据库；SONIC 当前使用默认 Hugging Face cache，
   但必须通过发布模型锁的逐文件校验；
3. 按 `docs/reproduction/README.md` 从 manifest 安装依赖并运行 L0/L1/L2；
4. 使用 `integration/start-e2e.sh --rebuild ...`，不能让旧 image 冒充候选源码；
5. 经操作者审核后执行对应 L3，并保存结构化结果。

工作站 profile 使用固定 Compose/container/systemd/端口/ROS domain，两个 clone 不允许
并行启动。审计通过后再把同一提交更新到 `main`，最后创建不可变标签
`integration/office-v2-e2e-YYYYMMDD-rN`（例如
`integration/office-v2-e2e-20260901-r1`）。禁止在没有审计的情况下 force-push main。

## 凭据

公开 clone 不需要 token；push 使用开发者本机 credential helper。令牌不得写入
remote URL、脚本、CI 日志或 commit message。若凭据曾在聊天/终端中暴露，发布后
立即撤销并重新生成。

PWA/浏览器使用 server-side session Cookie + CSRF；`CONTROL_API_TOKEN` 仅为 legacy
loopback/internal compatibility，不得注入前端或放进 URL。Mastra、Scene、Vision 和
Robot Agent gateway 使用彼此独立的内部服务凭据。

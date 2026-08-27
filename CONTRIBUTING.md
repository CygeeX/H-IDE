# 贡献与发布流程

## 分支

- `main` 只接受已验证的冻结集成版本。
- 日常改动从 `feature/<scope>` 或 `fix/<scope>` 分支开始。
- 运动控制、场景、模型锁、协议或权限变更必须在提交说明中写明影响的组件和
  L0/L1/L2 验收等级。

## 提交前

```bash
git diff --check
./integration/acceptance.sh --level L0
./integration/acceptance.sh --level L1
```

涉及真实模型/ROS/运动时，再保存 L2 证据。不要把 `.env`、数据库、日志、模型、
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

发布者先在远端推送临时审计分支，使用另一个干净目录执行 L0/L1 和文档中的
配置检查；审计通过后再把同一提交推送到 `main`，最后创建不可变标签
`integration/office-v2-e2e-YYYYMMDD`。禁止在没有审计的情况下 force-push main。

## 凭据

公开 clone 不需要 token；push 使用开发者本机 credential helper。令牌不得写入
remote URL、脚本、CI 日志或 commit message。若凭据曾在聊天/终端中暴露，发布后
立即撤销并重新生成。

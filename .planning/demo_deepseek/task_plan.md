# DeepSeek 本地演示部署计划

## 目标
配置 H-IDE 与独立 Mastra Agent 使用 DeepSeek 模型，下载 Whisper 模型，保持外置 Agent 为空，以 FakeAdapter 启动并验证可交互演示。

## 阶段
- [complete] 1. 核对配置项、模型提供方和启动入口
- [complete] 2. 写入本地私密配置并生成内部 Token
- [complete] 3. 下载并校验 Whisper 模型
- [complete] 4. 使用 speech profile 启动完整演示栈
- [complete] 5. 验证网页、服务健康、DeepSeek 多 Agent 规划与 FakeAdapter 执行回执

## 安全约束
- 不在日志、进度文件或最终回复中记录用户 API Key 与内部 Token。
- 不配置外置 A2A Agent。
- 不触碰 PCD 未实现功能。

## 错误记录
| 错误 | 次数 | 处理 |
|---|---:|---|
| DeepSeek 官方 `/models` 对首个用户 Key 返回 401 | 1 | 识别为阿里云 `sk-sp` 套餐 Key；用户随后提供 DeepSeek 官方 Key，认证已恢复 |
| Whisper 下载完成后初次哈希断言失败 | 1 | Hugging Face CDN 最终响应 ETag 不是 LFS 内容哈希；改用源响应 `x-linked-etag`，实际文件哈希与其一致 |
| Control API `/healthz` 返回 404 | 1 | 服务没有该路由；改用网页、OpenAPI 和带认证业务接口做验收 |
| 首次真实 DeepSeek 规划失败：Structuring 合并遗漏/虚构 Worker 步骤 | 1 | 增加 Worker 命令目录隔离与越权输出校验，回归测试覆盖 |
| 修复后首次复验遇到 DeepSeek TLS 建连前断开 | 1 | 与 Worker 逻辑故障不同；正在对比宿主/容器网络与认证链路，不盲目重试规划 |
| 宿主诊断脚本 Python f-string 转义错误 | 1 | 改为普通字符串拼接；不影响服务或容器诊断结果 |
| Worker 隔离修复后 Structuring 改写了夹爪步骤 | 1 | Structuring 改为只输出 merge advice，由代码确定性保留不可变 Worker 字段并严格校验 |

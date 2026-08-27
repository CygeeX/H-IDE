# Findings

- Mastra 1.57 的 provider registry 原生列出 `deepseek/deepseek-v4-flash`，基础地址为 DeepSeek 官方 API，凭证变量是 `DEEPSEEK_API_KEY`。
- H-IDE Agent 原先缺少 DeepSeek 就绪检查，Compose 也未透传 `DEEPSEEK_API_KEY`；RED 测试已复现此缺口。
- Whisper Compose profile 为 `speech`，期望宿主目录中的 `ggml-small.bin`。
- 用户 Key 前缀为阿里云套餐专用 `sk-sp`；DeepSeek 官方端点返回 401。阿里云官方文档明确不允许将 Coding/Token Plan Key 用于自定义应用后端，故不能用于 H-IDE Mastra 服务。
- 替换后的 DeepSeek 官方 Key 已通过 `/models` 认证，响应模型列表包含 `deepseek-v4-flash`。
- 官方 whisper.cpp Hugging Face 文件 `ggml-small.bin` 大小为 487,601,967 bytes；LFS 源响应 `x-linked-etag` 和下载文件 SHA-256 均为 `1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b`。CDN 最终响应的普通 ETag 不是内容哈希；磁盘剩余空间约 1.5 TB。
- 演示端口 8080、8081、4111 空闲；宿主 5432/6379 已占用，但 Compose 未向宿主发布 PostgreSQL/Redis 端口，因此不冲突。
- 完整 Compose 栈已启动；Planning Agent `healthz/readyz` 为 200、Whisper 根页面为 200、H-IDE 页面为 200。Control API 没有 `/healthz` 路由。
- Control API 同样主动关闭 `/openapi.json`；需依据仓库 README/测试中的真实业务接口验收。Whisper 已成功加载 487.01 MB small 模型，在无 GPU 环境下使用 CPU 后端。
- 首次 DeepSeek 失败根因已定位：`manipulation-worker` 片段错误地同时包含 `move.forward` 与 `gripper.open`，而 `motion-worker` 已包含另一个 `move.forward`；Structuring 顶层对重复移动步骤做了去重，严格“所有片段步骤必须逐一出现”校验因此正确拒绝。问题发生在 Worker 专属命令边界，而非 FakeAdapter 或 H-IDE。
- TLS 失败日志为连接 `api.deepseek.com:443` 时一次性 `ECONNRESET`，标记 `isRetryable=true`；随后 Planning Agent 容器内 `/models` 请求立即返回 200 且包含目标模型，说明 DNS、CA、Key 和当前容器出口都正常，故障属于瞬时上游/网络断连。
- Worker 隔离后的失败具体是 Structuring 为夹爪顶层步骤增加正确跨 Worker 依赖的同时，改写了 `rationale`；命令、payload、stepId 和 contributor 都未越界。结论：LLM 适合决定摘要、顺序和跨片段依赖，但不应负责重新抄写不可变 Worker 字段。将改为 Structuring Agent 输出 merge advice，代码按原片段确定性组装并再次用严格 PlanProposal schema 校验。
- 确定性合并版本的真实 DeepSeek 任务已成功：Supervisor 选择 motion + manipulation，两步草案包含 `move.forward` 后依赖执行 `gripper.open`；人工确认接口返回 confirmed，两条 FakeAdapter 回执均 succeeded。
- 已从持久化 API 复核：草案状态为 confirmed；两个命令均保存 accepted→running→succeeded 事件序列，receipt.result.accepted=true，且 review 中保留两个 Worker 原始贡献与跨 Worker 依赖。
- 真实 Whisper 推理链路通过：FFmpeg/Flite 生成的 16 kHz mono PCM16 WAV 经 Control API 上传后，CPU 上的 ggml-small 正确转写为 “Move forward then open the gripper.”。

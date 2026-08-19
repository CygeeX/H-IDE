# PCD 场景与 Agent 空间目标闭环

## 目标
实现 PCD 分片导入、后台转换、Potree 场景、双模式坐标标定、语义区域标注、Mastra 确定性空间解析，以及 FakeAdapter 审核执行后的位姿更新。

## 阶段
- [complete] 1. 场景/位姿/空间解析契约与纯函数
- [complete] 2. Scene API、本地存储、上传与作业状态机
- [complete] 3. Scene Worker、固定转换工具链与同源网关
- [complete] 4. PWA 场景库、查看、标定和移动端标注
- [complete] 5. Mastra 工具、PlanDraft 校验与 Fake 位姿闭环
- [complete] 6. Compose、端到端验收、文档和必要回归

## 已锁定决策
- 2.5D；ROS map identity 与交互 similarity 两种标定模式。
- 语义多边形 + 单一安全入口位姿；不可变 SceneRevision。
- 本地盘存储、原始 PCD 手动删除、最大 2 GiB、8 MiB 分片。
- scene-api + scene-worker 独立；统一 8080 网关。
- PCD ascii/binary 走 PDAL；binary_compressed 先由 PCL 规范化。
- Mastra 只选择意图，确定性工具算坐标，Control API 复核。
- H-IDE 只解析目标；真实路径规划属于 Robot Agent/Nav2。
- V1 使用 FakeAdapter 完整闭环，并支持桌面/手机完整标注。

## 安全约束
- 不覆盖或清理现有未提交改动，不提交用户密钥。
- 原始点云及 Potree 资产 URL 不进入 LLM 上下文。
- 场景服务故障不得影响现有 stop/cancel。
- 不执行浏览器提供的 pipeline、命令行或文件路径。

## 错误记录
| 错误 | 次数 | 处理 |
|---|---:|---|
| Scene Worker 容器运行 PotreeConverter 时缺少 `liblaszip.so` | 1 | 将运行库安装到 `/opt/potree/lib` 并加入 `LD_LIBRARY_PATH`，重建后 `ldd` 校验通过 |
| PotreeConverter 以 PATH 中的短命令名启动时错误解析工作目录 | 1 | Worker 在启动前用 `shutil.which` 固定为绝对可执行路径 |

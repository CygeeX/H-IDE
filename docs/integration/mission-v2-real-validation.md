# Mission V2 真实联调与验收清单

**状态：代码完成；自动化夹具通过；真实 Robot Agent / D435i / Mock Nav2 的 Mission V2 整链验收待执行。**

本文只封存需要真实组件的验证步骤。它不把 FakeAdapter、合成 RGB-D 或协议模拟器结果写成实体机器人成功。

## 已由自动化测试覆盖

- Mission 草案必须人工批准后才能签发技能。
- Qwen 假设与 RGB-D 几何事实分库存储，Qwen 不能提交事实。
- `视觉 → 几何目标 → ActionRuntime → Robot Agent → Nav2` 连续 DAG 可完成。
- Robot Agent 返回 `running` 后，H-IDE 查询同一命令 ID 并推进，不会重复 dispatch。
- 活动 Mission 取消会先取消同一 Nav2 命令 ID，再将 Mission 置为终态；不会发送替代动作。
- PCD 语义区域只有通过 Scene Map 检查才形成静态目标事实。
- 单段与累计直线目标预算、固定 Boot/Epoch、能力版本均在发送前复核。
- 所有 Mission 运动能力要求 `executor=nav2`，输入拒绝 `cmd_vel`、速度、路径和航点。
- 持续对话产生不可变 PlanRevision，并在需要时重新进入人工审核。
- PWA 默认使用 `/api/v2`，支持刷新恢复、文本、语音、PCD 场景入口和事件时间线。

## 联调前提

- [ ] 固定 Robot Agent commit SHA，工作树无未说明改动。
- [ ] `validate-config` 发布与 H-IDE 完全一致的 RobotProfile / CapabilityManifest digest。
- [ ] `headMotion=fixed`，D435i 为固定、深度可用的 `head_rgbd` 传感器。
- [ ] Robot Agent 至少发布 `navigation.navigate_to_pose@2.0.0`；扫描测试还需 `navigation.rotate@2.0.0`，两者 `executor=nav2`。
- [ ] Observation Gateway 提供稳定的 Boot、Observation Session、Localization Epoch、RGB、对齐深度、内参和 TF。
- [ ] Robot Agent 的 `/localization/pose` 为 `map` 坐标，并通过状态通道在 2 秒新鲜度窗口内投影为 H-IDE `currentPose`；新鲜度按定位字段变化对应的 Robot Agent UTC 状态时间计算（兼容 ROS 仿真时钟，无关遥测不会刷新），不得使用默认 `(0,0)`。
- [ ] Mock Nav2 的 NavigateToPose 与 Spin action server 可用，且测试域不会驱动物理底盘。
- [ ] Qwen 文本与视觉 API 只通过服务端环境变量注入；日志和浏览器中没有密钥。
- [ ] H-IDE Vision、Scene API、Mastra、Control API、Gateway 的内部 token 一致。

## 推荐配置

```dotenv
SUPERVISOR_MODEL=alibaba-cn/qwen3.8-max
WORKER_MODEL=alibaba-cn/qwen3.8-max
STRUCTURING_MODEL=alibaba-cn/qwen3.8-max
HIDE_VISION_VLM_MODEL=qwen3-vl-plus
HIDE_VISION_SOURCE_MODE=live
HIDE_VISION_COMPOSE_URL=http://host.docker.internal:8092
HIDE_VISION_SAM2_DEVICE=cuda
# HIDE_VISION_SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_tiny.pt
ROBOT_AGENT_GATEWAY_ENABLED=true
ROBOT_AGENT_GATEWAY_BACKEND=websocket_inbound
```

`DASHSCOPE_API_KEY`、控制 token 和 Gateway token 只写本机未提交的 `.env` 或进程环境。
视觉服务必须先在宿主机 GPU 环境启动；其 `/healthz` 只有同时报告
`provider=qwen`、`model=qwen3-vl-plus`、`segmentation.provider=sam2` 且两项
`available=true`，才可进入真实视觉用例。Docker 内不运行备用视觉推理服务。

## 验收用例

### M2-01：审核门

输入“找到银白色机器人并去机器人旁边”。审核前确认 Robot Agent ledger 和 Mock Nav2 没有新 Goal。页面应显示 `vision.observe → navigation.navigate_to_pose` 语义 DAG，不能出现坐标、路径或速度。

### M2-02：真实视觉闭环

批准计划后保持相机固定，让目标位于画面和有效深度范围内。确认：

- Qwen3-VL-Plus 产生像素假设；
- 三帧深度、TF、时钟、聚类和净空门通过；
- H-IDE 只选择一个 `map` 目标；
- Robot Agent 只收到一次 NavigateToPose；
- Mock Nav2 Goal 坐标与 H-IDE 几何事实逐字段一致；
- terminal receipt 使 Mission 进入下一节点或 `completed`。

### M2-03：PCD 厨房

导入并发布场景，标注“厨房”与入口点，绑定机器人后输入“去厨房”。审核卡必须显示场景版本。发送参数应为 `payloadType=spatial_resolution` 和一个入口位姿；PCD 不得提供路径。占用检查为 `occupied/unknown/outside` 时必须停止在 `needs_input`。

### M2-04：不可见目标与底盘扫描

让目标不在固定相机视野内。重规划若选择扫描，只能签发 `navigation.rotate`，Robot Agent 必须映射为 Nav2 Spin。ROS 图和日志中不得出现头部转动命令或 H-IDE 直接发布的 `/cmd_vel`。

### M2-05：异步与幂等

让 Mock Nav2 保持 running 数秒。多次调用 `/advance`，确认 Robot Agent 只有一次 command receive / ROS Goal；后续均为 query。断开工作站网络后恢复，也不得自动重放。

### M2-06：视觉歧义与补充对话

摆放两个相似目标。视觉服务应返回 `needs_input`，PWA 允许用户补充参照物。补充内容必须生成 PlanRevision；需要扩展或改变中风险动作时必须再次人工审核。

### M2-07：突发障碍

在目标点或路径上放置障碍。H-IDE 不自行改路径；Nav2/本地 BT 负责避障或失败。失败回执触发事件评估，H-IDE 只能重观察、选择新目标或请求人工审核。

### M2-08：安全与恢复

分别测试急停、取消、Gateway 断线、Robot Agent 重启、Localization Epoch 变化和 Control API 重启。任何身份变化都必须让旧授权失效；未知物理状态不得自动启动新运动。

## 证据包

每个用例保存：Mission JSON、事件序列、WorldRecord、视觉证据 SHA256、SkillRequest、Control API 回执、Gateway trace、Robot Agent ledger、ROS Goal、Nav2 result、组件 commit SHA 与配置 digest。验收报告必须明确是 Mock Nav2 还是真实物理底盘。

## V1 删除门

只有 M2-01～M2-08 通过并由操作者确认 PWA 工作流后，才删除旧 Assistant / VisualMission 路由、表、前端类型和测试。删除前导出仍有审计价值的历史记录；不迁移可执行状态，也不让 V1 与 V2 同时拥有运行中任务。

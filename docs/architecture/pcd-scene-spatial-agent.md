# PCD 场景与空间 Agent 闭环

本文描述 H-IDE 已实现的 PCD 导入、浏览器地图交互和 Agent 空间目标解析。它解决
“目标在哪里、目标坐标是什么”，不在工作站内规划路径；碰撞检测、可达性判断、
局部重规划和底盘控制仍由真机 Robot Agent/Nav2 负责。

## 数据流与信任边界

```text
PCD 文件 ── 浏览器分片上传 ── Scene API ── 本地持久化
                                      │ leased job
                                      ▼
                           Scene Worker（固定命令链）
                   PCL → PDAL → Open3D + PotreeConverter
                      │ 清理/自适应降采样/地面分析
                      │ 原子发布 Potree LOD + 未验证二维预览
                                      ▼
                    二维占用/高度图或三维查看、标定、语义标注
                                      │ 标定触发正式 map 栅格
                                      │ 发布不可变 revision 并绑定 robot
                                      ▼
文本 / 已审核语音 / 点云选点 ── TaskRequest.sceneContext
                                      ▼
                       Mastra motion-worker 选择空间意图
                                      ▼
                       resolve_spatial_target 确定性工具
                                      ▼
                       navigation.navigate_to_pose 草案
                                      ▼
                         H-IDE 校验 → 人工审核
                                      ▼
                     再校验场景/位姿漂移 → ActionRuntime
                                      ▼
                              Robot Agent / Nav2
```

浏览器不能提交转换命令、PDAL pipeline、可执行文件参数或服务器路径。Worker 只从
受内部 token 保护的作业端点取得 UUID，并自己推导固定路径和命令。Mastra 收到的
只是 `sceneRef`、map 边界、语义区域、正式地图摘要、当前位姿和可选点；原始 PCD、
栅格、Potree 文件、标定矩阵、资产 URL 与服务器路径都不会发送给模型。

## 使用步骤

1. 在 Agent 页面选择机器人，展开“PCD 场景”卡片。
2. 把 `.pcd` 拖入导入区，或点击“选择 PCD 文件”。浏览器按服务端给出的分片大小
   上传并显示进度；断线后协议支持从服务端 offset 继续。
3. 上传完成后等待后台状态从“排队转换/转换中”变为“转换完成”。转换任务已持久化，
   离开页面不会取消它；确定性失败可在页面重试。
4. 转换完成后默认查看未验证的二维占用预览，也可切换高度图或三维点云。预览只用于
   理解方位和辅助标定，不能作为 `map` 安全证据。完成一种坐标标定：

   - `ROS map 同坐标`：PCD 已经使用米、Z-up、正确 map 原点和轴向时选择；变换为
     identity。
   - `交互相似变换`：依次点选至少三个不共线地面点、场景原点和 X 轴参考点，填写
     参考段真实长度及其 map 原点。服务端拟合尺度、地面法向、朝向与平移，并显示
     `m/unit` 和拟合残差。

5. 可选：点选至少三个边界点和一个安全入口，填写稳定的区域 ID、中文名称、别名、
   高度范围与入口 yaw。例如 `kitchen / 厨房 / 烹饪区`。Agent 对“去厨房”的结果
   永远是该区域显式保存的入口位姿，而不是多边形中心或模型猜测的位置。
6. 保存标定会排队生成与“本次点云处理产物 + calibration”共同绑定的正式地图。
   等待正式地图完成后发布 revision。发布后的版本不可修改；修改时从它克隆一个新
   草稿。随后把已发布 revision 绑定到当前机器人。新版本不会悄悄切换现有绑定。
7. 用以下任一方式创建任务：

   - “向前 5 米、向左 1 米”：基于提交时的机器人 `map` 位姿和 yaw 计算相对目标；
   - “去厨房”：匹配已保存语义位置并采用安全入口；
   - 在点云中点击“选 Agent 目标点”，再提交“去选中的位置”：使用前端提交的单个
     map 点。H-IDE 只接受与当前已发布绑定匹配的 `sceneRef`，不会接受前端伪造的
     完整场景上下文。

8. 审核页会显示 `map` 下的 x/y/z/yaw，并把目标分类为“未发现障碍证据 / 障碍 /
   观测不足 / 地图外”。这只是人工审核提示，不改写目标、不规划路径，也不替代真机
   避障。确认后才进入 `ActionRuntime`。FakeAdapter 演示成功时会把机器人当前位姿
   更新为目标；接入真机后应由 Robot Agent 的观测状态成为权威位姿。

## 确定性解析与二次校验

Mastra 的模型只输出三类 `SpatialIntent`：

- `relative {forwardMeters, lateralMeters, yawDeltaRad}`；
- `semantic {locationId}`；
- `selected_point {pointMap, yaw?}`。

`motion-worker` 决定静态场景是否适合辅助当前任务，并输出一种 `SpatialIntent`；
`resolve_spatial_target` 只是候选坐标解析工具。运行时从 H-IDE 注入的 canonical
`SceneContext` 中读取模型已经选择的意图，生成不可变的 `goalPose +
spatialResolution`，其中带有 scene revision、输入意图、规划时位姿、resolver
版本和时间。模型不得自己做坐标运算，也不能改写工具输出。工具同样不判断现场
动态占用、路径可达性或执行安全；例如标注点后来被货箱占用时，PCD 结果仍只能作为
审核证据，实时结论由相机/传感器、Robot Agent 和 Nav2 提供。

Control API 有两道独立检查：

1. 创建 `PlanDraft` 前，用冻结的 scene/pose 快照重新解析，要求目标逐字段一致；
2. 用户确认时，重新读取机器人当前绑定、场景和位姿。revision 改变、语义区域改变、
   目标被篡改、resolver 版本不支持，或位姿相对规划时漂移超过默认 `0.5 m / 0.35
   rad`，都会在零下发状态下拒绝确认。

这使“模型返回了一个看似合理的坐标”和“这个坐标来自当前可信地图事实”成为两件
可分别验证的事情。

## 文件格式、存储与恢复

- 接受 PCD `ascii`、`binary` 和 `binary_compressed`。压缩格式先用 PCL 转成标准
  binary；PDAL 做有界抽样、统计离群点清理和自适应体素降采样，同时生成分析 PLY
  与 LAZ；PotreeConverter 生成分级点云，Open3D CPU 拟合主地面并派生占用/高度图。
  原始 PCD 不会被降采样覆盖。重新处理会创建绑定新产物的新草稿 revision；已发布
  revision 继续固定引用原产物，不会改变现有机器人绑定。
- 默认单文件上限 2 GiB、分片 8 MiB，并在接受上传前预留约四倍文件空间和 1 GiB
  安全余量。可通过环境变量调整，但调整前应做最坏情况磁盘演练。
- `scene-data` 保存 SQLite 元数据、原始 PCD、临时工作目录和发布产物。产物先在
  Worker 私有临时目录校验，再用目录重命名原子发布，浏览器不会看到半成品。
- Worker 通过 lease/heartbeat 领取任务；崩溃后的过期 lease 会重新排队，超过最大
  尝试次数后失败。停止 Worker 只会让场景转换停滞，不影响手动 `stop/cancel`。
- 资产会话有效期有限且使用同源 cookie。PWA service worker 明确不缓存
  `/api/` 和 `/scene-assets/`，避免离线页面把旧地图误当成当前地图。
- 原始 PCD 的删除是显式 API 操作，不随发布自动删除。备份至少应覆盖
  `control-data`、`planning-data` 和 `scene-data`，并定期验证恢复后的 revision
  绑定和文件清单一致。

## 部署配置

完整演示推荐使用根目录 Compose：

```bash
cp .env.example .env
# 填写三个彼此不同的长随机 token、Mastra 模型和供应商凭据
docker compose --env-file .env -f infra/docker-compose.yml up --build
```

浏览器和手机只访问 `http(s)://<workstation>:${CONTROL_API_PORT:-8080}`。Caddy 把
`/api/v1/scene-*`、`/api/v1/scenes*`、机器人场景绑定和 `/scene-assets/*` 路由到
Scene API，其余控制请求进入 Control API。`SCENE_API_PORT` 只发布到
`127.0.0.1`，供 `ros2-host` profile 的 Control API 读取场景上下文，不应暴露到
局域网。

主要配置：

| 配置 | 默认值/作用 |
|---|---|
| `SCENE_API_INTERNAL_TOKEN` | Control API、Worker 与 Scene API 间的独立长 token |
| `SCENE_MAX_UPLOAD_BYTES` | `2147483648`，单个原始 PCD 上限 |
| `SCENE_UPLOAD_CHUNK_SIZE` | `8388608`，浏览器上传分片大小 |
| `SCENE_DISK_RESERVE_BYTES` | `1073741824`，上传后仍需保留的空间 |
| `SCENE_JOB_LEASE_SECONDS` | `300`，Worker 作业租约 |
| `SCENE_WORKER_CPUS/MEMORY` | 默认 `2 CPU / 4g` 的转换资源上限 |
| `SCENE_TARGET_CLEARANCE_METERS` | 默认 `0.35`，审核时目标周围障碍证据的净空半径 |
| `SPATIAL_CONFIRM_TRANSLATION_DRIFT_METERS` | 默认 `0.5`，确认时允许的平移漂移 |
| `SPATIAL_CONFIRM_YAW_DRIFT_RADIANS` | 默认 `0.35`，确认时允许的朝向漂移 |
| `SCENE_ASSET_COOKIE_SECURE` | HTTPS 生产环境应设为 `true` |

手机 PWA 安装、录音和浏览器端 SHA-256 校验需要安全上下文。`localhost` 可使用
HTTP；通过局域网 IP 访问真机时，应在 Caddy/企业反向代理提供受信任 HTTPS，或经
WireGuard 使用受控域名和证书。不要把 Scene API、Mastra、Whisper、DDS 或 Robot
Agent 的内部端口直接开放给手机。

## 当前 V1 边界

- 当前正式地图针对静态、单楼层室内场景，不做在线 SLAM 更新、多楼层切换、自动
  拼接或工作站路径规划。
- 标注由人审核，没有让视觉模型直接理解原始 PCD；Agent 只读取受控摘要，栅格由
  Control API 用于确定性的目标证据查询。
- H-IDE 生成 `map` 目标，不声称目标可达。真机接入必须让 Robot Agent/Nav2 对
  frame、地图版本、代价地图、安全区和路径执行做最终校验与反馈。

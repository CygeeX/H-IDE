# 纯 VLM 产品运行配置

H-IDE 只有一条视觉产品链：宿主机 GPU 上的 H-IDE Vision 使用
`Qwen / qwen3-vl-plus` 做语义框选，SAM 2 做掩码细化，同步 RGB-D、相机标定和 TF
负责几何定位。视觉服务不规划路径、不发速度；任何底盘位移和原地转向仍由
Robot Agent 交给 Nav2。

## 为什么视觉服务运行在宿主机

当前锁定的 SAM 2/CUDA 环境和 checkpoint 位于宿主机。通用 CPU Docker 镜像既不
包含该 GPU 运行时，也不能被当成备用推理服务。因此 base Compose 和 simulation
overlay 都不声明视觉推理容器；`control-api` 与 `planning-agent` 统一访问
`http://host.docker.internal:8092`。端口 `8092` 是唯一产品 HTTP 服务，观测网关连接
同一进程的 gRPC `50061`。

`visual-fixture` profile 只保留 RGB-D 观测协议发送器。它不生成标签、像素框或掩码，
不能用于纯 VLM 验收，也不会随默认 Compose 启动。

## 启动顺序

先在 H-IDE Vision 的锁定 Python 3.11/GPU 环境设置以下服务端变量并启动服务：

```bash
cd /home/phi/learn-projects/h-ide-vision
export DASHSCOPE_API_KEY='由操作者在本机注入'
export HIDE_VISION_QWEN_MODEL=qwen3-vl-plus
export HIDE_VISION_SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_tiny.pt
export HIDE_VISION_SAM2_DEVICE=cuda
uv run h-ide-vision serve
```

再启动 H-IDE：

```bash
cd /home/phi/learn-projects/h-ide
docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.simulation.yml config --quiet
docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.simulation.yml up --build
```

## Readiness 门

开始视觉任务前，开发者应读取宿主机 `http://127.0.0.1:8092/healthz` 并确认：

- `releaseState` 是 `pure_vlm_product_profile`；
- `capabilities.vlm` 为 `provider=qwen`、`model=qwen3-vl-plus`、`available=true`；
- `capabilities.segmentation` 为 `provider=sam2`、`available=true`；
- `visualTaskV2.available=true`，语义边界为 `qwen_only`，掩码边界为
  `sam2_from_qwen_box_only`。

任一条件不满足都应停止视觉验收；不得切换到 fixture、渲染器标签或其他语义来源。
这项检查只验证本地配置与模型可用性，不会发起计费推理。

## 历史数据

迁移不会删除旧 Mission、SQLite 记录、视觉证据文件或 Docker volume。旧语义记录
仍可读取和导出，但 API 投影会标记 `legacyReadOnly=true`、
`executionEligible=false`；新 Mission 契约拒绝该来源，前端也不会把旧结果解析成
当前产品视觉结果。FoundationPose 只允许留在 H-IDE Vision 的未安装实验目录，
H-IDE 没有对应环境变量、provider 注册、API 或 Mission 路由。

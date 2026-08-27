# Robot Observation Gateway

## 作用与边界

Observation Gateway 是机器人侧独立 sidecar，只处理对导航/感知有用的 RGB-D 观测。
它不会接收运动命令，也不会把图片塞入 Robot Agent 的 64 KiB JSON WebSocket。控制面
的单会话、2 秒租约、safe-stop 与状态优先级完全保留。

```text
RealSense ROS (sync + align)
      │ RGBD / CameraInfo / metadata
      ▼
capture-time TF + localization-at-or-before
      │ capacity 1, newest-only
      ▼
JPEG 85 RGB + Zstd Z16 depth + CRC32C
      │ reverse mTLS bidirectional gRPC, one frame in-flight
      ▼
Workstation ACK_ACCEPTED ──► typed ROS health ──► navigation watchdog
```

每帧携带 robot/boot/gateway/session 身份、递增序号、原始采集时间、相机帧号、RGB、
对齐后的 16 位深度和尺度、内参与标定摘要、采集时刻的两个 TF、定位协方差、场景版本、
编码与校验值。遥测包括采集年龄、编码/排队/写阻塞/ACK 往返、扣除工作站处理时间后
估算的 `networkMs`、带宽、丢弃计数和序号缺口。

协议唯一标准是 H-IDE Vision 的 `robot_observation/v1/observation.proto` 与
`PROTOCOL.lock`。本仓库保存其逐字节副本，Hello 固定发送 `major=1`、`minor=0` 和
lock 中的 descriptor SHA-256。真相机入口只能声明 `SOURCE_MODE_LIVE`，同时上报
`source_id`、`localization_epoch_id` 和 `clock_uncertainty_ms`；没有 PCD 场景时
`scene_version` 保持未设置，而不是伪造一个必填场景。

## D435i 本地启动

本地配置已记录当前设备序列号 `261222077185`、固件 `5.15.1.55` 和 640×480×30 profile。
官方 RealSense ROS 必须启用同步 RGB-D、同步器和深度对齐：

```bash
unset CYCLONEDDS_URI
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch robot_observation_gateway realsense_observation.launch.py \
  serial_no:=_261222077185
```

上面的组合 launch 同时启动相机和 Gateway，适合已有 gRPC 工作站接收端的命令行联调；
不要再单独执行 Gateway CLI。若要使用本仓库的网页接收端观看真机画面，把 launch 参数
改为 `start_gateway:=false`，然后按[可视化测试台](web-demo.md)的三个终端步骤运行，
由网页进程唯一地启动本地接收端和 Gateway。

机器人还必须发布 `map → base_link → camera_color_optical_frame` 的 TF，以及
`/localization/pose` (`PoseWithCovarianceStamped`)。Gateway 使用 RGB-D 消息的采集时间
查询 TF，并只接受不晚于该采集时间且年龄不超过 100 ms 的定位；缺失时直接丢帧，绝不
用固定外参或“最新位姿”代替。`localization_fixture` 仅供本地台架测试，不能上生产。

不使用组合 launch 时，先启动工作站接收端，再单独运行 Gateway：

```bash
PYTHONPATH=src/robot_observation_gateway:${PYTHONPATH} .venv/bin/python \
  -m robot_observation_gateway.cli \
  --config src/robot_observation_gateway/config/robot_observation_gateway.local.yaml
```

最方便的端到端观看方式是：

```bash
PYTHONPATH=src/robot_agent:src/robot_observation_gateway:${PYTHONPATH} .venv/bin/python \
  tools/web_demo_server.py --observation-source realsense --open-browser
```

## 生产部署

- 使用生产配置模板并替换 robot ID、工作站地址、相机序列号/固件和证书路径。
- Gateway 主动连接工作站；服务端证书名必须匹配 `server_name`，客户端证书 SAN 必须
  绑定 `robot:<robot_id>`。
- Observation Gateway 使用独立用户、证书和 systemd unit；不要与 Robot Agent 共用
  私钥或进程资源。unit 将其限制为 `MemoryMax=512M`、`CPUQuota=150%`，控制 Agent 的
  CPU 权重更高。
- 生产协议 v1 强制严格定位，配置 `strict_localization: false` 会在启动时被拒绝。
- 不落盘、不回放图片；日志和 ROS health 只包含诊断与计数，不包含图像字节。

## 导航安全语义

工作站下发 `navigation.navigate_to_pose` 时必须绑定当前 `robotBootId`、
`observationSessionId` 和 `localizationEpochId`。PCD `sceneRef` 是可选辅助信息，不参与
强制准入。Robot Agent 只在最近 500 ms 内收到匹配绑定的 `ACK_ACCEPTED` 健康时准入。
执行期间 750 ms 没有新的 accepted sequence，或 boot/session/epoch 发生变化，就发送
本地 Nav2 Action cancel。普通动力/任务命令不受该观测看门狗影响；如果导航取消未被
ROS Action 确认，命令进入 `FAILED / CANCEL_NOT_ACKNOWLEDGED` 并触发 safe-stop。

工作站控制项目还应独立监测观测超时，并通过既有控制 WebSocket 发送
`command.cancel`；取消命令不能借道观测 gRPC。这个仓库实现的是机器人侧第二道防线，
即使工作站断线或来不及发出取消，本地看门狗仍会停止标记为
`requires_observation` 的导航目标。两条取消路径都走同一 ROS Action 取消确认和
fail-closed 处理。

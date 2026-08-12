# Robot Agent 可视化测试台

这个页面把项目中原本分散在 WebSocket 消息、SQLite 台账和 ROS 2 graph 中的行为，
集中显示为一个本地驾驶舱。它不是生产 BFF，也不会开放通用 ROS 操作。

## 一键运行真实本地链路

在项目根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src/robot_agent:${PYTHONPATH} .venv/bin/python \
  tools/web_demo_server.py --launch-stack --open-browser
```

若浏览器没有自动打开，访问 <http://127.0.0.1:8765/>。退出时在启动终端按
`Ctrl-C`，启动器会依次停止 Robot Agent 与 Mock Robot。

`--launch-stack` 会启动以下真实测试链路：

```text
测试网页 /ui
      │ 受限 dashboard action
      ▼
Visual BFF /agent
      │ robot-agent.v1 WebSocket
      ▼
Robot Agent
      │ ROS 2 Service / Action / Topic
      ▼
Mock Robot
```

Mock Robot 与 Agent 默认使用独立的 `ROS_DOMAIN_ID=232`，并把 DDS discovery 限制为
本机，避免碰到开发机上的其他 ROS graph。网页与 BFF 只监听 `127.0.0.1`；浏览器
WebSocket 还校验 Origin。该入口仍只应在开发机或隔离安全工位使用，不应替代生产
mTLS BFF。

## 推荐操作顺序

1. 确认页面顶部显示“Agent 在线”，初始状态是 `SAFE_HOLD`。
2. 点击“获取控制权”，观察 `control.acquire → control.status=ACTIVE`。
3. 点击 `ENABLE`，观察逻辑命令映射到 ROS Service，`robot.mode` 变为 `IDLE`。
4. 执行 Action 任务，观察 `RUNNING`、feedback 和 `SUCCEEDED`；也可在运行时取消。
5. 点击“暂停续租”，观察租约倒计时归零，Mock Robot 收到 safe-stop，Agent 回到
   `SAFE_HOLD`。
6. 再次点击“获取控制权”，验证失联恢复后不会自动恢复旧控制会话。

页面中的命令列表显示命令 ID、状态、结果和更新时间；协议时间线显示
Browser/BFF/Agent 之间的重要消息。续租消息本身不会逐条写入时间线，以免 500 ms
节拍淹没安全与命令事件，但计数器和租约序号会持续更新。

## 仅查看前端效果

不启动 ROS 2 时运行：

```bash
PYTHONPATH=src/robot_agent:${PYTHONPATH} .venv/bin/python \
  tools/web_demo_server.py --open-browser
```

然后点击右上角“进入浏览器演示”。此模式会模拟握手、命令反馈和租约超时，并明确
显示 `BROWSER DEMO`；它不会连接 Robot Agent、写 SQLite 或调用 ROS 2，不能作为
真实链路验收证据。

## 常用参数

```text
--port 8765              HTTP、/ui 与 /agent 共用端口
--ros-domain-id 232      一键测试栈使用的隔离 ROS Domain
--agent-config PATH      Robot Agent 本地配置
--open-browser           启动后尝试打开默认浏览器
```

若更改端口、机器人 ID 或租约参数，必须同步修改 Agent 配置；启动器会在创建子进程前
检查这些值，避免以不一致策略开始测试。

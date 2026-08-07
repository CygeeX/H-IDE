# 验证矩阵

| 层级 | 内容 | 命令 |
|---|---|---|
| 纯核心 | 协议、配置、账本、仲裁、租约、队列 | `pytest tests -m 'not integration and not soak'` |
| 传输 | 双模式 mTLS、CA/SAN、帧、重连 | `pytest tests/test_transport.py` |
| ROS Jazzy | Service、Action、反馈/取消、Topic、心跳 | `pytest tests/test_ros_integration.py` |
| Humble | Python 3.10 与 ROS Humble 容器 | `tools/test_humble.sh` |
| 静态 | Ruff、mypy、ament/colcon | 见 README |
| soak | 30 分钟租约/状态/有界队列 | `pytest -m soak` |

验收记录应保存测试时间、ROS 发行版、Git commit、失败注入、峰值 RSS、队列最大深度、事件循环最大延迟和重复执行次数。任何 safe-stop 失败、命令重复下发、队列无界增长或退出阶段 ROS 销毁告警都视为失败。

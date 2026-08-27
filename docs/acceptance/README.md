# L0 / L1 / L2 验收标准

## 分级定义

| 等级 | 证明什么 | 不证明什么 |
| --- | --- | --- |
| L0 | Git 内容完整、无秘密/运行数据/gitlink、manifest/脚本可解析、路径可移植 | Docker、模型或机器人可运行 |
| L1 | Compose/场景配置有效，关键组件合同和单元测试通过 | GPU/ROS/SONIC live motion |
| L2 | 当前主机的 GPU、ROS、模型、Plant、SONIC、Nav2、Vision、Robot Agent 健康 | 未经执行的具体自然语言 Mission 一定成功 |

## 命令

```bash
./integration/acceptance.sh --level L0
./integration/acceptance.sh --level L1
./integration/start-e2e.sh --viewer
./integration/acceptance.sh --level L2
```

验收脚本不会默认发运动命令。原因不是保守，而是“健康探测”和“改变机器人状态”
是两种不同权限；自动 health check 不应凭空选择路线。

## L2 运动用例

先确保 PWA 显示 `g1-sim`、Office V2 scene binding 和 fresh pose/observation。
每个用例完成后保存 Mission ID、最终 pose、SONIC telemetry、Robot Agent 回执和
至少一份视觉证据。

1. 原地转向：左右各约 90°，终态为站立/idle，不出现持续平移。
2. 直行与停止：前进 3 m，允许合理误差，Mission 与 Robot Agent 均终态完成。
3. 横移：左右短距离横移，SONIC 持续 50 Hz，Plant boot/session 不变。
4. 静态绕行：绕 Office V2 柱子到目标站点，路径不得穿越占用 PGM。
5. 房间进入：从走廊进入一个办公室内部站点，不把“门口”误判成“房间内”。
6. 完整巡检：走遍所有办公室，汇总每个房间及路上观察；PWA 顺序、计划卡片、
   流式进度、视觉凭证和最终汇总必须属于同一个 Mission/turn。

## 失败判定

以下任一情况不能标记 L2/任务成功：

- PWA 显示完成但 Robot Agent/Nav2 没有终态完成回执。
- 观察请求直接复用了旧画面，而任务要求进入房间后观察。
- 对象候选有歧义却没有把视觉凭证和可选项展示给用户。
- Plant 或 SONIC boot/session 变化后仍继续旧 Mission。
- SONIC/model lock/scene hash 验证被跳过。
- 人工辅助带恢复被当作“未倒下”或成功运动证据。

## 发布门禁

合入 `main` 前至少满足：

- clean clone 的 L0、L1；
- 涉及 runtime/场景/SONIC/Nav2/Robot Agent 的变更有 L2 对应证据；
- `integration/integration.lock.yaml`、vendor manifests、第三方 notices 已更新；
- 新提交不包含 `.env`、数据库、日志、视觉证据、模型或 LFS 测试数据；
- 远端临时审计分支验证后，main 与不可变 release tag 指向同一提交。

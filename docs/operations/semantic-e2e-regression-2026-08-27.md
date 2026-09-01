# Semantic conversation and Office V2 regression run

日期：2026-08-27
范围：PWA → Assistant V2 → Mission V2 → Robot Agent → Nav2 → SONIC → MuJoCo → H-IDE Vision

## 验收目标

同一会话中的自然语言动作必须生成可审核的 Mission；视觉结果必须绑定到原始
turn/Mission；对象指代必须使用坐标无关的 `SemanticReferent`，随后重新用 RGB-D
生成可执行 `target_pose`。任何视觉候选都必须在 PWA 展示原图、编号框和候选 ID。

## 自动化覆盖

| 区域 | 覆盖内容 |
| --- | --- |
| `tests/test_semantic_referents.py` | 语义别名、证据绑定、Boot/Epoch 过滤、不可执行记忆 |
| `tests/test_object_semantic_navigation.py` | 椅子/架子/电脑解析、主动搜索、双观察点、DAAAM 先验边界、到达对准节点 |
| `tests/test_office_semantic_navigation.py` | 进入房间、室内验证、走出房间、入口/内部站点选择 |
| `tests/test_mission_v2_api.py` | 视觉→Nav2 连续 DAG、候选重试、搜索命中/未命中、队列与容差 |
| `tests/test_assistant_history_v2.py` | turnSequence、Mission 链接、失败投影和历史分页 |
| `apps/control-pwa/src/conversation-timeline.test.ts` | 同会话消息、证据和 Mission 卡片按轮次归属 |
| `apps/control-pwa/src/mission-visual-evidence.test.tsx` | 原图加载、编号框、裁剪证据和候选选择 |
| `apps/control-pwa/src/mission-progress.test.ts` | Mission 事件到流式进度文案 |
| `apps/control-pwa/src/main.test.tsx` | 拒绝旧卡片、队列提示、取消、审核和历史删除 |

## 本地静态回归命令

```bash
cd /home/phi/learn-projects/h-ide
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q

cd ../h-ide-agent
npm test -- --run
npm run typecheck
npm run build

cd ../h-ide/apps/control-pwa
npm test -- --run
npm run build

cd ../../../h-ide-mujoco-sonic
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_strict_navigation_adapter.py \
  tests/test_strict_motion_runner.py \
  tests/test_gait_adapter.py
```

## 真实仿真验收顺序

1. 用 `h-ide-mujoco-sonic/scripts/start-workstation-demo.sh --viewer` 冷启动；确认
   `workstation-demo-status.sh` 中 Plant、SONIC、Nav2、Robot Agent、Vision 全部为 `ok`。
2. 在一个新会话提交“去开放办公室里看一下有什么东西”，审核并执行。应到达房间、
   完成四向扫描并写入 `SemanticReferent`。
3. 同会话提交“去椅子旁边站着”。若出现多个候选，PWA 必须显示证据后再选择；选择后
   应重新分析、生成接近点、执行 `object-approach-align`，最后通过到达复核。
4. 提交“走出开放办公室”。应生成入口→走廊→室外验证三节点并完成。
5. 提交“走到这个架子前面”。首个视角未命中时应继续搜索，命中后执行接近和到达复核。
6. 提交“你再观察一下办公室找到电脑”。应执行主动搜索（必要时旋转），最后输出
   `object-search-report`，不得退化为只描述当前帧或返回通用“服务不可用”。

## 2026-08-27 实际结果

- 开放办公区观察：`completed`；四个视觉方向均产生 Qwen/SAM2 证据。
- 椅子：第一次产生五个候选；选择带 `analysisId` 的第一候选后，接近点导航成功，
  `object-approach-align` 实际发送约 2.14rad Spin；到达距离约 0.092m、偏航误差约
  0.089rad，`completed`。
- 走出开放办公室：入口、走廊和验证全部 `succeeded`，`completed`。
- 架子：第一视角 `target_not_visible` 后继续观察，第二视角生成可信接近点；距离约
  0.122m、偏航误差约 0.018rad，`completed`。
- 电脑主动搜索：候选证据展示并选择后重新分析，找到 `monitor`（电脑显示器）并形成
  可信接近点，`object-search-report.found=true`，`completed`。
- 全程 Plant/SONIC 支撑高度约 0.78m，无倒地、辅助带介入或安全故障。

## 已知运行边界

- 仿真 Plant 若因控制时钟跳变退出，其他桥接进程可能留下半拓扑；先按
  `docs/operations/workstation-integration-startup.md` 的依赖逆序停止并冷启动，不能把
  `insufficient_fresh_frames` 当成规划器错误。
- 室内到达容差为 0.35m，但房间多边形和最小 0.25m 清障距离仍必须通过；入口锚点仍为
  0.25m。
- DAAAM 只提供异步、坐标无关的区域搜索先验；任何导航前仍须新鲜 Qwen + RGB-D 几何
  复核。

## 场景动作模板与自然语言扩展

“场景动作模板”不是按用户句子逐条硬编码的脚本，而是一个由三层组成的
编译边界：

1. 语言解析层从句子提取动作、地点、相对方向和完成条件等槽位；同义词通过
   场景词汇表归一化（例如“办公室”和“开放办公室”都解析为 `open_office`）。
2. 场景绑定层只从已发布的 Office V2 区域和站点中选择身份，不接受模型或浏览器
   提供的坐标、路径或速度。
3. 动作编译层把槽位组合成复用的 Mission DAG 原语（入口导航、内部导航、区域
   验证、观察和报告）。Nav2 负责路径计算，Robot Agent/SONIC 负责执行。

因此，“走到尽头左转进入办公室”不会新增一条句子脚本；它会复用“进入地点”的
模板，并将通用“办公室”绑定到 Office V2 的开放办公区。若地点仍有多个合法解释，
系统应在审核卡片中展示候选，而不是把不受约束的文本交给 Mission planner。

## 回滚

回滚前停止完整仿真链路。当前对话开始前的源码/数据库/镜像基线保存在：

`/home/phi/learn-projects/.h-ide-backups/semantic-agent-baseline-20260827-ysPZdt`

该目录的 `MANIFEST.md` 记录了 Git bundle、数据库和镜像标签；不要删除或覆盖它。

本次验收后的运行镜像也已保留为：

- `robot-action-console-control-api:semantic-agent-e2e-stable-20260827`
- `robot-action-console-planning-agent:semantic-agent-e2e-stable-20260827`

本轮“通用办公室”词汇扩展部署后的控制 API/网关镜像另存为：

- `robot-action-console-control-api:semantic-route-vocabulary-20260827`
- `robot-action-console-robot-agent-gateway:semantic-route-vocabulary-20260827`

当前仿真核心仍使用既有稳定镜像；回滚应用层时只需将上述两个服务切换到对应标签，
不需要重置 Scene、PostgreSQL、Redis 或 MuJoCo 状态。

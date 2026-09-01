# Changelog

本文件记录 Skillstore 集成发行版的用户可见变化。组件内部提交历史仍保留，但只有通过
跨组件门禁并创建不可变 integration tag 的版本才是可复刻 Release。

## `skillstore-office-v2-e2e-20260901-r1` — Release candidate

目标标签：`integration/office-v2-e2e-20260901-r1`。标签在 clean-clone、L0/L1/L2/L3 和
显式端到端行为验收完成后创建。

### Added

- PWA 初始会话增加 `inspection.office` 与 `assistant.voice` 两个白名单 Skill。
- 巡检支持中文地点/别名、单房间和多房间顺序 Mission；每个房间都包含进入、内部验证、
  多方向扫描和最终按房间报告。
- 同会话巡检历史问答可引用报告中的对象、房间与视觉 Evidence。
- 语音助手使用浏览器 `zh-CN` Web Speech 朗读回复。
- 新增 simulation-only 挥手代理；只在空闲边界接纳，通过唯一 SONIC LowCmd 发布路径执行，
  完成后恢复正常手臂姿态。
- PWA 增加独立机器人选择模态框和三家公司展示卡；只有 Unitree 绑定 `g1-sim`。
- 新增 `full_straight_drive` opt-in profile：地图证明安全的直线段最多以 4 m 动作执行，
  不满足条件时自动回退。
- 新增商业化目标架构、PRO 6000 同机 Cloud/Site/Edge Demo、核心契约和 Semantic Scene
  Compiler 实施路线文档。

### Changed

- Office V2 移除走廊中央圆柱与可见吊灯，采用网格地板并增加公告板、白板、纸箱等可识别
  物件及无碰撞展示机器人。
- PWA 不再加载 GLB 头像；使用 CSS/DOM 助手形象，并将机器人选择器移出对话框主体。
- Mission/Conversation 在多轮巡检中保留地点顺序、已完成节点、报告与 Evidence 归属，
  避免把后续输入错误归入旧 Mission 卡片。
- Robot Agent 保持驻留 SONIC authority；自动故障不会由普通服务终止平衡策略，真实取消、
  状态与回执仍保留。
- Nav2/strict adapter 调整连续直线路径的分段与终态处理，保持官方 SONIC 速度，不使用
  自行猜测的步态速度。

### Fixed

- 修复双房间巡检被拆解不完整、到达后漏观察、重复走位和最终报告缺失的问题。
- 修复上一轮报告明明包含电脑/椅子，后续历史问答却回答“没有确认看到”的投影问题。
- 修复 PWA 会话历史、Mission 子项、拒绝卡片、视觉 Evidence 与后续 turn 的排序/归属。
- 修复 full-straight 的 Nav2 时间预算、边界到达和终态完成判定。
- 修复 Office V2 展示 G1 散架或离地，改为完整固定站立视觉层级并贴地放置。

### Current limitations

- 纯转向 SONIC reference 被证据门拒绝后，严格的 1 秒 planner deadline 仍会升级为
  terminal Plant fault，导致 MuJoCo 偶发退出；R1 按冻结决定保留这一当前本地行为，
  发生后需完整冷启动，详见发布说明。
- 当前可信房间、入口、观察站点和中文目录仍绑定 Office V2 发布 catalog；换场地尚不能
  自动从零生成可执行 SceneRevision。
- FARM 未进入运行链；DAAAM 仅为默认关闭的 Mission 后异步 importer，不能提供实时对象
  姿态。
- Field Executive、Cognition Broker、World Model、Semantic Scene Compiler 和新场景
  onboarding 是 Proposed，不是本候选的已实现能力。
- 仿真挥手不是 Unitree 真机动作能力；三家公司卡片中非 Unitree 选项仅用于展示。
- 云模型可受供应商欠费、限流、区域和同名模型更新影响。
- G1 网格、SONIC/GR00T 权重和 SAM2 checkpoint 必须由操作者按条款外置获取。

### Verification snapshot

- H-IDE PWA 定向回归：7 个 test files、112 项测试通过。
- H-IDE PWA 全量回归：17 个 test files、146 项测试通过。
- PWA `tsc --noEmit && vite build` 通过。
- 最终 Release 仍需记录 clean-clone L0/L1、L2 readiness、双 Skill 行为验收、镜像 digest、
  SBOM 和许可证/秘密扫描结果；本节不把候选测试误写成最终发布批准。

## `skillstore-office-v2-e2e-20260827-r3`

标签：`integration/office-v2-e2e-20260827-r3`。

- 冻结 Assistant/Mission V2、Qwen/SAM2、Robot Agent、Nav2、SONIC 与旧 Office V2
  联调基线。
- 修复 clean-clone 组件链接、外部资产链接父目录和 pytest 插件隔离。
- 保留为 2026-09-01 候选之前的不可变回退版本。

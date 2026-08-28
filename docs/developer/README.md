# H-IDE 开发者文档

本目录记录可复现的 H-IDE 联调与领导演示版本。先阅读：

- [领导演示：双 Skill、3D 头像与语音动作](leader-demo.md)
- [工作站联调启动手册](../operations/workstation-integration-startup.md)
- [连续 Mission 与仿真链路](../architecture/mission-v2-continuous-supervisor-worker.md)

## 当前实现边界

领导演示使用 `inspection.office` 与 `assistant.voice` 两个界面 Skill。Skill
选择只携带白名单 advisory 元数据，不携带坐标，也不授予执行权限。办公室
目标仍由 Office V2 场景目录解析，行动仍经过 Mission V2 审核与 Robot Agent →
Nav2 → SONIC → MuJoCo 链路。

`leader-demo` 的助手头像是会话/登录页的纯 CSS 展示组件，不是地图模型、3DGS 输入
或运动控制模型。地图审计/PLY/3DGS 设计应使用独立的 scene-explorer worker，不能
把演示头像当作场景事实。

## 分支与回退

- H-IDE：`feature/leader-demo-skills-3d`
- MuJoCo/SONIC：`feature/leader-demo-scene`
- 运动稳定基线：`stable/office-v2-e2e-20260825`

场景变更通过 `scenes/office_v2/scene.yaml` 生成；不要手工编辑
`generated/` 文件。回退时应切换到稳定分支并按联调手册依赖逆序停止、冷启动
整条仿真链路。

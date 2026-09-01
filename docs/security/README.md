# 秘密、许可证与供应链边界

本页区分当前 frozen demo release 的实际安全措施与商业化 Proposed 架构。仿真中的放宽
profile 不能复制到真机；Cloud/Edge mTLS、签名 MissionBundle、离线 Entitlement 等目标
设计尚未成为当前代码的完整实现。

## 1. 秘密管理

- 本地推荐秘密文件只有仓库根 `.env`，必须为当前用户拥有的普通文件且权限 `0600`。
- API key、Gitea PAT、内部 bearer token、管理员密码、数据库和视觉 Evidence 不得进入
  Git、remote URL、命令文档、截图、日志或 Release notes。
- 公开仓库匿名 clone 不需要 token；push 使用 credential helper/askpass 和最小权限、
  短期凭据。
- 任何曾出现在聊天、终端输出、URL 或不可信日志中的凭据都应立即撤销并重新生成，不能
  因仓库仍可访问就继续复用。
- 首个管理员使用 `h-ide-admin-bootstrap` 的交互式隐藏输入；不要使用命令参数或 `.env`
  保存密码，也不要复制原开发者 `control.db`。

`integration/preflight.sh` 只扫描当前树中的常见秘密和运行文件，不能替代 Gitea 端 push
protection、gitleaks/trufflehog 全历史扫描或人工许可证评审。

## 2. 内网 HTTP 边界

当前 Git remote 使用团队接受的内网 HTTP 地址。HTTP 没有传输加密，因此：

- 只能在受控局域网使用，不通过公网、访客 Wi-Fi 或不可信代理发送写凭据；
- 禁止 `http://user:token@host/...`，避免凭据进入 shell history、Git config 和代理日志；
- Runner、Webhook 与文档应使用团队确认的同一 canonical Gitea 地址；
- 真机、客户部署和跨站点通信仍应使用 HTTPS/mTLS、SSH 或受控 VPN；
- Git 使用 HTTP 不代表 Robot Agent、ROS DDS、SONIC、模型服务或管理 API 可以明文暴露。

## 3. 当前运行数据边界

当前 Git 树禁止：

- `.onnx`、`.pt/.pth`、`.safetensors` 和大模型 `.bin`；
- G1 网格、SONIC/GR00T 权重、SAM2 checkpoint；
- SQLite/PostgreSQL 数据、Redis dump、ROS bag、视频、Evidence、日志、core/crash；
- `.env`、证书私钥、PAT、API key 和内部 service token；
- ROS `build/install/log`、Python venv、`node_modules` 和 PWA `dist`。

模型和外部资产只记录来源、revision、大小、hash、operator-fetch 方法和许可证状态。
运行证据保存在受控本机/对象存储；公开验收只提交去敏的 manifest 与摘要。

## 4. 仓库和第三方许可证

根 `LICENSE` 是第一方内部协作条款，不覆盖组件或第三方许可证。总览见
`THIRD_PARTY_NOTICES.md`、`manifests/vendor/*.json` 和各目录 LICENSE。

- LeRobot source snapshot：Apache-2.0；当前树排除 test-only LFS artifacts。
- Unitree SDK2 Python snapshot：BSD-3-Clause。
- NVIDIA GR00T source：Apache-2.0；模型权重另受 NVIDIA Open Model License。
- G1 MuJoCo 网格：再分发状态未确认，只记录来源/revision/overlay，由操作者接受条款后获取。
- Office V2 Kenney 家具：CC0，资产 manifest 保留来源和 hash。
- 领导演示 PNG：仅按
  [`ASSET_LICENSE.md`](../../components/h-ide/apps/control-pwa/public/assets/leader-demo/ASSET_LICENSE.md)
  及 `asset-manifest.json` 的精确 hash 随 H-IDE 演示/UI 分发；不授权独立图片包销售，
  也不表示与展示卡公司存在背书或合作。
- DAAAM 上游本体为 BSD-3-Clause，但完整环境含 AGPL、非商业、专有 TensorRT 和数据集
  限制；本仓库不分发其 checkpoint/dataset。
- FARM 尚未集成；未来即使通过独立 Provider API 接入，也必须单独解决 AGPL、模型权重、
  数据集和商业服务义务。进程隔离本身不消除许可证责任。

公开可读不自动授予商用、品牌、模型或网格再分发权。新增或替换任何图片、网格、模型、
数据集、Viewer 或 Provider 前，必须完成来源、权利人、许可证、商用条件和出口边界审查。

## 5. Git 历史与当前树不是同一件事

本发行版的 L0 门禁检查当前 Git 树。仓库此前通过 subtree 保留 LeRobot 上游历史，历史对象
中可能仍出现已从当前树删除的测试模型、视频、bag 或 dataset 路径。因此：

1. “当前发行树不包含权重”不能写成“完整历史从未包含二进制”；
2. 正式公开 Release 前运行 full-history secret、large-blob、license 和 dataset 审计；
3. 若发现秘密，先撤销凭据，再决定受控历史净化；历史重写不能被当作撤销泄露；
4. 未经维护者批准不得 force-push 或静默重写公开 `main`/release tag；
5. Release notes 应说明审计范围是 current tree、full history 还是两者。

## 6. 模型和 Agent 权限

Current 边界：

- DeepSeek/h-ide-agent 生成意图和 Mission proposal，不持有 Robot Agent/ROS 凭据；
- Qwen/SAM2 生成语义、分割和几何候选，不直接发布导航目标；
- H-IDE 用 Scene revision、Robot Boot、Localization Epoch、fresh pose/evidence 和策略复核；
- Robot Agent 是唯一受限机器人协议出口；Nav2/SONIC 执行并反馈真实终态；
- 仿真挥手通过 allow-listed gesture proxy 进入唯一 SONIC LowCmd 发布路径；
- `full_straight_drive` 仍需地图证明、终点停止、取消和故障处理。

Proposed 的 Cognition Broker 可以在 Local、Site Brain 与 Cloud 模型之间升级复杂任务，但
模型输出仍只是带证据候选，不能获得速度或关节 authority。详见
[平台架构](../../components/h-ide/docs/platform-architecture/README.md)。

## 7. 仿真与真机隔离

- 当前 release 是带已知问题的 MuJoCo Office V2 冻结 Demo，不是 G1 真机安全认证。
- 仿真 collision/retry/failure profile 用于验证 Agent 编排，不能导出为 pilot/production
  配置。
- 真机必须重新验证硬件急停、速度、碰撞、停止距离、人员进入、网络、定位、平衡连续性
  和生产 mTLS。
- SONIC 平衡策略驻留不等于上层任务不可停止；取消、停止新动作和硬件安全路径必须始终
  可用且不可收费锁定。

## 8. 新依赖与发布审查清单

1. 记录上游 URL、完整 commit、许可证、权重/数据条款和 redistribution 状态。
2. 决定 source snapshot、patch、operator-fetch 或独立 Provider；不留下无 manifest gitlink。
3. 对源码树、patch、模型、网格和图片记录可重复 hash。
4. 更新 `THIRD_PARTY_NOTICES.md`、vendor manifest、integration lock 和资产许可证。
5. 在空缓存 clean clone 中验证失败信息、下载条款和安装命令。
6. 生成 CycloneDX/SPDX SBOM、容器 digest、漏洞与许可证报告作为 Release artifact。
7. 普通公开 PR Runner 不接触 GPU/ROS/机器人网络或供应商 secrets；L2/L3 只在可信、受
   保护的 self-hosted Runner 或人工审核环境运行。
8. 通过 L0/L1；涉及模型 authority、Mission、导航或运动时完成 L2 readiness 和 L3 显式
   行为验收。
9. 保护 `main` 和 `integration/*` 标签，禁止未经审核的 force-push。

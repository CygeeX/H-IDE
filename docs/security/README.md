# 秘密、许可证与供应链边界

## 秘密管理

- 唯一推荐的本地秘密文件是根目录 `.env`，权限必须为 `0600`。
- API key、Gitea PAT、内部 bearer token、数据库和证据不得出现在 Git、remote URL、
  命令文档、截图或日志中。
- Git 写权限通过本机 credential helper/askpass 提供；公开仓库的匿名 clone 不需要
  token，push 仍需要写权限。
- 任何在聊天、终端输出或 URL 中暴露过的 PAT/API key 都应在完成发布后轮换。

`integration/preflight.sh` 会扫描常见秘密/运行文件类型，但它不能替代服务端
secret scanning。建议远端开启 push protection，并在 CI 加入 gitleaks/trufflehog。

## HTTP 边界

本仓库 remote 使用用户明确接受的内网 HTTP 地址。HTTP 不提供传输加密，因此：

- 只在受控内网使用，不通过公网、访客 Wi-Fi 或不可信代理传输凭据。
- PAT 不写入 URL；避免 `http://user:token@host/...` 留在 shell history、Git config。
- 真机/PWA 生产部署仍应使用 HTTPS 或 WireGuard；“Git remote 是 HTTP”不授权把
  Robot Agent、ROS、DDS 或管理 API 暴露到明文网络。

## 仓库许可证

根 `LICENSE` 是本团队的内部协作条款，不会覆盖导入组件的原许可证。每个第三方
目录保留自己的 LICENSE；总览见 `THIRD_PARTY_NOTICES.md` 和
`manifests/vendor/*.json`。

- LeRobot snapshot：Apache-2.0，已删除 test-only LFS artifacts。
- Unitree SDK2 Python snapshot：BSD-3-Clause。
- NVIDIA GR00T source：Apache-2.0；模型权重另受 NVIDIA Open Model License。
- G1 MuJoCo 网格：redistribution 状态未确认，因此只保存来源、revision 和 patch，
  由操作者在接受条款后下载。
- DAAAM 上游本体为 BSD-3-Clause，但完整环境包含 AGPL、非商业、专有 TensorRT 和
  数据集限制；本仓库不再分发其 checkpoints/datasets。

## 模型和运行数据

Git 中禁止：`.onnx`、`.pt/.pth`、`.safetensors`、大模型 `.bin`、ROS bags、视频、
SQLite、运行日志、crash/core 和视觉证据。模型 manifest 记录 revision、大小和
SHA-256；bootstrap 下载到 operator cache，preflight/lock verifier 只读验证。

PWA 账号/会话历史和 Mission 数据不属于可复现源码。新开发者应执行正式 bootstrap/
admin 初始化流程创建自己的账户，不能复制原开发者的 `control.db`。

## 新依赖审查清单

1. 记录上游 URL、完整 commit、许可证和 redistribution 状态。
2. 决定 subtree、ordinary snapshot、patch 还是 operator-fetch；不能留下没有
   `.gitmodules` 的 gitlink。
3. 对普通 snapshot 计算稳定内容哈希；对大资产记录 SHA-256/大小。
4. 更新 `THIRD_PARTY_NOTICES.md`、vendor manifest 和 integration lock。
5. 在没有本机缓存的 clean clone 中验证失败信息和安装命令。
6. 通过 L0/L1，涉及运动或视觉 authority 时完成 L2。

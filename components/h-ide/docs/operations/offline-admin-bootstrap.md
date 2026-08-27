# 首个管理员离线引导

`h-ide-admin-bootstrap` 只用于一个已经完成数据库迁移、但 `users` 表仍为空的
新部署。它不启动 Web 服务、不自动创建或迁移数据库，也不会提升已有账户。请先
停止 Control API，并备份目标 SQLite 文件；命令在一个 `BEGIN IMMEDIATE` 事务中
完成空表检查、ADMIN 创建和无密码审计写入。并发运行时最多一个进程可以成功。

## 交互式方式（推荐）

安装项目后，在本机私有终端执行：

```bash
h-ide-admin-bootstrap \
  --database /absolute/path/to/control-plane.sqlite3 \
  --username administrator \
  --display-name "Administrator" \
  --site-id default
```

命令会通过 `getpass` 两次读取密码，密码不会显示，也不会成为进程参数或环境变量。
成功后再启动 Control API，并用该账户登录 `/app`；ADMIN/DEVELOPER 可进入
`/developer`，普通 USER 会被服务端 RBAC 拒绝。

## 0600 秘密文件方式

自动化安装可以临时准备仅含一行 UTF-8 密码的普通文件。文件必须由当前用户拥有，
权限必须**恰好**是 `0600`，且不能是符号链接：

```bash
chmod 0600 /secure/local/admin-password
h-ide-admin-bootstrap \
  --database /absolute/path/to/control-plane.sqlite3 \
  --username administrator \
  --secret-file /secure/local/admin-password
```

成功后应按部署方的秘密管理规范删除该临时文件。CLI 的审计只记录用户名、角色、
站点、凭据来源类型和时间；不会记录密码、密码哈希或秘密文件路径。

## 失败关闭规则

- 数据库不存在、不是普通文件、认证表结构不完整：拒绝，不创建新数据库。
- `users` 中已有任何账户（包括已有 ADMIN）：拒绝，不进行修改。
- 密码不满足现有 AuthService 的 12–1024 UTF-8 字节规则：拒绝。
- 审计写入失败：整个事务回滚，ADMIN 不会残留。
- 第二次运行：拒绝且不新增用户/审计；这是一条一次性命令，不是用户晋升工具。

已有的 `AUTH_BOOTSTRAP_ADMIN_PASSWORD` 环境变量启动方式仅为兼容旧部署保留，
新部署应视其为已弃用：环境明文可能被进程检查工具或容器元数据读取。不要同时配置
旧环境变量和本离线流程，也不要把密码放进命令行参数、shell 历史或 `.env` 文件。

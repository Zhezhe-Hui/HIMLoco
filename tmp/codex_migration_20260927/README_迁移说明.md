# Codex 会话迁移说明

本包保存线程“规划 Go2 视觉导航毕业课题”截至 2026-09-27 的本地记录和继续工作所需的关键文件。

## 包内容

- `codex_data/sessions/`：同一线程 ID 的 11 段原始 Codex JSONL 记录。
- `codex_data/session_index_entry.jsonl`：该线程的索引条目。
- `readable/会话文本_用户与助手.md`：仅保留用户和助手消息的可读版本。
- `继续会话提示.md`：在 Windows 新建 GPT/Codex 会话时可直接上传或粘贴。
- `project_snapshot/`：当前论文文档、关键规划代码、实验统计摘要及 Git 状态。
- `deliverables/`：最新的中期报告和技术路线图。
- `MANIFEST.sha256`：包内文件校验值。

## Windows 上最稳定的继续方式

1. 解压本文件。
2. 在 GPT 或 Codex 中新建会话。
3. 上传 `继续会话提示.md` 和需要继续编辑的文档。
4. 告诉新会话：“请按续接说明继续 HIMLoco 毕业论文任务。”

这种方式不依赖 Codex 的本地线程数据库版本，成功率最高。

## 尝试恢复 Codex 原窗口

Codex 会话记录是本地数据，使用同一账号并不保证自动同步。如果 Windows Codex 版本兼容，可以尝试：

1. 完全退出 Windows 上的 Codex。
2. 备份 `%USERPROFILE%\.codex` 目录。
3. 将 `codex_data/sessions/` 内容按原目录层级合并到 `%USERPROFILE%\.codex\sessions\`。
4. 将 `codex_data/session_index_entry.jsonl` 的内容追加到 `%USERPROFILE%\.codex\session_index.jsonl`，不要覆盖原索引。
5. 重新打开 Codex，查找“规划 Go2 视觉导航毕业课题”。

如果线程没有出现，不要替换 Windows 端的整个 `.codex` 目录，直接使用上面的“继续会话提示”方式。Linux 路径 `/home/hzz/project/HIMLoco` 在 Windows 上需换成实际项目路径。

## 隐私说明

本包不包含 `.codex/auth.json`、API 密钥、模型权重和完整原始仿真日志。原始会话记录仍包含本地路径和工具输出，请不要公开上传。

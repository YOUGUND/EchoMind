# 运行链路（仅 multi）

本仓库后端固定为 **JWT + PostgreSQL + Milvus** 一条链路（已不再支持 memory / Redis+Chroma full 分支）。

## 启动

```bash
# 1) 基础设施（Postgres + Milvus + MinIO + etcd）
docker compose -f docker-compose.multi.yml up -d

# 2) 环境变量（backend/.env.local）
# DATABASE_URL=postgresql+asyncpg://epochflow:epochflow@127.0.0.1:5432/epochflow
# MILVUS_URI=http://127.0.0.1:19530
# JWT_SECRET=change-me
# ANTHROPIC_API_KEY=...
```

启动后端后：

1. `POST /auth/register` 或前端「工作台设置 → 多用户登录」
2. 请求头带 `Authorization: Bearer <token>`
3. `/chat` 与 `/knowledge/*` 只读写当前用户数据

## 记忆

- **工作记忆**：PostgreSQL 最近消息  
- **摘要压缩**：轮次超阈值后 LLM/启发式摘要，旧消息归档  
- **情景召回**：归档片段按关键词轻量召回（压缩后找回细节）  
- **拼装预算**：`context_budget` 按 token 上限裁剪进模上下文  

## 隔离规则

- **身份**：只信任 JWT `sub`，忽略 body 里的 `user_id`
- **会话**：PostgreSQL `messages` / `conversations` 按 `user_id` 过滤
- **知识库**：Milvus（或进程内回退）写入带 `user_id`，检索 `user_id == current`

<div align="center">

# EchoMind

### 可观测、可评测的多 Agent 客服编排运行时

**理解请求 · 路由职责 · 执行工具 · 装箱上下文 · 评测结果**

[![Python](https://img.shields.io/badge/Python-3.12-0B1F33?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Vue](https://img.shields.io/badge/Vue-3-0E8F9A?style=flat-square&logo=vuedotjs&logoColor=white)](https://vuejs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Runtime-2EC4B6?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Author](https://img.shields.io/badge/GitHub-YOUGUND-0B1F33?style=flat-square)](https://github.com/YOUGUND)

</div>

## 项目简介

EchoMind 是面向客服与知识服务场景的 Agent 工程项目。它把一次请求拆成可观察的工程链路：

**意图识别 → 主辅 Agent 路由 → Skills / 知识注入 → 工具调用 → 上下文预算装箱 → 回复合成 → Trace / 评测**

重点不是“让模型多说一句话”，而是让系统具备清晰的角色边界、可追踪的执行过程、可恢复的运行时状态，以及可重复的质量验证。

## 能力地图

| 能力 | 解决的问题 |
| --- | --- |
| Intent & Routing | 从混合诉求中提取意图与实体，选择主 Agent 与辅助 Agent |
| Multi-Agent | 隔离角色提示词、输出契约、模型参数与工具白名单 |
| Skills & Tools | 按角色注入业务规则，在白名单内校验并执行工具 |
| Context Packing | 按 token 预算裁剪摘要 / 情景 / 政策 / 最近消息 |
| Knowledge | 文档切片、用户隔离检索，并按业务意图预取政策片段 |
| Observe & Evaluate | request span、工具日志、意图门禁与 LLM-as-Judge |

## 当前架构

```text
浏览器 (Vue 3)
  → FastAPI 请求边界 (JWT / 限流 / request_id)
  → LangGraph 编排 (clarify → route → execute → compose)
  → PostgreSQL 会话记忆 + 摘要压缩 + 情景召回
  → Milvus 用户隔离知识库
  → SiliconFlow 等模型网关
  → Trace / Eval 门禁
```

| 维度 | 实现 |
| --- | --- |
| 编排 | LangGraph + LangChain；Checkpointer 支持 interrupt / resume / history / fork |
| Agent | General / Technical / Billing + Escalation 交接 |
| 意图 | 约 19 类；LLM / Embedding / Pattern 三路融合 |
| 存储 | JWT + PostgreSQL + Milvus 单链路 |
| 前端 | Vue 3 工作台：对话、知识、评测、概览、模型中心、设置 |

## 技术栈

| 层次 | 技术 |
| --- | --- |
| 前端 | Vue 3、Vite、Lucide、Marked、DOMPurify |
| 后端 | Python 3.12、FastAPI、Pydantic、HTTPX |
| Agent | LangGraph、自研意图融合、Skills、工具网关 |
| 数据 | PostgreSQL、Milvus、Docker Compose |
| 测试 | Pytest、Playwright |

## 本地运行

环境要求：Python 3.12+、Node.js 22+、Docker。

```powershell
# 1. 基础设施
docker compose -f docker-compose.multi.yml up -d

# 2. 后端
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-free.txt
Copy-Item .env.example .env.local
# 填写 DATABASE_URL / MILVUS_URI / JWT_SECRET / ANTHROPIC_API_KEY 等

# 3. 前端
cd ..\frontend
npm ci

# 4. 一键启动（如仓库提供）
cd ..
python 启动开发.py
```

前端默认：`http://127.0.0.1:5174`  
后端默认：`http://127.0.0.1:8003`

多用户说明见 [`docs/multi-user.md`](docs/multi-user.md)。

## 关键代码

| 模块 | 路径 |
| --- | --- |
| FastAPI 主链路 | `backend/api/main.py` |
| LangGraph 编排 | `backend/agents/graph/` |
| 意图识别 | `backend/core/intent_recognizer.py` |
| 上下文装箱 | `backend/memory/cs_context.py` |
| PostgreSQL 记忆 | `backend/memory/pg_memory.py` |
| Milvus 知识库 | `backend/mcp/milvus_knowledge_base.py` |
| 前端工作台 | `frontend/src/App.vue` |

## 测试

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

覆盖范围包括：Agent Graph、恢复场景、工具韧性、上下文预算、多用户隔离、安全边界与评测门禁。

## 文档

- [文档中心](docs/README.md)
- [多用户运行链路](docs/multi-user.md)
- [架构与运行约定](docs/architecture.md)
- [LangGraph 改造说明](docs/langgraph-migration-plan.md)

## 作者

**YOUGUND** · [GitHub](https://github.com/YOUGUND)

实习 / 作品集用途：请以仓库实现与测试结果为准陈述能力，避免把实验室压测数字写成线上流量指标。

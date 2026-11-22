# EchoMind 文档中心

[返回 README](../README.md) · [English](../README_EN.md)

| 文档 | 内容 |
| --- | --- |
| [架构与运行约定](architecture.md) | 前后端边界、主链路、可观测性边界 |
| [项目导览](project-guide.md) | 定位、设计取舍、体验路线 |
| [多用户运行链路](multi-user.md) | JWT + PostgreSQL + Milvus |
| [模型中心](model-studio.md) | 模型目录与接口类别 |
| [LangGraph 改造计划](langgraph-migration-plan.md) | 编排迁移记录与验收 |
| [Phase 0 行为基线](langgraph-phase0-baseline.md) | 回归对照清单 |

## 推荐阅读

1. 先跑通一次混合诉求对话（登录失败 + 退款）
2. 读 `multi-user.md` 确认存储与身份边界
3. 对照 `backend/agents/graph/` 与 `backend/api/main.py` 看主链路

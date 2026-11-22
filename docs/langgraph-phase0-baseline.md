# Phase 0 行为基线用例

[返回改造计划](langgraph-migration-plan.md) · [文档中心](README.md)

> 用途：Phase 1+ 迁移时的行为锚点。优先保证**路由结构 / 工具权限 / API 字段**一致；最终回复文案允许与 legacy 不完全相同。

抽取来源：

- `backend/tests/test_agent_orchestrator.py`
- README / project-guide 在线体验路径
- `ChatResponse` 字段契约（`backend/api/main.py`）

---

## 1. API 契约基线（任何后端都必须满足）

`/chat` 成功响应至少包含：

| 字段 | 约束 |
| --- | --- |
| `request_id` | 非空字符串 |
| `intent` / `intent_group` / `intent_confidence` | 有值；confidence ∈ [0, 1] |
| `entities` | dict（可空） |
| `primary_agent` | `general` / `technical` / `billing` / `escalation` 之一 |
| `supporting_agents` | list[str] |
| `agent_types` | 非空；应覆盖 primary + supporting（按实现约定） |
| `routing_reason` | 非空，可读 |
| `routing_confidence` | number |
| `tools_used` | list[str] |
| `knowledge_used` | bool |
| `escalated` | bool |
| `latency_ms` | number ≥ 0 |
| `response` | 非空字符串 |

`/trace/tool/{request_id}`：在工具被调用后应能查到对应 `request_id` 的 trace。

---

## 2. 路由与协作基线

### B01 纯技术

- **输入**：`登录失败，出现 401 错误`
- **现有锚点**：`test_specific_intent_uses_local_fast_path_without_model_round_trip`
- **期望**：
  - intent 快路径为 `technical_login`（或同组 technical）
  - `primary_agent == technical`
  - `supporting_agents` 为空或不含 billing（无账单信号时）

### B02 纯账单

- **输入**：`我要退款，刚才被重复扣款了`
- **期望**：
  - intent 落在 billing / refund / payment_issue 组
  - `primary_agent == billing`
  - 无技术关键词时不应把 technical 当作 primary

### B03 复合主辅

- **输入**：`登录时报 401，同时这笔订单被重复扣款`
- **现有锚点**：`test_composite_request_routes_explicit_billing_signal_as_supporting_agent`
- **期望**：
  - `primary_agent == technical`
  - `supporting_agents == ["billing"]`（或等价只含 billing）
  - `multi_agent` 语义成立（API 上 supporting 非空 / agent_types 含两者）

### B04 低置信度澄清

- **输入**：较长但意图不明的 OTHER，且 `intent_confidence < 0.5`
- **现有逻辑**：`AgentOrchestrator._needs_clarification`
- **期望**：
  - 返回澄清话术（询问订单/退款/账户/技术等）
  - 不进入专属 Agent 工具循环
  - `routing_reason` 体现低置信度澄清

### B05 升级 / 转人工

- **输入 A**：intent=`human_handoff` 或 `escalation`
- **输入 B**：`urgency == CRITICAL`
- **现有锚点**：`test_escalation_agent_is_a_real_non_llm_handoff_node`
- **期望**：
  - primary 为 `escalation`，或 `escalated == true`
  - EscalationAgent 路径可不调用 LLM

### B06 紧急度覆盖

- **输入**：任意业务消息 + `urgency=CRITICAL`
- **期望**：路由到 escalation（与现网 `_route_decision` 一致）

---

## 3. 工具权限基线

### B07 角色工具隔离

- **现有锚点**：`test_agent_tool_scopes_are_real_and_isolated`
- **期望工具集合**：

| Agent | 私有工具 |
| --- | --- |
| general | `inspect_request_context`, `suggest_required_fields` |
| technical | `lookup_error_code`, `build_diagnostic_plan` |
| billing | `check_billing_fields`, `compare_amounts` |
| escalation | `create_handoff_summary` |

- 私有工具集合两两不相交

### B08 共享 RAG 工具

- **现有锚点**：`test_shared_rag_tool_is_available_to_all_agents`
- **期望**：四类 Agent 在注入 shared tools 后均可调用 `search_knowledge_base`

### B09 参数校验

- **现有锚点**：`test_tool_input_validation_rejects_unknown_fields`
- **期望**：未知字段（如 `secret`）被拒绝

### B10 白名单执行

- **现有锚点**：`test_tool_use_round_trip_executes_only_whitelisted_tool`
- **期望**：technical 仅暴露并执行其白名单工具；trace 记录工具名

---

## 4. Agent 契约基线（结构，不必比字符串）

### B11 Profile / Role Packet

- **现有锚点**：
  - `test_agent_profiles_have_distinct_contracts_and_generation_config`
  - `test_domain_agents_build_different_role_packets`
- **期望**：不同 Agent 的 role / workflow / tool_scope / role_packet 字段不同

### B12 Composer 降级

- **现有锚点**：`test_composer_fallback_preserves_primary_and_supporting_results`
- **期望**：合成模型失败时，仍保留主辅结果可读合并，不丢 supporting 内容

---

## 5. 体验路径冒烟（手工 / E2E）

对应 README 路线 A：

1. 对话实验室发送混合诉求（登录 + 扣款）
2. 执行面板可见：intent、primary/supporting、routing_reason、tools
3. （可选）知识空间导入后提问，检查 `knowledge_used` / 工具名

冒烟通过标准：页面不报错；面板字段齐全；路由符合 B03。

---

## 6. Phase 1 对比测试建议

过渡期对同一 fixture 跑 `legacy` 与 `langgraph`：

**必须一致**

- `primary_agent`
- `supporting_agents`（集合相等即可）
- `escalated`（升级场景）
- 工具白名单集合

**允许不一致**

- `response` 全文
- `latency_ms`
- `routing_reason` 措辞（但应仍含 primary / scores 等关键信息，推荐保持现格式）

---

## 7. 基线映射到现有测试

| 基线 ID | 现有测试 |
| --- | --- |
| B01 | `test_specific_intent_uses_local_fast_path_without_model_round_trip` |
| B03 | `test_composite_request_routes_explicit_billing_signal_as_supporting_agent` |
| B05 | `test_escalation_agent_is_a_real_non_llm_handoff_node` |
| B07 | `test_agent_tool_scopes_are_real_and_isolated` |
| B08 | `test_shared_rag_tool_is_available_to_all_agents` |
| B09 | `test_tool_input_validation_rejects_unknown_fields` |
| B10 | `test_tool_use_round_trip_executes_only_whitelisted_tool` |
| B11 | profile / role_packet 两个测试 |
| B12 | `test_composer_fallback_preserves_primary_and_supporting_results` |
| B02/B04/B06 | 现测覆盖不足 → Phase 1 需补 `test_agent_graph_baseline.py` |

---

## 8. Phase 0 冻结结论

- 基线以本节为准；Phase 1 合并前不得无说明地放宽 B03/B07/B10。
- 业务编排代码在 Phase 0 **不修改**；仅新增计划/基线/依赖草案与分支。

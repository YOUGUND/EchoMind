# LangGraph / LangChain 改造计划

[返回文档中心](README.md) · [项目导览](project-guide.md) · [架构约定](architecture.md)

> 状态：**全量迁移已完成（本地）** — 唯一编排后端为 LangGraph + LangChain；已删除 `agent_orchestrator.py`  
> 已具备：Checkpointer、interrupt 断点续送、history 回溯、fork 分叉  
> 基线清单：[langgraph-phase0-baseline.md](langgraph-phase0-baseline.md)  
> 依赖：已并入 `backend/requirements-free.txt`

---

## Phase 0 完成记录

| 项 | 结果 |
| --- | --- |
| 依赖选型 | `langgraph==1.2.20` + `langchain-core==1.7.6`（钉死 checkpoint/prebuilt）；`langchain-openai` Phase 2 再开 |
| Python | 需要 **≥3.10**（推荐 3.12，与 README 一致）；3.9 无法安装 langgraph |
| 与现依赖兼容 | 现有 `pydantic==2.10.3`、`httpx==0.28.1` 满足框架约束 |
| 基线用例 | 见 [phase0-baseline](langgraph-phase0-baseline.md)，含 B01–B12 与现有测试映射 |
| 文档中心入口 | `docs/README.md` 已挂改造计划 |
| 版本管理 | **不提交 git、不建分支**；本地直接改文件 |
| 业务代码（Phase 0） | 当时未改；Phase 1 已接开关与 Graph 目录 |

### Phase 1 完成记录

| 项 | 结果 |
| --- | --- |
| 目录 | `backend/agents/graph/`（state / nodes / build / facade / factory） |
| 开关 | `ORCHESTRATOR_BACKEND=legacy\|langgraph`（默认 legacy） |
| API | `api/main.py` 经 `create_orchestrator()` 创建；`/health` 增加 `orchestrator_backend` |
| 行为 | Graph 调度 clarify → route → execute → compose；Agent/工具仍复用 legacy `_execute` |
| 测试 | `tests/test_agent_graph.py` + 原 orchestrator 测试通过 |
| 运行环境 | 需 Python ≥3.10；本地可用 `backend/.venv311` |

### Phase 0 锁定的实施决策

1. **LLM 接入**：Phase 1 继续 Anthropic SDK + `model_scope`；Phase 2 再评估 ChatModel。
2. **并行协作**：Phase 1 节点内可先 `asyncio.gather`；之后再改原生并行。
3. **旧代码**：Phase 1–3 保持 `ORCHESTRATOR_BACKEND=legacy|langgraph` dual-run。
4. **存储路径**：先打 free/memory 主路径。

---

## 1. 改造动机

### 1.1 现状问题

当前编排核心在 `backend/agents/agent_orchestrator.py`（约 1100 行），自行实现：

- 意图识别调用
- 结构化路由（意图映射 / 领域打分 / 主辅协作）
- Agent 执行与工具循环
- 失败降级与升级判断
- 回复合成与 Trace 记录

这对理解系统内部很好，但对实习/求职叙事有两点不利：

1. **框架认同度不足**：面试官更熟悉 LangGraph / LangChain 的状态图、Tool、Runnable 抽象。
2. **边界难讲清**：项目看起来像“智能客服”，也像“模型实验室”，容易被理解成 Demo 堆砌。

### 1.2 改造后的一句话定位

> 基于 **LangGraph** 的多 Agent 客服编排运行时：用状态图表达意图识别、路由、主辅协作、工具调用与回复合成，并用 LangChain Tool / Retriever 承接工具与知识检索；保留自研意图融合、Skills、Trace 与评测闭环。

重点不是“换成框架”，而是：

- **图化编排**（可画、可测、可讲）
- **能力边界清晰**（角色工具白名单）
- **工程闭环保留**（可观测 + 可评测）

---

## 2. 目标与非目标

### 2.1 目标（必须做到）

| # | 目标 | 验收标准 |
| --- | --- | --- |
| G1 | `/chat` 主链路由 LangGraph 驱动 | 混合诉求（登录失败 + 退款）仍能主辅路由 |
| G2 | API 响应字段兼容现有前端 | `primary_agent` / `supporting_agents` / `routing_reason` / `tools_used` / `request_id` 等不变 |
| G3 | 工具白名单与参数校验仍生效 | 技术 Agent 不能调用账单专属工具 |
| G4 | Trace 面板可继续展示 | `/trace/tool/{request_id}` 仍能查到工具调用明细 |
| G5 | 关键回归测试通过 | `test_agent_orchestrator.py` 行为等价或迁移后重写并通过 |
| G6 | 简历可讲清迁移取舍 | 能说明为何用 LangGraph、保留了哪些自研能力 |

### 2.2 非目标（本轮不做）

- 不重写 Vue 前端信息架构
- 不强制把模型中心（Model Studio）迁到 LangChain
- 不引入付费向量库作为免费部署默认依赖
- 不把演示工具改成真实订单/退款系统
- 不追求一次删光全部自研编排代码（允许过渡期双实现）

---

## 3. 现状盘点（改造锚点）

### 3.1 主链路（当前）

```text
POST /chat
  → memory.get_context
  → IntentRecognizer.recognize
  → AgentOrchestrator.run
       → 低置信度澄清？
       → _route_decision（主辅）
       → 单 Agent._execute 或 run_parallel
       → BaseAgent.handle（LLM + tool_use 循环）
       → ResponseComposer.compose（多 Agent 时）
  → memory.add_message
  → ChatResponse（含路由与工具字段）
```

### 3.2 关键模块与职责

| 模块 | 路径 | 改造策略 |
| --- | --- | --- |
| API 入口 | `backend/api/main.py` | 保留；内部改调 Graph 适配层 |
| 编排器 | `backend/agents/agent_orchestrator.py` | **迁移核心**；最终由 Graph 替代 |
| 工具定义 | `backend/agents/tools.py` | 包装为 LangChain Tool，逻辑尽量复用 |
| 意图识别 | `backend/core/intent_recognizer.py` | **保留自研**；作为 Graph 一个节点 |
| Skills | `backend/skills/*` | **保留**；在 agent 节点注入 |
| 知识检索 | `backend/mcp/free_knowledge.py` 等 | Phase 2 用 Retriever/Tool 包装 |
| 模型隔离 | `backend/core/model_scope.py` | **保留**；与 Graph 调用兼容 |
| 评测 | eval 相关模块 | Phase 3 再接；继续打 Orchestrator/Graph 同一入口 |
| 前端面板 | `frontend/` | 原则上不动；依赖 API 字段稳定 |

### 3.3 必须兼容的对外契约

`ChatResponse` 至少继续提供：

- `request_id`
- `intent` / `intent_group` / `intent_confidence` / `entities`
- `primary_agent` / `supporting_agents` / `agent_types`
- `routing_reason` / `routing_confidence`
- `tools_used` / `knowledge_used` / `escalated` / `latency_ms`

改造期间：**前端零改或极少改** 是硬约束。

---

## 4. 目标架构

### 4.1 分层

```text
Vue 工作台
   ↓ HTTP
FastAPI（鉴权 / 限流 / request_id / model_scope）
   ↓
GraphFacade（兼容旧 Orchestrator 接口：run / recognize_intent / get_tool_trace）
   ↓
LangGraph StateGraph
   ├── intent_node          （复用 IntentRecognizer）
   ├── clarify_node         （低置信度追问）
   ├── route_node           （复用领域打分 / 主辅决策逻辑）
   ├── agent nodes          （general / technical / billing / escalation）
   ├── tools（LangChain）   （白名单工具 + RAG tool）
   ├── compose_node         （多 Agent 合并）
   └── finalize_node        （写 Trace、统计、升级标记）
   ↓
Memory / Skills / Eval（周边能力继续挂接）
```

### 4.2 LangGraph 状态设计（草案）

```python
class AgentGraphState(TypedDict, total=False):
    # 输入
    request_id: str
    user_id: str
    conv_id: str
    message: str
    history: list[dict]
    context: str

    # 意图
    intent: str
    intent_group: str
    urgency: str
    intent_confidence: float
    entities: dict
    source_scores: dict

    # 路由
    primary_agent: str
    supporting_agents: list[str]
    routing_reason: str
    routing_confidence: float
    needs_clarification: bool
    multi_agent: bool

    # 执行
    agent_responses: list[dict]   # {agent_type, content, tools_used, tool_traces, escalate, success}
    final_response: str
    escalated: bool
    tools_used: list[str]
    tool_traces: list[dict]
    latency_ms: float
```

说明：

- 状态字段尽量对齐现有 `Request` / `RoutingDecision` / `OrchestratorResult`，降低迁移成本。
- Trace 不依赖 LangGraph 内部私有结构，而是在 `finalize_node` 显式写入与现网一致的结构。

### 4.3 图拓扑（目标）

```text
START
  → intent
  → clarify_gate
       ├─(yes)→ clarify → END
       └─(no) → route
                 → dispatch_gate
                      ├─ single → agent_* → escalate_check → finalize → END
                      └─ multi  → [agent_a, agent_b] → compose → escalate_check → finalize → END
```

并行实现优先使用 LangGraph 的并行边 / `Send` API（以选定版本文档为准），行为对齐现有 `asyncio.gather` + `ResponseComposer`。

### 4.4 框架职责边界

| 能力 | 用什么 | 原因 |
| --- | --- | --- |
| 编排状态机 | **LangGraph** | 路由、分支、并行、可可视化 |
| 工具抽象 | **LangChain Tools** | 与生态一致，便于面试表达 |
| RAG 检索接口 | LangChain Retriever 包装自研检索 | 免费版仍可用 n-gram，不绑死向量库 |
| 意图识别 | **保留自研** | 已有三路融合与测试，不必为换而换 |
| Skills 热加载 | **保留自研** | 文件化业务规则，与框架无关 |
| 请求级模型隔离 | **保留** `ContextVar` | 现有并发隔离是亮点 |
| LLM-as-Judge 评测 | **保留**，入口改打 Graph | 评测闭环是项目差异点 |
| HTTP / 前端 | FastAPI + Vue | 不动产品壳 |

---

## 5. 分阶段实施计划

### Phase 0：准备（0.5–1 天） — ✅ 已完成

**已完成**

1. 依赖草案写入 `backend/requirements-langgraph.txt`（**未**合并进 `requirements-free.txt`）
2. 版本管理：按本地约定不提交 git、不建功能分支
3. 基线用例：`docs/langgraph-phase0-baseline.md`
4. 文档中心已挂改造计划

**基线用例（最少，详见 baseline 文档）**

| 用例 | 期望 |
| --- | --- |
| 纯技术：登录失败 / 401 | primary=technical |
| 纯账单：退款 / 重复扣款 | primary=billing |
| 复合：登录失败 + 退款 | multi-agent，supporting 非空 |
| 低置信度模糊问题 | 返回澄清，不乱路由 |
| CRITICAL / 转人工 | escalated 或 escalation agent |
| 工具白名单 | technical 看不到退款执行类工具 |

**退出标准**：基线清单确认；依赖选型确认；不改业务代码。 → **已满足**

---

### Phase 1：LangGraph MVP（3–5 天） — ✅ 已完成（本地）

**目标**：`/chat` 可通过开关走 Graph，行为基本等价。

**已落地**

```text
backend/agents/graph/
  state.py
  factory.py
  build.py
  facade.py
  nodes/routing.py
  nodes/execute.py
```

- `GraphFacade` 兼容 `recognize_intent` / `run` / trace / stats / skills / shared tools
- `ORCHESTRATOR_BACKEND=legacy|langgraph`（默认 legacy）
- 路由复用 `_route_decision`；执行复用 `_execute` + `ResponseComposer`
- 测试：`tests/test_agent_graph.py`

**明确仍未做（留给 Phase 2+）**

- 不重写工具循环为 LangChain ToolNode
- 不删 `agent_orchestrator.py`

**退出标准**

- `ORCHESTRATOR_BACKEND=langgraph` 时，基线路由/澄清/主辅协作单测通过 ✅
- legacy 开关仍可用 ✅
- 前端字段契约未改（依赖 Facade 返回 `OrchestratorResult`） ✅

---

### Phase 2：Tools / RAG 主流化（3–5 天）

**目标**：工具层变成 LangChain Tool，RAG 以 Retriever/Tool 形式接入。

**做什么**

1. 将 `AgentToolSpec` 适配为 LangChain `StructuredTool`
2. 每个 Agent 节点只绑定自己的 tool 白名单（对应现有 `tool_scope`）
3. 用 LangGraph tool-calling 循环（或 `ToolNode`）替换 `BaseAgent` 内部手写 tool_use 循环
4. `search_knowledge_base` 包装为 Tool；免费检索实现仍用 `free_knowledge`
5. 统一 `tool_traces` 写入格式，保证 `/trace/tool/{id}` 不变

**风险点**

- Anthropic Messages 与 OpenAI tool 协议差异：继续复用 `model_scope` / `llm_utils`，或改为 LangChain 统一 ChatModel 抽象
- 需要回归“模型中心选模型是否影响 Agent 节点”

**退出标准**

- Agent 工具循环不再依赖手写 Anthropic content block 解析（或仅留兼容层）
- 工具权限测试通过
- 知识工具命中时 `knowledge_used=true`

---

### Phase 3：可观测与评测对齐（2–3 天）

**目标**：评测与监控打到同一 Graph 入口，简历可讲“闭环”。

**做什么**

1. `/eval/run` 改为调用 `GraphFacade.run`
2. 如有 Monitor / routing penalty，定义如何回写 Graph 路由节点（可先兼容旧 `update_routing_penalties`）
3. 补充图结构说明图（文档 + 可选 Mermaid 导出）
4. 增加若干“迁移回归”测试：legacy vs langgraph 对同一输入的路由字段对比（允许文案不完全一致，路由结构需一致）

**退出标准**

- 评测链路绿
- 文档可展示 StateGraph 拓扑
- 对比测试覆盖主辅路由与澄清分支

---

### Phase 4：收敛与叙事收尾（1–2 天）

**做什么**

1. 默认切换 `ORCHESTRATOR_BACKEND=langgraph`
2. 将旧编排器标为 deprecated，或移到 `backend/agents/legacy/`
3. 更新 README / project-guide：技术栈增加 LangGraph / LangChain
4. 更新 `backend/wiki/项目表达.md` 简历话术
5. （可选）删除死代码；保留最小兼容层直到确认无回归

**退出标准**

- 默认路径全走 Graph
- README 与简历描述一致
- CI 全绿

---

## 6. 建议目录与接口草图

### 6.1 新文件

| 文件 | 职责 |
| --- | --- |
| `agents/graph/state.py` | 状态类型 |
| `agents/graph/build.py` | `build_agent_graph()` |
| `agents/graph/facade.py` | 对 API 的稳定门面 |
| `agents/graph/nodes/*.py` | 各节点 |
| `agents/graph/tools_adapter.py` | AgentToolSpec ↔ LangChain Tool |
| `tests/test_agent_graph.py` | Graph 专用测试 |

### 6.2 API 侧改动原则

```text
main.py 只依赖 Facade，不直接依赖 StateGraph 细节
ChatRequest / ChatResponse schema 不因为迁移而改名改义
```

### 6.3 配置项

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `ORCHESTRATOR_BACKEND` | `legacy` / `langgraph` | Phase1–3: `legacy`；Phase4: `langgraph` |
| `LANGGRAPH_DEBUG` | 是否输出节点级调试日志 | `false` |

---

## 7. 测试策略

### 7.1 分层

1. **节点单测**：route / clarify / domain score（可无 LLM）
2. **Graph 集成测**：mock LLM，断言路由字段与工具名
3. **API 契约测**：`/chat` JSON 字段齐全
4. **对比测**（过渡期）：同一 fixture 下 legacy vs graph 的 `primary_agent` / `supporting_agents` 一致
5. **E2E**（可选）：Playwright 仍走对话实验室混合诉求

### 7.2 优先迁移的现有测试

- `backend/tests/test_agent_orchestrator.py` → 抽公共断言，双后端复用
- 涉及 orchestrator 初始化的 `main` / free profile 测试需支持 backend 开关

### 7.3 不做的测试承诺

- 不宣称线上准确率、QPS；只保证行为契约与回归用例。

---

## 8. 风险与回滚

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 框架版本 API 变动 | 构建失败 / 写法过时 | 锁定版本；封装 `build.py` |
| 工具协议不兼容 | 工具调失败 | Phase1 先复用 BaseAgent；Phase2 再切 |
| 并行节点语义不一致 | 主辅协作结果漂移 | 对比测试 + 先搬运 composer |
| 依赖体积变大 | Render 冷启动变慢 | 免费配置只装必要包；观察构建时长 |
| 过度重写导致不可演示 | 实习展示中断 | 功能开关 + 分阶段 |
| “只是套壳”质疑 | 面试减分 | 保留自研意图/Skills/评测，并能讲清迁移点 |

**回滚策略**：任何阶段都可通过 `ORCHESTRATOR_BACKEND=legacy` 回到旧编排器；Phase4 前不删除旧文件。

---

## 9. 工作量与里程碑（建议）

| 阶段 | 工期 | 可对外展示的成果 |
| --- | --- | --- |
| Phase 0 | 0.5–1 天 | 计划确认、基线用例 |
| Phase 1 | 3–5 天 | Graph 驱动 `/chat`，开关可切 |
| Phase 2 | 3–5 天 | LangChain Tools + RAG Tool |
| Phase 3 | 2–3 天 | 评测/监控对齐，拓扑可讲 |
| Phase 4 | 1–2 天 | 默认切换，文档与简历更新 |
| **合计** | **约 10–16 天** | 视每日投入与 LLM 联调情况浮动 |

---

## 10. 实习叙事（改造完成后怎么讲）

### 10.1 推荐项目描述（简历）

```text
EpochFlow 多 Agent 编排运行时（LangGraph）
技术栈：Python / FastAPI / LangGraph / LangChain / Vue3 / SiliconFlow

- 将自研 Orchestrator 迁移为 LangGraph StateGraph，用节点表达意图识别、结构化路由、
  主辅 Agent 并行、工具调用与回复合成，并通过功能开关保证平滑过渡。
- 使用 LangChain Tool 统一工具协议，按 Agent 角色绑定白名单，保留参数校验与执行 Trace。
- 复用自研三路意图融合、动态 Skills 与 LLM-as-Judge 评测，形成“路由可观察、结果可回归”的闭环。
```

### 10.2 面试高频问答预案

**Q: 为什么原来不用 LangGraph？**  
A: 先用自研实现验证路由/权限/评测是否成立；确认链路后迁移到主流编排框架，降低后续扩展成本。

**Q: 为什么不全盘 LangChain？**  
A: 编排用 LangGraph，工具用 LangChain；意图融合、Skills、评测是项目差异化能力，不必为换框架而换。

**Q: 和直接套 LangChain Agent 模板有什么区别？**  
A: 有结构化主辅路由、工具权限隔离、请求级模型隔离、Trace 契约和评测回归，不只是单 Agent ReAct。

---

## 11. 决策清单（开工前确认）

开始写代码前，建议先确认以下选择：

1. **LLM 接入方式**
   - A. LangChain ChatModel 统一封装（更主流）
   - B. 继续 Anthropic SDK + 自研协议转换（更少改动）  
   **建议 Phase1 用 B，Phase2 评估切 A。**

2. **并行协作实现**
   - A. LangGraph 原生并行 / `Send`
   - B. 节点内 `asyncio.gather`（过渡）  
   **建议 Phase1 可用 B，Phase3 前改为 A。**

3. **旧代码命运**
   - A. 长期 dual-run  
   - B. Phase4 后删除 legacy  
   **建议实习展示期保留 dual-run 至少一段时间。**

4. **是否同时改完整版 Redis/Chroma 路径**
   - **建议先打 free/memory 主路径**，完整存储模式后置。

---

## 12. 下一步

1. ~~Phase 0~~ ✅  
2. ~~Phase 1 MVP~~ ✅  
3. ~~全量迁移：LangChain Tools + Checkpointer + interrupt/resume + fork + 删除 legacy orchestrator~~ ✅  

**运行：**

```powershell
cd backend
.\.venv311\Scripts\python.exe -m pip install -r requirements-free.txt
# 可选：CHECKPOINT_BACKEND=sqlite
```

**新 API：**

- `POST /chat` — 可能返回 `interrupted=true`
- `POST /chat/resume` — 断点续送
- `GET /chat/threads/{thread_id}/history` — 回溯
- `POST /chat/fork` — 从 checkpoint 分叉

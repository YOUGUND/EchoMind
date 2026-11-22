"""
EpochFlow Agent 学习工作台系统 — FastAPI 入口

所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from api.auth import multi_user_enabled, require_user_if_multi, router as auth_router
from api.schemas import (ChatRequest, ChatResponse, ResumeRequest, ForkRequest,
                         ToolTraceResponse, RecentToolTracesResponse,
                         DocInput, BatchDocInput, EvalIntentInput, EvalDialogInput, EvalRunInput)
from api.user_context import AuthUser, bind_request_user

load_dotenv(pathlib.Path(_ROOT) / ".env.local")
load_dotenv(pathlib.Path(_ROOT) / ".env")

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = "EpochFlow v1.1 — Agent Engineering Studio"

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None

from api.risk_guard import default_risk_guard
from api.security import RequestBoundary, model_configured, public_demo_enabled, security_audit
from api.workspace import router as workspace_router
from api.models import router as models_router
from core.model_scope import agent_model_scope, model_errors


def _storage_mode() -> str:
    """Product path is multi-only (PostgreSQL + Milvus)."""
    return "multi"


def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not model_configured():
        key = "not-configured-local-client"  # Never sent: model routes are guarded.

    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager

    print(BANNER, flush=True)

    from agents.contracts import Request
    from agents.graph.factory import create_orchestrator
    from agents.tool_registry import register_all_agent_tools
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from memory.pg_memory import MemoryManager
    from mcp.milvus_knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from storage.db import init_db

    await init_db()
    logger.info("运行链路：multi（PostgreSQL 会话记忆 + Milvus 知识库，按 user_id 隔离）")

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")
    logger.info("编排后端: langgraph+langchain (checkpointer=%s)", os.getenv("CHECKPOINT_BACKEND", "memory"))

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # LangGraph + LangChain 编排器（含 checkpointer / interrupt / fork）
    _orchestrator = create_orchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    _memory = MemoryManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(milvus_uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    app.state.knowledge = kb
    app.state.tool_manager = _tool_manager
    logger.info("知识库后端: %s（按 user_id 隔离）", getattr(kb, "_backend", "milvus"))

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的检索。请稍后重试。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索当前用户知识库（Milvus 向量检索，按 user_id 隔离）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))
    # 全部角色工具也注册进同一网关：校验 / 缓存 / 熔断 / 统计
    register_all_agent_tools(_tool_manager)
    if _orchestrator is not None:
        _orchestrator.set_tool_manager(_tool_manager)

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    from evaluation.store import baseline_path as eval_baseline_path

    # 评测中台：默认把 baseline 落到 data/eval/baselines/current.json（可用 ENV 覆盖）
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=str(eval_baseline_path()),
    )

    logger.info("EpochFlow 已就绪")
    try:
        yield
    finally:
        await _monitor.stop()
        if _memory is not None:
            await _memory.close()
        from storage.db import close_db
        await close_db()
        clients = [
            recognizer.client,
            getattr(_orchestrator, "_intent_recognizer", None) and _orchestrator._intent_recognizer.client,
            _tool_manager._client,
            _evaluator._judge._client,
        ]
        for client in {id(client): client for client in clients if client is not None}.values():
            await client.close()
        logger.info("EpochFlow 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EpochFlow 多 Agent 工程平台",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(RequestBoundary)
app.include_router(auth_router)
app.include_router(workspace_router)
app.include_router(models_router)


@app.exception_handler(ValueError)
async def invalid_input(request, error):
    return JSONResponse(status_code=400, content={"detail": str(error)})


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",") if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.suite_loader import list_suites
    from evaluation.store import list_runs

    return {"status": "ok", "model_configured": model_configured(),
            "public_demo": public_demo_enabled(),
            "storage_mode": "multi",
            "auth": "jwt",
            "eval_platform": {
                "suites": len(list_suites()),
                "recent_runs": len(list_runs(limit=20)),
            },
            "observability": {
                "request_trace": True,
                "monitor": _monitor is not None,
                "security_audit": True,
            },
            "persistent": True,
            "retrieval": "milvus",
            "memory": {
                "working": "postgresql",
                "summary_compress": True,
                "episodic": "postgresql_keyword",
                "context_budget": True,
            },
            "orchestrator_backend": "langgraph",
            "checkpoint_backend": os.getenv("CHECKPOINT_BACKEND", "memory"),
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: Optional[AuthUser] = Depends(require_user_if_multi)):
    async with agent_model_scope(req):
        with bind_request_user(user):
            return await _chat_with_model(req, user=user)


async def _chat_with_model(req: ChatRequest, user: Optional[AuthUser] = None):
    """
    主对话接口。完整流程：
      风控检查 → 记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    multi 模式下 user_id 只信任 JWT，忽略 body.user_id。
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    risk = default_risk_guard.assess_message(req.message)
    if not risk.allowed:
        security_audit.record(
            code=risk.code,
            status=400,
            path="/chat",
            request_id=uuid.uuid4().hex,
            detail=risk.reason,
        )
        raise HTTPException(
            status_code=400,
            detail=f"[{risk.code}] {risk.reason}",
            headers={"X-Risk-Code": risk.code},
        )

    from agents.contracts import Request as OrcReq
    from memory.models import MsgRole

    # Isolation boundary: token subject wins over client-supplied user_id.
    user_id = user.id if user is not None else req.user_id
    conv_id = req.conv_id or str(uuid.uuid4())
    thread_id = req.thread_id or conv_id

    if multi_user_enabled():
        kb = getattr(getattr(app.state, "knowledge", None), "ensure_default_docs_async", None)
        if kb is not None:
            await kb(user_id)

    try:
        mem_ctx = await _memory.get_context(user_id, conv_id, query=req.message)
    except PermissionError as ex:
        raise HTTPException(403, str(ex)) from ex

    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    intent_result = await _orchestrator.recognize_intent(req.message, history=history)

    # Customer-service context pack: intent/entity gated layers + token budget
    # (not a generic code-agent prompt compressor).
    from memory.cs_context import pack_customer_context, pack_input_from_request

    policy_snippets: list[str] = []
    kb_obj = getattr(app.state, "knowledge", None)
    business = (intent_result.intent_group or "") in {
        "billing", "technical", "account", "logistics", "request", "complaint",
    } or (intent_result.intent and intent_result.intent.value not in {
        "greeting", "feedback", "other", "human_handoff", "escalation",
    })
    if business and kb_obj is not None and hasattr(kb_obj, "search_async"):
        try:
            hits = await kb_obj.search_async(req.message, top_k=3, user_id=user_id)
            for hit in hits or []:
                title = hit.get("title") or ""
                content = hit.get("content") or ""
                snippet = f"{title}: {content}".strip(": ").strip()
                if snippet:
                    policy_snippets.append(snippet[:500])
        except Exception as ex:
            logger.warning("context policy prefetch failed: %s", ex)

    pack_inp = pack_input_from_request(
        user_message=req.message,
        intent=intent_result.intent,
        intent_group=intent_result.intent_group,
        urgency=intent_result.urgency,
        confidence=intent_result.confidence,
        entities=intent_result.entities,
        memory=mem_ctx,
        policy_snippets=policy_snippets,
        agent_role="general",
        mode="normal",
    )
    full_context, budget_report = pack_customer_context(pack_inp)
    if budget_report.truncated or budget_report.saved_tokens:
        logger.info(
            "cs_context_budget saved=%s before=%s after=%s ratio=%.2f%% sections=%s",
            budget_report.saved_tokens,
            budget_report.before_tokens,
            budget_report.after_tokens,
            budget_report.compression_ratio * 100,
            budget_report.sections,
        )

    orch_req = OrcReq(
        message=req.message,
        user_id=user_id,
        conv_id=conv_id,
        context=full_context,
        history=history,
        entities=intent_result.entities,
        intent=intent_result.intent,
        intent_group=intent_result.intent_group,
        urgency=intent_result.urgency,
        intent_confidence=intent_result.confidence,
        thread_id=thread_id,
    )

    result = await _orchestrator.run(orch_req)
    if model_errors.get():
        raise model_errors.get()[0]

    await _memory.add_message(user_id, conv_id, MsgRole.USER, req.message)
    await _memory.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

    return ChatResponse(
        model=req.model or os.getenv("ANTHROPIC_MODEL", ""),
        conv_id=conv_id,
        request_id=result.request_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        intent_group=intent_result.intent_group,
        agent_type=result.agent_type.value,
        agent_types=[agent_type.value for agent_type in result.agent_types],
        primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,
        supporting_agents=[agent_type.value for agent_type in result.supporting_agents],
        tools_used=result.tools_used,
        routing_reason=result.routing_reason,
        routing_confidence=result.routing_confidence,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used="search_knowledge_base" in result.tools_used,
        entities=intent_result.entities,
        intent_confidence=round(intent_result.confidence, 4),
        intent_source_scores=intent_result.source_scores,
        thread_id=result.thread_id or thread_id,
        checkpoint_id=result.checkpoint_id,
        interrupted=result.interrupted,
        interrupt_payload=result.interrupt_payload or {},
    )


@app.post("/chat/resume", response_model=ChatResponse, tags=["Checkpoint"])
async def chat_resume(
    req: ResumeRequest,
    user: Optional[AuthUser] = Depends(require_user_if_multi),
):
    """断点续送：向 clarify / escalation 中断点注入输入并继续。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    with bind_request_user(user):
        result = await _orchestrator.resume(req.thread_id, req.resume, request_id=req.request_id)
    return ChatResponse(
        model=os.getenv("ANTHROPIC_MODEL", ""),
        conv_id=result.thread_id or req.thread_id,
        request_id=result.request_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        agent_type=result.agent_type.value,
        agent_types=[agent_type.value for agent_type in result.agent_types],
        primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,
        supporting_agents=[agent_type.value for agent_type in result.supporting_agents],
        tools_used=result.tools_used,
        routing_reason=result.routing_reason,
        routing_confidence=result.routing_confidence,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used="search_knowledge_base" in result.tools_used,
        thread_id=result.thread_id or req.thread_id,
        checkpoint_id=result.checkpoint_id,
        interrupted=result.interrupted,
        interrupt_payload=result.interrupt_payload or {},
    )


@app.get("/chat/threads/{thread_id}/history", tags=["Checkpoint"])
async def chat_history(thread_id: str, limit: int = 20):
    """回溯：列出该 thread 的 checkpoint 历史。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"thread_id": thread_id, "items": await _orchestrator.history(thread_id, limit=limit)}


@app.post("/chat/fork", tags=["Checkpoint"])
async def chat_fork(req: ForkRequest):
    """分叉：从历史 checkpoint 复制到新 thread，可继续对话。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    try:
        return await _orchestrator.fork(
            req.source_thread_id,
            req.checkpoint_id,
            new_thread_id=req.new_thread_id,
        )
    except ValueError as ex:
        raise HTTPException(404, str(ex)) from ex


async def _build_knowledge_context(message: str, intent=None, top_k: int = 3) -> tuple[str, bool]:
    """
    为 /chat 主链路构建 RAG 知识上下文。

    这里复用 MCPToolManager 的查询改写、并行召回、重排、fallback 能力。
    """
    if _tool_manager is None:
        return "", False
    if not _should_use_knowledge(message, intent=intent):
        return "", False
    try:
        result = await _tool_manager.search_with_rewrite("knowledge_search", message, top_k=top_k)
        if not result.success or not isinstance(result.data, list) or not result.data:
            return "", False

        parts = ["[知识库检索结果]"]
        used = False
        for i, item in enumerate(result.data[:top_k], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("content", "")).strip()
            score = item.get("score", "")
            if not content:
                continue
            used = True
            parts.append(f"{i}. 标题: {title}\n   相关度: {score}\n   内容: {content[:600]}")

        if not used:
            return "", False
        parts.append("请优先依据以上知识库内容回答；如果知识库内容不足，再结合通用客服能力说明。")
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False


def _should_use_knowledge(message: str, intent=None) -> bool:
    """跳过纯寒暄，业务类问题才检索知识库，避免无关 RAG 干扰回复。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    intent_value = getattr(intent, "value", intent)
    if intent_value in {"greeting", "feedback", "escalation", "human_handoff", "other"}:
        return False
    if intent_value in {
        "query", "request", "technical", "billing", "account", "complaint",
        "order_status", "logistics", "refund", "invoice", "payment_issue",
        "account_security", "technical_login", "technical_crash",
    }:
        return True
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}
    if msg in greetings:
        return False
    business_keywords = [
        "退款", "订单", "物流", "配送", "发票", "扣款", "支付", "账单", "订阅",
        "登录", "报错", "错误", "崩溃", "会员", "积分", "账户", "密码", "地址",
        "refund", "order", "invoice", "payment", "error", "login",
    ]
    return len(msg) >= 4 or any(kw in msg for kw in business_keywords)


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议、安全审计。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    payload = _monitor.summary()
    payload["security"] = security_audit.summary()
    return payload


@app.get("/security/summary")
async def security_summary():
    """鉴权 / 限流 / 风控拦截摘要（演示级审计）。"""
    return security_audit.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    """查看某次请求的工具调用明细（含 spans / token / tool_tree）。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_request_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/request/{request_id}", response_model=ToolTraceResponse)
async def get_request_trace(request_id: str):
    """请求级可回放 trace：intent→route→execute→compose spans + tokens。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_request_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = 20):
    """查看最近 N 次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_request_traces(limit=limit))

@app.get("/trace/requests", response_model=RecentToolTracesResponse)
async def list_recent_request_traces(limit: int = 20):
    """查看最近 N 次请求级 trace。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_request_traces(limit=limit))


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(
    query: str,
    top_k: int = 5,
    user: Optional[AuthUser] = Depends(require_user_if_multi),
):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    multi 模式下只检索当前登录用户的知识库。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    with bind_request_user(user):
        kb = _tool_manager._tools["knowledge_search"].handler.__self__
        return {
            "query": query,
            "results": await kb.search_async(query, top_k),
            "reranked": False,
            "retrieval": "milvus",
            "user_id": user.id if user else None,
        }


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(
    body: BatchDocInput,
    user: Optional[AuthUser] = Depends(require_user_if_multi),
):
    """批量导入文档。multi 模式下写入当前用户私有空间。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    with bind_request_user(user):
        docs = [{"title": d.title, "content": d.content} for d in body.documents]
        if multi_user_enabled() and user is not None:
            count = await kb.add_documents_async(docs, user_id=user.id)
            total = await kb.doc_count_async(user_id=user.id)
        else:
            count = await kb.add_documents_async(docs)
            total = await kb.doc_count_async()
    _tool_manager.clear_cache()
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": total}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(
    file: UploadFile = File(...),
    user: Optional[AuthUser] = Depends(require_user_if_multi),
):
    """上传文件导入知识库（TXT/MD/JSON）。multi 模式写入当前用户私有空间。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024:
        raise HTTPException(413, "文件大小超过 1MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"
    if pathlib.Path(filename).suffix.lower() not in {".txt", ".md", ".json"}:
        raise HTTPException(400, "仅支持 TXT、Markdown 或 JSON 文件")

    if filename.lower().endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    if not all(isinstance(d, dict) and isinstance(d.get("content"), str) and isinstance(d.get("title", ""), str) for d in docs):
        raise HTTPException(400, "每篇文档必须包含文本 content 与可选文本 title")
    with bind_request_user(user):
        if multi_user_enabled() and user is not None:
            count = await kb.add_documents_async(docs, user_id=user.id)
            total = await kb.doc_count_async(user_id=user.id)
        else:
            count = await kb.add_documents_async(docs)
            total = await kb.doc_count_async()
    _tool_manager.clear_cache()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats(user: Optional[AuthUser] = Depends(require_user_if_multi)):
    """查看知识库统计信息（multi 模式下为当前用户片段数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    with bind_request_user(user):
        if multi_user_enabled() and user is not None:
            total = await kb.doc_count_async(user_id=user.id)
            return {"total_chunks": total, "user_id": user.id, "scoped": True}
        return {"total_chunks": await kb.doc_count_async(), "scoped": False}


@app.post("/knowledge/seed-defaults", tags=["知识库"])
async def seed_default_knowledge(
    force: bool = True,
    user: Optional[AuthUser] = Depends(require_user_if_multi),
):
    """将 config/default_knowledge.json 灌入当前用户知识库（默认强制覆盖同名默认文档）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    ensure = getattr(kb, "ensure_default_docs_async", None)
    if ensure is None:
        # free / chroma: 直接追加 JSON 语料
        from mcp.default_knowledge import load_default_knowledge_docs

        docs = [dict(item) for item in load_default_knowledge_docs()]
        with bind_request_user(user):
            if multi_user_enabled() and user is not None and hasattr(kb, "add_documents_async"):
                added = await kb.add_documents_async(docs, user_id=user.id)
                total = await kb.doc_count_async(user_id=user.id)
            else:
                added = await kb.add_documents_async(docs)
                total = await kb.doc_count_async()
        _tool_manager.clear_cache()
        return {"message": "已导入默认知识库", "added_chunks": added, "total_chunks": total, "force": force}

    with bind_request_user(user):
        if multi_user_enabled() and user is None:
            raise HTTPException(401, "请先登录")
        added = await ensure(user.id if user else "anonymous", force=force)
        total = await kb.doc_count_async(user_id=user.id) if user else await kb.doc_count_async()
    _tool_manager.clear_cache()
    return {
        "message": "已加载 config/default_knowledge.json 为默认知识库",
        "added_chunks": added,
        "total_chunks": total,
        "force": force,
        "user_id": user.id if user else None,
    }


def _serialize_eval_report(report) -> Dict[str, Any]:
    return {
        "pass_rate": report.pass_rate,
        "total": report.total,
        "passed": report.passed,
        "avg_scores": report.avg_scores,
        "regressions": report.regressions,
        "recommendations": report.recommendations,
        "timestamp": report.timestamp,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


@app.get("/eval/suites", tags=["Eval Platform"])
async def eval_suites():
    """列出评测套件（含 suite_catalog 配置：tags / fail_under / purpose）。"""
    from evaluation.suite_loader import default_suite_id, list_suites, load_catalog

    catalog = load_catalog()
    return {
        "default_suite": default_suite_id(),
        "catalog_version": catalog.get("version"),
        "items": list_suites(),
    }


@app.get("/eval/runs", tags=["Eval Platform"])
async def eval_runs(limit: int = 20):
    """历史跑次列表（新→旧）。"""
    from evaluation.store import list_runs

    return {"items": list_runs(limit=min(max(limit, 1), 100))}


@app.get("/eval/runs/{run_id}", tags=["Eval Platform"])
async def eval_run_detail(run_id: str):
    """单个跑次详情。"""
    from evaluation.store import get_run

    try:
        return get_run(run_id)
    except FileNotFoundError as ex:
        raise HTTPException(404, str(ex)) from ex


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行评测：支持 suite_id 中台套件，或工作台内联少量 case。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_INTENT_CASES, IntentTestCase
    from evaluation.store import (
        append_regression_cases,
        evaluate_gate,
        save_baseline_report,
        save_run,
    )
    from evaluation.suite_loader import load_suite

    body = body or EvalRunInput()
    suite_id = None
    suite_version = None

    gate_threshold = body.fail_under
    if body.suite_id:
        try:
            suite = load_suite(body.suite_id)
        except FileNotFoundError as ex:
            raise HTTPException(404, str(ex)) from ex
        except ValueError as ex:
            raise HTTPException(400, str(ex)) from ex
        intent_cases = suite.intent_cases
        dialog_cases = suite.dialog_cases
        suite_id = suite.id
        suite_version = suite.version
        if gate_threshold is None and suite.config and suite.config.fail_under is not None:
            gate_threshold = suite.config.fail_under
        max_units = int(os.getenv("EVAL_SUITE_MAX_UNITS", "100"))
        if suite.case_count > max_units:
            raise HTTPException(400, f"套件过大（约 {suite.case_count} 单位），超过 EVAL_SUITE_MAX_UNITS={max_units}")
    else:
        # 工作台实验模式：兼容旧前端单案例
        if body.intent_cases is not None:
            intent_cases = [
                IntentTestCase(
                    message=c.message,
                    expected_intent=c.expected_intent,
                    context=c.context,
                )
                for c in body.intent_cases
            ]
        else:
            intent_cases = DEFAULT_INTENT_CASES[:1]

        if body.dialog_cases is not None:
            dialog_cases = [c.model_dump(exclude_none=True) for c in body.dialog_cases]
        else:
            dialog_cases = []

        if len(intent_cases) + len(dialog_cases) > 5:
            raise HTTPException(400, "免费工作台每次最多评测 5 个案例；批量请传 suite_id")
        if any(len(case.get("turns") or []) > 3 for case in dialog_cases):
            raise HTTPException(400, "单个案例最多 3 轮对话")

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    gate = evaluate_gate(report, gate_threshold)
    run_payload = None
    if body.save_run:
        run_payload = save_run(
            report,
            suite_id=suite_id,
            suite_version=suite_version,
            gate=gate,
        )
    if body.update_baseline:
        save_baseline_report(report)
        _evaluator._baseline = report
        if _evaluator._baseline_path:
            _evaluator._save_baseline(report)
    regressions_added = 0
    if body.collect_regressions:
        regressions_added = append_regression_cases(report)

    response = {
        **_serialize_eval_report(report),
        "suite_id": suite_id,
        "suite_version": suite_version,
        "run_id": (run_payload or {}).get("run_id"),
        "gate": gate,
        "regressions_added": regressions_added,
    }
    if gate.get("enabled") and not gate.get("passed"):
        return JSONResponse(status_code=422, content=response)
    return response

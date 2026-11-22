"""LangGraph + LangChain orchestrator facade (sole runtime)."""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

from langgraph.types import Command

from agents.contracts import AgentType, OrchestratorResult, Request
from agents.graph.build import build_agent_graph
from agents.graph.checkpointer import create_checkpointer
from agents.graph.lc_runtime import AgentRuntime
from agents.graph.state import AgentGraphState
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from monitor.request_tracer import build_request_trace

logger = logging.getLogger(__name__)


def _intent_value(intent: Optional[IntentCategory]) -> Optional[str]:
    return intent.value if intent else None


def _urgency_value(urgency: Optional[UrgencyLevel]) -> Optional[str]:
    return urgency.name if urgency else None


def _parse_intent(value: Optional[str]) -> Optional[IntentCategory]:
    if not value:
        return None
    try:
        return IntentCategory(value)
    except ValueError:
        return IntentCategory.OTHER


def _interrupt_payload(snapshot: Any) -> Optional[Dict[str, Any]]:
    tasks = getattr(snapshot, "tasks", None) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or ()
        for item in interrupts:
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                return value
            return {"value": value}
    values = getattr(snapshot, "values", None) or {}
    raw = values.get("__interrupt__")
    if raw:
        first = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        value = getattr(first, "value", first)
        return value if isinstance(value, dict) else {"value": value}
    return None


class GraphFacade:
    """Checkpointed multi-agent runtime with resume / history / fork."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        rag_tool_manager: Optional[Any] = None,
    ):
        self._intent_recognizer = IntentRecognizer(
            api_key=api_key, base_url=base_url, model=model
        )
        self._runtime = AgentRuntime(
            api_key=api_key,
            base_url=base_url,
            default_model=model,
            skill_manager=skill_manager,
        )
        # Compatibility aliases used by api/main shutdown.
        self._composer = self._runtime
        self._checkpointer = create_checkpointer()
        self._graph = build_agent_graph(self._runtime, self._checkpointer)
        self._recent_tool_traces: deque = deque(maxlen=200)
        if rag_tool_manager is not None:
            self.set_tool_manager(rag_tool_manager)
        logger.info("GraphFacade ready (LangGraph + LangChain + ToolManager gateway)")

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        self._runtime.set_skill_manager(skill_manager)

    def set_tool_manager(self, tool_manager: Optional[Any]) -> None:
        self._runtime.set_tool_manager(tool_manager)

    def set_shared_tools(self, tools: Optional[Dict[str, Any]]) -> None:
        """兼容旧调用；优先使用 set_tool_manager。"""
        self._runtime.set_shared_tools(tools)

    async def recognize_intent(self, message: str, history: Optional[List[Dict[str, str]]] = None):
        return await self._intent_recognizer.recognize(message, history=history)

    def get_tool_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        """兼容旧名；返回完整请求级 trace（spans / tokens / tool_tree）。"""
        return self.get_request_trace(request_id)

    def get_request_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._recent_tool_traces):
            if trace.get("request_id") == request_id:
                return trace
        return None

    def get_recent_tool_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._recent_tool_traces:
            return []
        limit = max(1, min(int(limit or 20), len(self._recent_tool_traces)))
        return list(reversed(list(self._recent_tool_traces)[-limit:]))

    def get_recent_request_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.get_recent_tool_traces(limit=limit)

    def get_stats(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for agent_type, stats in self._runtime.stats.items():
            result[f"{agent_type.value}_0"] = {
                "total": stats.total,
                "success_rate": round(stats.success_rate, 3),
                "avg_ms": round(stats.avg_ms, 1),
                "monitor_penalty": round(stats.monitor_penalty, 3),
                "routing_score": round(stats.routing_score(), 3),
                "role": agent_type.value,
            }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        for agent_type, stats in self._runtime.stats.items():
            key = f"{agent_type.value}_0"
            stats.monitor_penalty = min(max(penalties.get(key, 0.0), 0.0), 0.9)

    def _record_tool_trace(self, result: OrchestratorResult) -> None:
        self._recent_tool_traces.append(build_request_trace(result))

    def _config(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        configurable: Dict[str, Any] = {"thread_id": thread_id}
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable}

    async def _to_result(
        self,
        state: AgentGraphState,
        *,
        req: Request,
        thread_id: str,
        latency_ms: float,
        interrupted: bool = False,
        interrupt_payload: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> OrchestratorResult:
        primary_name = state.get("primary_agent") or AgentType.GENERAL.value
        primary_agent = AgentType(primary_name)
        supporting = [AgentType(name) for name in state.get("supporting_agents") or []]
        agent_types = [
            AgentType(name) for name in state.get("agent_types") or [primary_name]
        ]
        response_text = state.get("final_response") or ""
        if interrupted and interrupt_payload and not response_text:
            response_text = str(interrupt_payload.get("prompt") or "等待人工输入以继续。")

        result = OrchestratorResult(
            request_id=req.request_id,
            response=response_text,
            agent_type=primary_agent,
            intent=_parse_intent(state.get("intent")) or req.intent,
            escalated=bool(state.get("escalated")),
            latency_ms=latency_ms,
            agent_types=agent_types,
            primary_agent=primary_agent,
            supporting_agents=supporting,
            tools_used=list(state.get("tools_used") or []),
            tool_traces=list(state.get("tool_traces") or []),
            token_usage=dict(state.get("token_usage") or {}),
            routing_reason=state.get("routing_reason") or "",
            routing_confidence=float(state.get("routing_confidence") or 0.0),
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            interrupted=interrupted,
            interrupt_payload=interrupt_payload,
        )
        self._record_tool_trace(result)
        return result

    async def _finish_from_config(
        self,
        config: Dict[str, Any],
        req: Request,
        t0: float,
        graph_values: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        snapshot = await self._graph.aget_state(config)
        payload = _interrupt_payload(snapshot)
        interrupted = payload is not None
        values: AgentGraphState = dict(graph_values or snapshot.values or {})
        checkpoint_id = None
        cfg = getattr(snapshot, "config", None) or {}
        checkpoint_id = (cfg.get("configurable") or {}).get("checkpoint_id")
        return await self._to_result(
            values,
            req=req,
            thread_id=config["configurable"]["thread_id"],
            latency_ms=(time.monotonic() - t0) * 1000,
            interrupted=interrupted,
            interrupt_payload=payload,
            checkpoint_id=checkpoint_id,
        )

    async def run(self, req: Request) -> OrchestratorResult:
        t0 = time.monotonic()
        if req.intent is None:
            intent_result = await self.recognize_intent(req.message, history=req.history)
            req.intent = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.urgency = intent_result.urgency
            req.intent_confidence = intent_result.confidence

        thread_id = req.thread_id or req.conv_id or str(uuid.uuid4())
        req.thread_id = thread_id
        config = self._config(thread_id)

        initial: AgentGraphState = {
            "request_id": req.request_id,
            "user_id": req.user_id,
            "conv_id": req.conv_id,
            "thread_id": thread_id,
            "message": req.message,
            "history": list(req.history or []),
            "context": req.context or "",
            "entities": dict(req.entities or {}),
            "intent": _intent_value(req.intent),
            "intent_group": req.intent_group,
            "urgency": _urgency_value(req.urgency),
            "intent_confidence": float(req.intent_confidence or 0.0),
            "supporting_agents": [],
            "agent_types": [],
            "tools_used": [],
            "tool_traces": [],
            "token_usage": {},
            "agent_responses": [],
            "escalated": False,
            "needs_clarification": False,
            "multi_agent": False,
            "await_human_escalation": False,
            "final_response": "",
            "routing_reason": "",
            "routing_confidence": 0.0,
        }

        values = await self._graph.ainvoke(initial, config)
        return await self._finish_from_config(config, req, t0, values)

    async def resume(self, thread_id: str, resume_value: Any, request_id: Optional[str] = None) -> OrchestratorResult:
        """断点续送：向 interrupt 点注入用户输入并继续执行。"""
        t0 = time.monotonic()
        config = self._config(thread_id)
        values = await self._graph.ainvoke(Command(resume=resume_value), config)
        snap = await self._graph.aget_state(config)
        state_values = dict(values or snap.values or {})
        req = Request(
            message=str(resume_value if isinstance(resume_value, str) else state_values.get("message", "")),
            user_id=state_values.get("user_id", "anonymous"),
            conv_id=state_values.get("conv_id", thread_id),
            request_id=request_id or state_values.get("request_id") or str(uuid.uuid4())[:8],
            intent=_parse_intent(state_values.get("intent")),
            intent_group=state_values.get("intent_group"),
            urgency=_urgency_value_to_enum(state_values.get("urgency")),
            intent_confidence=float(state_values.get("intent_confidence") or 0.0),
            entities=dict(state_values.get("entities") or {}),
            thread_id=thread_id,
        )
        return await self._finish_from_config(config, req, t0, state_values)

    async def history(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """回溯：列出线程 checkpoint 历史。"""
        config = self._config(thread_id)
        items: List[Dict[str, Any]] = []
        async for snap in self._graph.aget_state_history(config):
            cfg = getattr(snap, "config", {}) or {}
            conf = cfg.get("configurable") or {}
            values = getattr(snap, "values", {}) or {}
            items.append(
                {
                    "checkpoint_id": conf.get("checkpoint_id"),
                    "thread_id": conf.get("thread_id", thread_id),
                    "next": list(getattr(snap, "next", []) or []),
                    "primary_agent": values.get("primary_agent"),
                    "routing_reason": values.get("routing_reason"),
                    "final_response": (values.get("final_response") or "")[:200],
                    "interrupted": _interrupt_payload(snap) is not None,
                }
            )
            if len(items) >= limit:
                break
        return items

    async def fork(
        self,
        source_thread_id: str,
        checkpoint_id: str,
        new_thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """分叉：从历史 checkpoint 复制状态到新 thread，可继续改写。"""
        source = self._config(source_thread_id, checkpoint_id)
        snap = await self._graph.aget_state(source)
        if not snap or not snap.values:
            raise ValueError("找不到指定 checkpoint")

        target_thread = new_thread_id or str(uuid.uuid4())
        target = self._config(target_thread)
        # 将历史状态写入新线程，作为分叉起点
        await self._graph.aupdate_state(target, dict(snap.values))
        new_snap = await self._graph.aget_state(target)
        new_cfg = (getattr(new_snap, "config", {}) or {}).get("configurable") or {}
        return {
            "source_thread_id": source_thread_id,
            "source_checkpoint_id": checkpoint_id,
            "thread_id": target_thread,
            "checkpoint_id": new_cfg.get("checkpoint_id"),
            "values_preview": {
                "message": (snap.values or {}).get("message"),
                "primary_agent": (snap.values or {}).get("primary_agent"),
                "final_response": ((snap.values or {}).get("final_response") or "")[:200],
            },
        }


def _urgency_value_to_enum(value: Optional[str]) -> Optional[UrgencyLevel]:
    if not value:
        return None
    try:
        return UrgencyLevel[value]
    except KeyError:
        return None

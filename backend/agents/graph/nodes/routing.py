"""Clarify / route nodes with LangGraph interrupt for 断点续送."""

from __future__ import annotations

from typing import Any, Callable, Dict

from langgraph.types import interrupt

from agents.contracts import AgentType, Request
from agents.graph.state import AgentGraphState
from agents.routing_engine import needs_clarification, route_decision
from core.intent_recognizer import IntentCategory, UrgencyLevel


def _parse_intent(value: str | None) -> IntentCategory | None:
    if not value:
        return None
    try:
        return IntentCategory(value)
    except ValueError:
        return IntentCategory.OTHER


def _parse_urgency(value: str | None) -> UrgencyLevel | None:
    if not value:
        return None
    try:
        return UrgencyLevel[value]
    except KeyError:
        return None


def request_from_state(state: AgentGraphState) -> Request:
    return Request(
        message=state.get("message", ""),
        user_id=state.get("user_id", ""),
        conv_id=state.get("conv_id", ""),
        context=state.get("context", ""),
        history=state.get("history"),
        entities=dict(state.get("entities") or {}),
        intent=_parse_intent(state.get("intent")),
        intent_group=state.get("intent_group"),
        urgency=_parse_urgency(state.get("urgency")),
        intent_confidence=float(state.get("intent_confidence") or 0.0),
        request_id=state.get("request_id") or "",
        thread_id=state.get("thread_id"),
    )


def build_clarify_node() -> Callable[[AgentGraphState], Dict[str, Any]]:
    def clarify_node(state: AgentGraphState) -> Dict[str, Any]:
        req = request_from_state(state)
        if not needs_clarification(req):
            return {"needs_clarification": False}

        prompt = (
            "我还不能确定您要处理的是哪类问题。"
            "请补充一下是订单物流、退款账单、账户资料，还是技术故障？"
        )
        # 断点：等待用户补充；resume 值会作为澄清后的新消息继续图
        resumed = interrupt(
            {
                "type": "clarify",
                "prompt": prompt,
                "request_id": req.request_id,
                "thread_id": state.get("thread_id"),
            }
        )
        new_message = resumed if isinstance(resumed, str) else str(
            (resumed or {}).get("message") or (resumed or {}).get("text") or prompt
        )
        return {
            "needs_clarification": False,
            "message": new_message.strip() or req.message,
            "intent_confidence": max(req.intent_confidence, 0.6),
            "clarification_prompt": prompt,
        }

    return clarify_node


def build_route_node() -> Callable[[AgentGraphState], Dict[str, Any]]:
    def route_node(state: AgentGraphState) -> Dict[str, Any]:
        req = request_from_state(state)
        decision = route_decision(req)
        return {
            "needs_clarification": False,
            "primary_agent": decision.primary_agent.value,
            "supporting_agents": [agent.value for agent in decision.supporting_agents],
            "agent_types": [agent.value for agent in decision.agent_types],
            "routing_reason": decision.reason,
            "routing_confidence": decision.confidence,
            "multi_agent": decision.multi_agent,
            "await_human_escalation": decision.primary_agent is AgentType.ESCALATION,
        }

    return route_node

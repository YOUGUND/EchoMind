"""Execute / compose / human-escalation nodes."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List

from langgraph.types import interrupt

from agents.contracts import AgentResponse, AgentType
from agents.graph.lc_runtime import AgentRuntime
from agents.graph.nodes.routing import request_from_state
from agents.graph.state import AgentGraphState
from core.intent_recognizer import IntentCategory, UrgencyLevel


def _serialize_response(response: AgentResponse) -> Dict[str, Any]:
    return {
        "agent_type": response.agent_type.value,
        "content": response.content,
        "success": response.success,
        "confidence": response.confidence,
        "latency_ms": response.latency_ms,
        "escalate": response.escalate,
        "tools_used": list(response.tools_used),
        "tool_traces": list(response.tool_traces),
        "token_usage": dict(response.token_usage or {}),
    }


def _deserialize_response(payload: Dict[str, Any]) -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType(payload["agent_type"]),
        content=payload.get("content", ""),
        success=bool(payload.get("success")),
        confidence=float(payload.get("confidence") or 1.0),
        latency_ms=float(payload.get("latency_ms") or 0.0),
        escalate=bool(payload.get("escalate")),
        tools_used=list(payload.get("tools_used") or []),
        tool_traces=list(payload.get("tool_traces") or []),
        token_usage=dict(payload.get("token_usage") or {}),
    )


def build_execute_node(runtime: AgentRuntime) -> Callable[[AgentGraphState], Any]:
    async def execute_node(state: AgentGraphState) -> Dict[str, Any]:
        req = request_from_state(state)
        primary = AgentType(state["primary_agent"])
        supporting = [AgentType(name) for name in state.get("supporting_agents") or []]

        if supporting:
            agent_types = [primary] + supporting
            gathered = await asyncio.gather(
                *[runtime.run_agent(agent_type, req) for agent_type in agent_types],
                return_exceptions=True,
            )
            valid = [item for item in gathered if isinstance(item, AgentResponse)]
            return {
                "agent_responses": [_serialize_response(item) for item in valid],
                "multi_agent": True,
            }

        response = await runtime.run_agent(primary, req)
        return {
            "agent_responses": [_serialize_response(response)],
            "multi_agent": False,
        }

    return execute_node


def build_human_gate_node() -> Callable[[AgentGraphState], Dict[str, Any]]:
    """升级前人工确认断点：resume 可传 approve/reject。"""

    def human_gate(state: AgentGraphState) -> Dict[str, Any]:
        if not state.get("await_human_escalation"):
            return {"await_human_escalation": False}

        decision = interrupt(
            {
                "type": "escalation_approval",
                "prompt": "需要人工升级确认。请回复 approve 继续，或 reject 取消升级。",
                "request_id": state.get("request_id"),
                "thread_id": state.get("thread_id"),
                "primary_agent": state.get("primary_agent"),
            }
        )
        text = decision if isinstance(decision, str) else str(
            (decision or {}).get("action") or (decision or {}).get("message") or "approve"
        )
        approved = text.strip().lower() not in {"reject", "deny", "no", "取消", "拒绝"}
        if not approved:
            return {
                "await_human_escalation": False,
                "escalated": False,
                "final_response": "已取消人工升级。如需继续，请补充更具体的问题。",
                "agent_responses": [],
                "tools_used": [],
                "tool_traces": [],
            }
        return {"await_human_escalation": False, "escalated": True}

    return human_gate


def build_compose_node(runtime: AgentRuntime) -> Callable[[AgentGraphState], Any]:
    async def compose_node(state: AgentGraphState) -> Dict[str, Any]:
        if state.get("final_response") and not state.get("agent_responses"):
            return {
                "final_response": state["final_response"],
                "tools_used": list(state.get("tools_used") or []),
                "tool_traces": list(state.get("tool_traces") or []),
                "escalated": bool(state.get("escalated")),
            }

        req = request_from_state(state)
        payloads = list(state.get("agent_responses") or [])
        responses = [_deserialize_response(item) for item in payloads]

        if state.get("multi_agent") and len(responses) > 1:
            content = await runtime.compose(req, responses)
        elif responses:
            content = responses[0].content
        else:
            content = state.get("final_response") or "抱歉，所有 Agent 均处理失败。"

        tools_used = list(
            dict.fromkeys(
                tool_name for response in responses for tool_name in response.tools_used
            )
        )
        tool_traces = [
            trace for response in responses for trace in response.tool_traces
        ]
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for response in responses:
            usage = response.token_usage or {}
            token_usage["input_tokens"] += int(usage.get("input_tokens") or 0)
            token_usage["output_tokens"] += int(usage.get("output_tokens") or 0)
            token_usage["total_tokens"] += int(usage.get("total_tokens") or 0)
        if token_usage["total_tokens"] == 0:
            token_usage["total_tokens"] = token_usage["input_tokens"] + token_usage["output_tokens"]
        escalated = bool(state.get("escalated")) or any(item.escalate for item in responses)
        if req.urgency == UrgencyLevel.CRITICAL or req.intent in (
            IntentCategory.ESCALATION,
            IntentCategory.HUMAN_HANDOFF,
        ):
            escalated = True

        primary = state.get("primary_agent") or (
            responses[0].agent_type.value if responses else AgentType.GENERAL.value
        )
        supporting = list(state.get("supporting_agents") or [])
        agent_types = list(state.get("agent_types") or ([primary] + supporting))

        return {
            "final_response": content,
            "tools_used": tools_used,
            "tool_traces": tool_traces,
            "token_usage": token_usage,
            "escalated": escalated,
            "primary_agent": primary,
            "supporting_agents": supporting,
            "agent_types": agent_types,
        }

    return compose_node

"""Build replayable request traces (spans + tokens + tool tree) for observability."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.contracts import OrchestratorResult


def _sum_tokens(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for item in traces:
        usage = item.get("token_usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _tool_tree(tool_traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group tool/llm events under each agent for UI / interview replay."""
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for item in tool_traces:
        agent = str(item.get("agent_type") or "unknown")
        by_agent.setdefault(agent, []).append(
            {
                "kind": item.get("kind") or ("llm" if item.get("tool_name") == "llm" else "tool"),
                "name": item.get("tool_name") or item.get("name") or "unknown",
                "success": item.get("success"),
                "latency_ms": item.get("latency_ms"),
                "error": item.get("error") or "",
                "token_usage": item.get("token_usage") or {},
                "input": item.get("input") or {},
            }
        )
    return [
        {"agent_type": agent, "calls": calls}
        for agent, calls in by_agent.items()
    ]


def build_request_trace(result: OrchestratorResult) -> Dict[str, Any]:
    """
    Synthesize an OpenTelemetry-style span tree from one OrchestratorResult.

    Spans are derived from graph outcomes (intent → route → execute → compose),
    so operators can replay a conversation without a full APM vendor.
    """
    tool_traces = list(result.tool_traces or [])
    token_usage = dict(result.token_usage or {}) or _sum_tokens(tool_traces)
    primary = result.primary_agent.value if result.primary_agent else (
        result.agent_type.value if result.agent_type else None
    )
    supporting = [agent.value for agent in result.supporting_agents]

    spans: List[Dict[str, Any]] = [
        {
            "name": "intent",
            "status": "ok" if result.intent else "unknown",
            "attributes": {
                "intent": result.intent.value if result.intent else None,
                "confidence": result.routing_confidence,
            },
        },
        {
            "name": "route",
            "status": "ok",
            "attributes": {
                "primary_agent": primary,
                "supporting_agents": supporting,
                "reason": result.routing_reason,
                "confidence": result.routing_confidence,
                "escalated": result.escalated,
            },
        },
        {
            "name": "execute",
            "status": "ok" if not result.interrupted else "interrupted",
            "attributes": {
                "agents": [agent.value for agent in result.agent_types] or ([primary] if primary else []),
                "tools_used": list(result.tools_used),
            },
            "children": _tool_tree(tool_traces),
        },
        {
            "name": "compose",
            "status": "ok" if result.response else "empty",
            "attributes": {
                "response_chars": len(result.response or ""),
                "interrupted": result.interrupted,
            },
        },
    ]

    return {
        "request_id": result.request_id,
        "timestamp": datetime.now().isoformat(),
        "thread_id": result.thread_id,
        "checkpoint_id": result.checkpoint_id,
        "intent": result.intent.value if result.intent else None,
        "primary_agent": primary,
        "supporting_agents": supporting,
        "tools_used": list(result.tools_used),
        "tool_calls": tool_traces,
        "tool_tree": _tool_tree(tool_traces),
        "spans": spans,
        "token_usage": token_usage,
        "escalated": result.escalated,
        "interrupted": result.interrupted,
        "latency_ms": round(result.latency_ms, 1),
        "routing_reason": result.routing_reason,
        "routing_confidence": result.routing_confidence,
    }


def extract_message_usage(message: Any) -> Dict[str, int]:
    """Best-effort token usage from LangChain AIMessage / provider metadata."""
    usage = getattr(message, "usage_metadata", None) or {}
    if isinstance(usage, dict) and any(usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")):
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    meta = getattr(message, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    if not isinstance(token_usage, dict):
        return {}
    input_tokens = int(
        token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or 0
    )
    output_tokens = int(
        token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or 0
    )
    total_tokens = int(token_usage.get("total_tokens") or (input_tokens + output_tokens))
    if not (input_tokens or output_tokens or total_tokens):
        return {}
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def merge_token_usage(*parts: Optional[Dict[str, int]]) -> Dict[str, int]:
    merged = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for part in parts:
        if not part:
            continue
        merged["input_tokens"] += int(part.get("input_tokens") or 0)
        merged["output_tokens"] += int(part.get("output_tokens") or 0)
        merged["total_tokens"] += int(part.get("total_tokens") or 0)
    if merged["total_tokens"] == 0:
        merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
    return merged

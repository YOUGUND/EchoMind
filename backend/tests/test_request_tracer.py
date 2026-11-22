"""Request-level spans / token aggregation / tool tree."""

from __future__ import annotations

from types import SimpleNamespace

from agents.contracts import AgentType, OrchestratorResult
from core.intent_recognizer import IntentCategory
from monitor.request_tracer import (
    build_request_trace,
    extract_message_usage,
    merge_token_usage,
)


def test_extract_message_usage_from_usage_metadata():
    msg = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    assert extract_message_usage(msg) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_extract_message_usage_from_openai_style_metadata():
    msg = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 3, "completion_tokens": 7}},
    )
    assert extract_message_usage(msg)["total_tokens"] == 10


def test_merge_token_usage_sums_parts():
    merged = merge_token_usage(
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
    )
    assert merged == {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}


def test_build_request_trace_has_spans_and_tool_tree():
    result = OrchestratorResult(
        request_id="req1",
        response="已处理",
        agent_type=AgentType.TECHNICAL,
        intent=IntentCategory.TECHNICAL_LOGIN,
        primary_agent=AgentType.TECHNICAL,
        supporting_agents=[AgentType.BILLING],
        tools_used=["lookup_error_code"],
        tool_traces=[
            {
                "kind": "llm",
                "agent_type": "technical",
                "tool_name": "llm",
                "success": True,
                "latency_ms": 12.0,
                "token_usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
            },
            {
                "kind": "tool",
                "agent_type": "technical",
                "tool_name": "lookup_error_code",
                "success": True,
                "latency_ms": 3.0,
                "input": {"error_code": "401"},
            },
        ],
        token_usage={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
        routing_reason="primary=technical",
        routing_confidence=0.9,
        latency_ms=42.5,
        thread_id="t1",
    )
    trace = build_request_trace(result)
    assert trace["request_id"] == "req1"
    assert [s["name"] for s in trace["spans"]] == ["intent", "route", "execute", "compose"]
    assert trace["token_usage"]["total_tokens"] == 28
    assert trace["tool_tree"][0]["agent_type"] == "technical"
    assert len(trace["tool_tree"][0]["calls"]) == 2
    assert trace["spans"][2]["children"]

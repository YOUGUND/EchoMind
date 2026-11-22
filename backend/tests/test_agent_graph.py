"""LangGraph checkpointer / interrupt / fork baselines."""

from __future__ import annotations

import asyncio

import pytest

from agents.contracts import AgentResponse, AgentType, Request
from agents.graph.facade import GraphFacade
from agents.graph.factory import create_orchestrator
from agents.graph.nodes.routing import build_route_node
from agents.routing_engine import route_decision
from core.intent_recognizer import IntentCategory, UrgencyLevel

pytest.importorskip("langgraph")
pytest.importorskip("langchain_openai")


def make_request(**kwargs) -> Request:
    values = {
        "message": "登录时报 401，同时这笔订单被重复扣款",
        "user_id": "u1",
        "conv_id": "c1",
        "thread_id": "thread-demo",
        "intent": IntentCategory.TECHNICAL_LOGIN,
        "intent_group": "technical",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"error_code": ["401"], "amount": ["99 元"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_factory_returns_graph_facade():
    facade = create_orchestrator(api_key="k", model="m")
    assert isinstance(facade, GraphFacade)


def test_graph_route_node_matches_composite_baseline():
    route = build_route_node()
    out = route(
        {
            "message": make_request().message,
            "user_id": "u1",
            "conv_id": "c1",
            "intent": IntentCategory.TECHNICAL_LOGIN.value,
            "intent_group": "technical",
            "urgency": UrgencyLevel.HIGH.name,
            "intent_confidence": 0.92,
            "entities": {"error_code": ["401"], "amount": ["99 元"]},
        }
    )
    assert out["primary_agent"] == AgentType.TECHNICAL.value
    assert out["supporting_agents"] == [AgentType.BILLING.value]
    assert out["multi_agent"] is True


def test_graph_run_parallel_and_compose(monkeypatch):
    async def exercise():
        facade = GraphFacade(api_key="test-key", model="test-model")

        async def fake_run(agent_type, req):
            return AgentResponse(
                agent_type=agent_type,
                content=f"from-{agent_type.value}",
                success=True,
                tools_used=[f"tool-{agent_type.value}"],
            )

        async def fake_compose(req, responses):
            return " | ".join(item.content for item in responses)

        monkeypatch.setattr(facade._runtime, "run_agent", fake_run)
        monkeypatch.setattr(facade._runtime, "compose", fake_compose)

        result = await facade.run(make_request(thread_id="t-parallel"))
        assert result.interrupted is False
        assert result.primary_agent is AgentType.TECHNICAL
        assert result.supporting_agents == [AgentType.BILLING]
        assert "from-technical" in result.response
        assert result.thread_id == "t-parallel"
        assert facade.get_tool_trace(result.request_id) is not None

    asyncio.run(exercise())


def test_clarify_interrupt_then_resume(monkeypatch):
    async def exercise():
        facade = GraphFacade(api_key="test-key", model="test-model")

        async def fake_run(agent_type, req):
            return AgentResponse(
                agent_type=agent_type,
                content=f"answered:{req.message}",
                success=True,
            )

        monkeypatch.setattr(facade._runtime, "run_agent", fake_run)

        first = await facade.run(
            make_request(
                thread_id="t-clarify",
                message="随便问问这个那个怎么办呢到底",
                intent=IntentCategory.OTHER,
                intent_group="other",
                intent_confidence=0.2,
                entities={},
                urgency=UrgencyLevel.LOW,
            )
        )
        assert first.interrupted is True
        assert first.interrupt_payload
        assert first.interrupt_payload.get("type") == "clarify"

        second = await facade.resume("t-clarify", "我的订单物流一直没更新")
        assert second.interrupted is False
        assert "订单物流" in second.response or "answered:" in second.response

    asyncio.run(exercise())


def test_escalation_interrupt_approve_resume(monkeypatch):
    async def exercise():
        facade = GraphFacade(api_key="test-key", model="test-model")

        async def fake_run(agent_type, req):
            return AgentResponse(
                agent_type=agent_type,
                content="handoff-done",
                success=True,
                escalate=True,
            )

        monkeypatch.setattr(facade._runtime, "run_agent", fake_run)

        first = await facade.run(
            make_request(
                thread_id="t-esc",
                message="我要转人工",
                intent=IntentCategory.HUMAN_HANDOFF,
                intent_group="escalation",
                urgency=UrgencyLevel.CRITICAL,
                intent_confidence=0.99,
                entities={},
            )
        )
        assert first.interrupted is True
        assert first.interrupt_payload.get("type") == "escalation_approval"

        second = await facade.resume("t-esc", "approve")
        assert second.interrupted is False
        assert second.response == "handoff-done"
        assert second.escalated is True

    asyncio.run(exercise())


def test_history_and_fork(monkeypatch):
    async def exercise():
        facade = GraphFacade(api_key="test-key", model="test-model")

        async def fake_run(agent_type, req):
            return AgentResponse(agent_type=agent_type, content="ok", success=True)

        monkeypatch.setattr(facade._runtime, "run_agent", fake_run)
        result = await facade.run(
            make_request(
                thread_id="t-hist",
                message="登录失败 401",
                intent=IntentCategory.TECHNICAL_LOGIN,
                entities={"error_code": ["401"]},
            )
        )
        # force single-agent by message without billing keywords
        assert result.thread_id == "t-hist"
        history = await facade.history("t-hist", limit=10)
        assert history
        checkpoint_id = next((item["checkpoint_id"] for item in history if item.get("checkpoint_id")), None)
        assert checkpoint_id
        forked = await facade.fork("t-hist", checkpoint_id, new_thread_id="t-fork")
        assert forked["thread_id"] == "t-fork"
        assert forked["source_checkpoint_id"] == checkpoint_id

    asyncio.run(exercise())


def test_route_decision_billing_only():
    decision = route_decision(
        make_request(
            message="我要退款，刚才被重复扣款了",
            intent=IntentCategory.REFUND,
            intent_group="billing",
            entities={"amount": ["99"]},
        )
    )
    assert decision.primary_agent is AgentType.BILLING

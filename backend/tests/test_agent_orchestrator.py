"""Routing / tools / profile baselines after LangGraph migration."""

from __future__ import annotations

import asyncio

from agents.contracts import (
    AgentType,
    BILLING_PROFILE,
    GENERAL_PROFILE,
    Request,
    RoutingDecision,
    TECHNICAL_PROFILE,
)
from agents.graph.lc_runtime import AgentRuntime
from agents.routing_engine import needs_clarification, route_decision
from agents.tools import billing_tools, general_tools, technical_tools
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


def make_request(**kwargs) -> Request:
    values = {
        "message": "登录时报 401，同时这笔订单被重复扣款",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.TECHNICAL_LOGIN,
        "intent_group": "technical",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"error_code": ["401"], "amount": ["99 元"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert GENERAL_PROFILE.role != TECHNICAL_PROFILE.role
    assert TECHNICAL_PROFILE.workflow != BILLING_PROFILE.workflow
    assert TECHNICAL_PROFILE.temperature < GENERAL_PROFILE.temperature
    assert "search_knowledge_base" in GENERAL_PROFILE.tool_scope
    assert "lookup_error_code" in TECHNICAL_PROFILE.tool_scope
    assert "check_billing_fields" in BILLING_PROFILE.tool_scope


def test_specific_intent_uses_local_fast_path_without_model_round_trip():
    async def exercise():
        recognizer = IntentRecognizer(api_key="test-key")

        async def model_must_not_run(*args, **kwargs):
            raise AssertionError("specific intent should not call the model classifier")

        recognizer._llm_recognize = model_must_not_run
        recognizer._embedding_recognize = model_must_not_run
        try:
            result = await recognizer.recognize("登录失败，出现 401 错误")
        finally:
            await recognizer.client.close()

        assert result.intent is IntentCategory.TECHNICAL_LOGIN
        assert result.intent_group == "technical"
        assert result.source_scores["fast_path"] == 1.0

    asyncio.run(exercise())


def test_routing_decision_can_target_escalation_pool():
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_composite_request_routes_explicit_billing_signal_as_supporting_agent():
    decision = route_decision(make_request())
    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.BILLING]
    assert decision.multi_agent is True


def test_high_confidence_refund_stays_billing_without_weak_support():
    decision = route_decision(
        make_request(
            message="我要退款，刚才被重复扣款了",
            intent=IntentCategory.REFUND,
            intent_group="billing",
            intent_confidence=0.93,
            entities={"amount": ["99"]},
            urgency=UrgencyLevel.MEDIUM,
        )
    )
    assert decision.primary_agent is AgentType.BILLING
    assert decision.supporting_agents == []


def test_low_confidence_other_leans_on_keyword_evidence():
    decision = route_decision(
        make_request(
            message="页面一直白屏报错 500",
            intent=IntentCategory.OTHER,
            intent_group="other",
            intent_confidence=0.3,
            entities={},
            urgency=UrgencyLevel.MEDIUM,
        )
    )
    assert decision.primary_agent is AgentType.TECHNICAL


def test_weak_single_keyword_does_not_attach_supporting_agent():
    decision = route_decision(
        make_request(
            message="登录失败，提示验证码错误",
            intent=IntentCategory.TECHNICAL_LOGIN,
            intent_group="technical",
            intent_confidence=0.9,
            entities={"error_code": ["401"]},
            urgency=UrgencyLevel.MEDIUM,
        )
    )
    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == []


def test_needs_clarification_for_low_confidence_other():
    assert needs_clarification(
        make_request(
            message="随便问问这个那个怎么办呢到底",
            intent=IntentCategory.OTHER,
            intent_confidence=0.2,
            entities={},
        )
    )


def test_agent_tool_scopes_are_real_and_isolated():
    assert set(general_tools()) == {"inspect_request_context", "suggest_required_fields"}
    assert set(technical_tools()) == {"lookup_error_code", "build_diagnostic_plan"}
    assert set(billing_tools()) == {"check_billing_fields", "compare_amounts"}


def test_all_agent_tools_register_into_manager():
    from agents.tool_registry import all_agent_tool_specs, register_all_agent_tools
    from mcp.tool_manager import MCPToolManager

    manager = MCPToolManager(api_key="test-key", model="test-model")
    register_all_agent_tools(manager)
    for name in all_agent_tool_specs():
        assert name in manager._tools
    assert "lookup_error_code" in manager._tools
    assert "check_billing_fields" in manager._tools


def test_langchain_tools_execute_via_manager_gateway():
    import asyncio

    from agents.graph.lc_tools import tools_for_agent
    from agents.tool_registry import register_all_agent_tools
    from mcp.tool_manager import MCPToolManager

    manager = MCPToolManager(api_key="test-key", model="test-model")
    register_all_agent_tools(manager)
    req = make_request()
    tools = {t.name: t for t in tools_for_agent("technical", req, manager)}
    assert "lookup_error_code" in tools
    assert "search_knowledge_base" in tools

    async def exercise():
        result = await tools["lookup_error_code"].ainvoke({"error_code": "401"})
        assert result["meaning"]
        assert "gateway_latency_ms" in result or result.get("success") is not False

    asyncio.run(exercise())


def test_escalation_runtime_is_non_llm():
    runtime = AgentRuntime(api_key="k", base_url=None, default_model="m")
    result = asyncio.run(
        runtime.run_agent(
            AgentType.ESCALATION,
            make_request(intent=IntentCategory.HUMAN_HANDOFF, urgency=UrgencyLevel.CRITICAL),
        )
    )
    assert result.success is True
    assert result.escalate is True
    assert "人工升级" in result.content

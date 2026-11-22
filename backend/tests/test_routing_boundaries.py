"""Routing edge cases: clarify, escalation, override, supporting evidence floors."""

from __future__ import annotations

from agents.contracts import AgentType, Request
from agents.routing_engine import (
    PRIMARY_OVERRIDE_MARGIN,
    SUPPORT_EVIDENCE_MIN,
    needs_clarification,
    route_decision,
    score_components,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


def _req(**kwargs) -> Request:
    values = {
        "message": "hello",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.OTHER,
        "intent_group": "other",
        "urgency": UrgencyLevel.MEDIUM,
        "intent_confidence": 0.5,
        "entities": {},
    }
    values.update(kwargs)
    return Request(**values)


def test_critical_urgency_hard_routes_to_escalation():
    decision = route_decision(_req(urgency=UrgencyLevel.CRITICAL, intent=IntentCategory.QUERY))
    assert decision.primary_agent is AgentType.ESCALATION
    assert decision.confidence == 1.0


def test_human_handoff_intent_routes_to_escalation():
    decision = route_decision(
        _req(intent=IntentCategory.HUMAN_HANDOFF, intent_confidence=0.88)
    )
    assert decision.primary_agent is AgentType.ESCALATION


def test_high_confidence_specialist_resists_weak_cross_domain_noise():
    # Billing intent high confidence; technical keywords alone should not steal primary
    # unless rival evidence beats intent_part + PRIMARY_OVERRIDE_MARGIN.
    decision = route_decision(
        _req(
            message="我想退款，顺便页面有点卡",
            intent=IntentCategory.REFUND,
            intent_group="billing",
            intent_confidence=0.95,
            entities={"amount": ["99"]},
        )
    )
    assert decision.primary_agent is AgentType.BILLING
    components = score_components(
        _req(
            message="我想退款，顺便页面有点卡",
            intent=IntentCategory.REFUND,
            intent_confidence=0.95,
            entities={"amount": ["99"]},
        )
    )
    assert components.intent_weight >= 0.8
    tech_evidence = components.evidence.get(AgentType.TECHNICAL, 0.0)
    intent_part = components.intent.get(AgentType.BILLING, 0.0)
    assert tech_evidence <= intent_part + PRIMARY_OVERRIDE_MARGIN


def test_supporting_requires_independent_evidence_floor():
    decision = route_decision(
        _req(
            message="登录失败 401",
            intent=IntentCategory.TECHNICAL_LOGIN,
            intent_group="technical",
            intent_confidence=0.9,
            entities={"error_code": ["401"]},
        )
    )
    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == []
    components = score_components(
        _req(
            message="登录失败 401",
            intent=IntentCategory.TECHNICAL_LOGIN,
            intent_confidence=0.9,
            entities={"error_code": ["401"]},
        )
    )
    assert components.evidence.get(AgentType.BILLING, 0.0) < SUPPORT_EVIDENCE_MIN


def test_composite_evidence_can_attach_one_supporting_specialist():
    decision = route_decision(
        _req(
            message="登录时报 401，同时这笔订单被重复扣款",
            intent=IntentCategory.TECHNICAL_LOGIN,
            intent_group="technical",
            intent_confidence=0.92,
            entities={"error_code": ["401"], "amount": ["99 元"]},
            urgency=UrgencyLevel.HIGH,
        )
    )
    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.BILLING]
    assert len(decision.supporting_agents) == 1


def test_needs_clarification_skips_known_intents():
    assert not needs_clarification(
        _req(intent=IntentCategory.BILLING, intent_confidence=0.2, message="退款")
    )
    assert needs_clarification(
        _req(intent=IntentCategory.OTHER, intent_confidence=0.2, message="这个怎么办呢")
    )

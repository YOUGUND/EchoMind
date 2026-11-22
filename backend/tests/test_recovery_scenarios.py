"""Fixed recovery scenarios for checkpoint / interrupt / resume / fork (resume metrics)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pytest

from agents.contracts import AgentResponse, AgentType, Request
from agents.graph.facade import GraphFacade
from agents.routing_engine import needs_clarification, route_decision
from core.intent_recognizer import IntentCategory, UrgencyLevel

pytest.importorskip("langgraph")
pytest.importorskip("langchain_openai")


@dataclass
class RecoveryCase:
    id: str
    kind: str  # clarify_resume | escalation_resume | fork | route_gate
    spec: Dict[str, Any]


RECOVERY_CASES: List[RecoveryCase] = [
    RecoveryCase(
        "R1_vague_clarify_resume",
        "clarify_resume",
        {
            "message": "这个怎么办呢到底啊",
            "intent": IntentCategory.OTHER,
            "intent_group": "other",
            "intent_confidence": 0.15,
            "urgency": UrgencyLevel.LOW,
            "entities": {},
            "resume": "我想查一下物流为什么不更新",
        },
    ),
    RecoveryCase(
        "R2_unclear_other_clarify_resume",
        "clarify_resume",
        {
            "message": "帮我弄一下那个事",
            "intent": IntentCategory.OTHER,
            "intent_group": "other",
            "intent_confidence": 0.18,
            "urgency": UrgencyLevel.LOW,
            "entities": {},
            "resume": "订单重复扣款了要退款",
        },
    ),
    RecoveryCase(
        "R3_human_handoff_approve",
        "escalation_resume",
        {
            "message": "我要转人工",
            "intent": IntentCategory.HUMAN_HANDOFF,
            "intent_group": "escalation",
            "intent_confidence": 0.99,
            "urgency": UrgencyLevel.CRITICAL,
            "entities": {},
            "resume": "approve",
        },
    ),
    RecoveryCase(
        "R4_critical_urgency_approve",
        "escalation_resume",
        {
            "message": "账号被盗紧急处理",
            "intent": IntentCategory.ACCOUNT_SECURITY,
            "intent_group": "account",
            "intent_confidence": 0.9,
            "urgency": UrgencyLevel.CRITICAL,
            "entities": {},
            "resume": "approve",
        },
    ),
    RecoveryCase(
        "R5_fork_after_technical",
        "fork",
        {
            "message": "登录失败 401",
            "intent": IntentCategory.TECHNICAL_LOGIN,
            "intent_group": "technical",
            "intent_confidence": 0.93,
            "urgency": UrgencyLevel.HIGH,
            "entities": {"error_code": ["401"]},
        },
    ),
    RecoveryCase(
        "R6_fork_after_billing",
        "fork",
        {
            "message": "我要退款，刚才被重复扣款了",
            "intent": IntentCategory.REFUND,
            "intent_group": "billing",
            "intent_confidence": 0.94,
            "urgency": UrgencyLevel.MEDIUM,
            "entities": {"amount": ["99"]},
        },
    ),
    RecoveryCase(
        "R7_gate_critical_to_escalation",
        "route_gate",
        {
            "message": "很紧急",
            "intent": IntentCategory.QUERY,
            "urgency": UrgencyLevel.CRITICAL,
            "expect_primary": AgentType.ESCALATION,
        },
    ),
    RecoveryCase(
        "R8_gate_handoff_to_escalation",
        "route_gate",
        {
            "message": "转人工",
            "intent": IntentCategory.HUMAN_HANDOFF,
            "intent_confidence": 0.9,
            "urgency": UrgencyLevel.HIGH,
            "expect_primary": AgentType.ESCALATION,
        },
    ),
    RecoveryCase(
        "R9_gate_needs_clarification_true",
        "route_gate",
        {
            "message": "那个怎么弄啊",
            "intent": IntentCategory.OTHER,
            "intent_confidence": 0.2,
            "expect_clarify": True,
        },
    ),
    RecoveryCase(
        "R10_gate_known_intent_skips_clarify",
        "route_gate",
        {
            "message": "我要退款",
            "intent": IntentCategory.REFUND,
            "intent_confidence": 0.2,
            "expect_clarify": False,
        },
    ),
]


def _req(thread_id: str, spec: Dict[str, Any]) -> Request:
    return Request(
        message=str(spec.get("message") or "hello"),
        user_id="u-recovery",
        conv_id="c-recovery",
        thread_id=thread_id,
        intent=spec.get("intent", IntentCategory.OTHER),
        intent_group=str(spec.get("intent_group") or "other"),
        urgency=spec.get("urgency", UrgencyLevel.MEDIUM),
        intent_confidence=float(spec.get("intent_confidence") or 0.5),
        entities=dict(spec.get("entities") or {}),
    )


@pytest.mark.parametrize("case", RECOVERY_CASES, ids=[c.id for c in RECOVERY_CASES])
def test_recovery_scenario(case: RecoveryCase, monkeypatch):
    if case.kind == "route_gate":
        req = _req(f"t-{case.id}", case.spec)
        if "expect_primary" in case.spec:
            decision = route_decision(req)
            assert decision.primary_agent is case.spec["expect_primary"]
            return
        if "expect_clarify" in case.spec:
            assert needs_clarification(req) is bool(case.spec["expect_clarify"])
            return
        raise AssertionError(f"bad route_gate spec: {case.id}")

    async def exercise():
        facade = GraphFacade(api_key="test-key", model="test-model")

        async def fake_run(agent_type, req):
            return AgentResponse(
                agent_type=agent_type,
                content=f"ok:{req.message}",
                success=True,
                escalate=(case.kind == "escalation_resume"),
            )

        monkeypatch.setattr(facade._runtime, "run_agent", fake_run)
        thread_id = f"t-{case.id}"

        if case.kind == "clarify_resume":
            first = await facade.run(_req(thread_id, case.spec))
            assert first.interrupted is True
            assert (first.interrupt_payload or {}).get("type") == "clarify"
            second = await facade.resume(thread_id, case.spec["resume"])
            assert second.interrupted is False
            assert second.response
            return

        if case.kind == "escalation_resume":
            first = await facade.run(_req(thread_id, case.spec))
            assert first.interrupted is True
            assert (first.interrupt_payload or {}).get("type") == "escalation_approval"
            second = await facade.resume(thread_id, case.spec["resume"])
            assert second.interrupted is False
            assert second.escalated is True
            return

        if case.kind == "fork":
            result = await facade.run(_req(thread_id, case.spec))
            assert result.interrupted is False
            history = await facade.history(thread_id, limit=20)
            checkpoint_id = next(
                (item["checkpoint_id"] for item in history if item.get("checkpoint_id")),
                None,
            )
            assert checkpoint_id, "checkpoint missing — stale/missing state"
            forked = await facade.fork(thread_id, checkpoint_id, new_thread_id=f"{thread_id}-fork")
            assert forked["thread_id"] == f"{thread_id}-fork"
            assert forked["source_checkpoint_id"] == checkpoint_id
            return

        raise AssertionError(f"unknown kind {case.kind}")

    asyncio.run(exercise())


def test_recovery_suite_size_locked():
    assert len(RECOVERY_CASES) == 10

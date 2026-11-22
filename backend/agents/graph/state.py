"""LangGraph state for checkpointed multi-agent runs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentGraphState(TypedDict, total=False):
    request_id: str
    user_id: str
    conv_id: str
    thread_id: str
    message: str
    history: List[Dict[str, str]]
    context: str
    entities: Dict[str, List[str]]

    intent: Optional[str]
    intent_group: Optional[str]
    urgency: Optional[str]
    intent_confidence: float
    source_scores: Dict[str, float]

    primary_agent: str
    supporting_agents: List[str]
    routing_reason: str
    routing_confidence: float
    needs_clarification: bool
    multi_agent: bool
    await_human_escalation: bool

    agent_responses: List[Dict[str, Any]]
    final_response: str
    escalated: bool
    tools_used: List[str]
    tool_traces: List[Dict[str, Any]]
    token_usage: Dict[str, int]
    latency_ms: float
    agent_types: List[str]
    clarification_prompt: str

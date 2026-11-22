"""Compile checkpointed LangGraph with interrupt + resume support."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agents.graph.lc_runtime import AgentRuntime
from agents.graph.nodes.execute import (
    build_compose_node,
    build_execute_node,
    build_human_gate_node,
)
from agents.graph.nodes.routing import build_clarify_node, build_route_node
from agents.graph.state import AgentGraphState


def _after_human_gate(state: AgentGraphState) -> Literal["execute", "compose"]:
    # 用户拒绝升级时已写好 final_response，直接合成收尾
    if state.get("final_response") and not state.get("agent_responses"):
        return "compose"
    return "execute"


def build_agent_graph(runtime: AgentRuntime, checkpointer: Any) -> Any:
    """
    Topology:

      START → clarify(interrupt?) → route → human_gate(interrupt?)
            → execute → compose → END
    """
    graph = StateGraph(AgentGraphState)
    graph.add_node("clarify", build_clarify_node())
    graph.add_node("route", build_route_node())
    graph.add_node("human_gate", build_human_gate_node())
    graph.add_node("execute", build_execute_node(runtime))
    graph.add_node("compose", build_compose_node(runtime))

    graph.add_edge(START, "clarify")
    graph.add_edge("clarify", "route")
    graph.add_edge("route", "human_gate")
    graph.add_conditional_edges(
        "human_gate",
        _after_human_gate,
        {"execute": "execute", "compose": "compose"},
    )
    graph.add_edge("execute", "compose")
    graph.add_edge("compose", END)
    return graph.compile(checkpointer=checkpointer)

"""Register all agent tools into MCPToolManager (unified gateway)."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional

from agents.contracts import Request
from agents.tools import (
    AgentToolSpec,
    billing_tools,
    escalation_tools,
    general_tools,
    make_tool,
    technical_tools,
)
from mcp.tool_manager import Tool

logger = logging.getLogger(__name__)

# 读多写少的工具可短缓存；依赖当前 Request 的一律不缓存
_CACHE_TTL: Dict[str, float] = {
    "lookup_error_code": 300.0,
    "compare_amounts": 60.0,
    "knowledge_search": 300.0,
}


def all_agent_tool_specs() -> Dict[str, AgentToolSpec]:
    """业务 Agent 工具目录（不含 knowledge_search，该工具在 main 里单独注册）。"""
    specs: Dict[str, AgentToolSpec] = {}
    for group in (general_tools, technical_tools, billing_tools, escalation_tools):
        specs.update(group())
    return specs


def agent_whitelist(agent_type: str) -> Dict[str, AgentToolSpec]:
    """某角色可见工具（含共享 RAG 入口名）。"""
    private = {
        "general": general_tools,
        "technical": technical_tools,
        "billing": billing_tools,
        "escalation": escalation_tools,
    }.get(agent_type, general_tools)()
    specs = dict(private)
    specs["search_knowledge_base"] = make_tool(
        "search_knowledge_base",
        "检索知识库并返回最相关的文档片段；可用于通用、技术、账单和升级场景。",
        {
            "query": {"type": "string", "description": "用户问题或检索关键词"},
            "top_k": {"type": "integer", "description": "返回结果条数"},
        },
        _search_knowledge_placeholder,
        required=["query"],
    )
    return specs


async def _search_knowledge_placeholder(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """占位；实际由 lc_tools 走 Manager.search_with_rewrite。"""
    return {"success": False, "error": "search_knowledge_base 必须经 ToolManager 网关调用"}


def _wrap_handler(spec: AgentToolSpec):
    """AgentToolSpec(req, args) → MCP handler(params, context)."""

    async def handler(params: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Any:
        ctx = context or {}
        req = ctx.get("request")
        if req is None:
            req = Request(message="", user_id="system", conv_id="system")
        result = spec.handler(req, dict(params or {}))
        if inspect.isawaitable(result):
            result = await result
        return result

    return handler


def register_all_agent_tools(tool_manager: Any) -> None:
    """
    把全部角色工具注册进 MCPToolManager。

    knowledge_search 仍由 main 注册（绑定 KnowledgeBase handler）。
    search_knowledge_base 不在此注册为独立 MCP 工具，Agent 侧经网关改写检索 knowledge_search。
    """
    for name, spec in all_agent_tool_specs().items():
        tool_manager.register(
            Tool(
                name=name,
                description=spec.description,
                handler=_wrap_handler(spec),
                schema=spec.input_schema,
                cache_ttl=_CACHE_TTL.get(name, 0.0),
                timeout_s=15.0,
                supports_rerank=False,
            )
        )
    logger.info("已注册全部 Agent 工具到 MCPToolManager（共 %d 个）", len(all_agent_tool_specs()))

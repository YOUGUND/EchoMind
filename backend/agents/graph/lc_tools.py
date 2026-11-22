"""LangChain StructuredTools that always execute via MCPToolManager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from agents.contracts import Request
from agents.tool_registry import agent_whitelist
from agents.tools import AgentToolSpec


def _schema_model(spec: AgentToolSpec) -> type[BaseModel]:
    properties = spec.input_schema.get("properties", {})
    required = set(spec.input_schema.get("required") or [])
    fields: Dict[str, Any] = {}
    for name, meta in properties.items():
        typ = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
        }.get(meta.get("type"), str)
        default = ... if name in required else None
        fields[name] = (
            Optional[typ] if default is None else typ,
            Field(default=default, description=meta.get("description", "")),
        )
    if not fields:
        fields["noop"] = (Optional[str], Field(default=None, description="unused"))
    return create_model(f"{spec.name}_Args", **fields)


def _pack_result(tr: Any) -> Any:
    """把 ToolResult 收成模型可读的 payload，并带上治理元数据。"""
    data = getattr(tr, "data", None)
    success = bool(getattr(tr, "success", False))
    error = getattr(tr, "error", None)
    cached = bool(getattr(tr, "cached", False))
    latency_ms = getattr(tr, "latency_ms", 0.0)
    reranked = bool(getattr(tr, "reranked", False))

    if isinstance(data, dict):
        out = dict(data)
        out.setdefault("success", success)
        if error and "error" not in out:
            out["error"] = error
        out["cached"] = cached
        out["gateway_latency_ms"] = round(float(latency_ms or 0.0), 1)
        if reranked:
            out["reranked"] = True
        return out

    return {
        "success": success,
        "data": data,
        "error": error,
        "cached": cached,
        "gateway_latency_ms": round(float(latency_ms or 0.0), 1),
        "reranked": reranked,
    }


def to_gateway_tool(
    spec: AgentToolSpec,
    req: Request,
    tool_manager: Any,
) -> StructuredTool:
    """LangChain 工具：参数进 Manager.call / search_with_rewrite，不直调业务 handler。"""
    args_model = _schema_model(spec)
    tool_name = spec.name

    async def _arun(**kwargs: Any) -> Any:
        if tool_manager is None:
            return {"success": False, "error": "ToolManager 未注入"}
        clean = {k: v for k, v in kwargs.items() if k != "noop" and v is not None}
        context = {"request": req}

        if tool_name == "search_knowledge_base":
            query = str(clean.get("query") or req.message or "").strip()
            top_k = int(clean.get("top_k", 5) or 5)
            if not query:
                return {"success": False, "error": "query 不能为空", "results": []}
            tr = await tool_manager.search_with_rewrite(
                "knowledge_search",
                query,
                top_k=top_k,
                context=context,
            )
            if not getattr(tr, "success", False):
                return {
                    "success": False,
                    "query": query,
                    "error": getattr(tr, "error", "知识库检索失败"),
                    "results": [],
                    "reranked": False,
                    "cached": bool(getattr(tr, "cached", False)),
                }
            return {
                "success": True,
                "query": query,
                "top_k": top_k,
                "results": tr.data,
                "reranked": bool(getattr(tr, "reranked", False)),
                "cached": bool(getattr(tr, "cached", False)),
                "gateway_latency_ms": round(float(getattr(tr, "latency_ms", 0.0) or 0.0), 1),
            }

        tr = await tool_manager.call(tool_name, clean, context)
        return _pack_result(tr)

    def _run(**kwargs: Any) -> Any:
        raise RuntimeError(f"{tool_name} 必须异步经 ToolManager 调用")

    return StructuredTool.from_function(
        name=tool_name,
        description=spec.description,
        func=_run,
        coroutine=_arun,
        args_schema=args_model,
    )


def tools_for_agent(
    agent_type: str,
    req: Request,
    tool_manager: Any = None,
    shared_specs: Optional[Dict[str, AgentToolSpec]] = None,
) -> List[StructuredTool]:
    """
    按角色白名单生成 LangChain 工具；执行一律走 Manager。

    shared_specs 保留兼容参数，网关模式下忽略（白名单以 tool_registry 为准）。
    """
    del shared_specs  # 网关模式不再直绑 shared handler
    specs = agent_whitelist(agent_type)
    return [to_gateway_tool(spec, req, tool_manager) for spec in specs.values()]

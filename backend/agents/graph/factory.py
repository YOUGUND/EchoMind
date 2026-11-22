"""Orchestrator factory — LangGraph facade is the only runtime."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.graph.facade import GraphFacade

logger = logging.getLogger(__name__)


def create_orchestrator(
    api_key: str,
    base_url: Optional[str] = None,
    model: str = "claude-3-5-sonnet-20241022",
    skill_manager: Optional[Any] = None,
    rag_tool_manager: Optional[Any] = None,
) -> GraphFacade:
    logger.info("使用 LangGraph/LangChain GraphFacade 作为唯一编排后端")
    return GraphFacade(
        api_key=api_key,
        base_url=base_url,
        model=model,
        skill_manager=skill_manager,
        rag_tool_manager=rag_tool_manager,
    )

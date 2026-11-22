"""LangChain chat + tool-calling runtime for role agents."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agents.contracts import (
    PROFILES,
    AgentResponse,
    AgentStats,
    AgentType,
    Request,
)
from agents.graph.lc_tools import tools_for_agent
from core.model_scope import selected_model
from monitor.request_tracer import extract_message_usage, merge_token_usage

logger = logging.getLogger(__name__)


def openai_compatible_base_url(base_url: Optional[str]) -> Optional[str]:
    """
    LangChain ChatOpenAI needs an OpenAI-compatible root (…/v1).

    DeepSeek / some gateways expose Anthropic under `…/anthropic` while chat
    completions live at the parent host. Prefer OPENAI_BASE_URL when set.
    """
    override = os.getenv("OPENAI_BASE_URL", "").strip()
    raw = (override or base_url or "").strip()
    if not raw:
        return None
    cleaned = raw.rstrip("/")
    if cleaned.endswith("/anthropic"):
        cleaned = cleaned[: -len("/anthropic")]
    if cleaned.endswith("/v1"):
        return cleaned
    return cleaned + "/v1"


def build_chat_model(
    api_key: str,
    base_url: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
) -> ChatOpenAI:
    kwargs: Dict[str, Any] = {
        "api_key": api_key or "not-configured",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    openai_base = openai_compatible_base_url(base_url)
    if openai_base:
        kwargs["base_url"] = openai_base
    return ChatOpenAI(**kwargs)


class AgentRuntime:
    """Runs general/technical/billing/escalation agents via LangChain tools."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        default_model: str,
        skill_manager: Optional[Any] = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._default_model = default_model
        self._skill_manager = skill_manager
        self._tool_manager: Optional[Any] = None
        self.stats: Dict[AgentType, AgentStats] = {agent: AgentStats() for agent in AgentType}

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        self._skill_manager = skill_manager

    def set_tool_manager(self, tool_manager: Optional[Any]) -> None:
        """注入 MCPToolManager：LangChain 工具一律经此网关执行。"""
        self._tool_manager = tool_manager

    def set_shared_tools(self, tools: Optional[Dict[str, Any]]) -> None:
        """兼容旧接口；网关模式下共享 RAG 已由 Manager 注册，此处忽略。"""
        if tools:
            logger.info("set_shared_tools 已废弃（工具经 MCPToolManager 网关），忽略 %d 个 spec", len(tools))

    def _model_name(self, agent_type: AgentType) -> str:
        profile = PROFILES[agent_type]
        env_name = f"ECHOMIND_{agent_type.value.upper()}_MODEL"
        scoped = selected_model.get()
        return (
            os.getenv(env_name, "").strip()
            or profile.model
            or scoped
            or self._default_model
        )

    def _system_prompt(self, agent_type: AgentType, req: Request) -> str:
        profile = PROFILES[agent_type]
        text = (
            f"{profile.system_prompt}\n\n[角色契约]\n"
            f"角色：{profile.role}\n"
            f"职责：{profile.mission}\n"
            f"处理流程：{' -> '.join(profile.workflow)}\n"
            f"可用输入：{'；'.join(profile.input_contract)}\n"
            f"输出要求：{'；'.join(profile.output_contract)}\n"
            f"升级条件：{'；'.join(profile.handoff_conditions) or '无'}\n"
            f"允许工具：{'、'.join(profile.tool_scope)}\n"
            "不要声称执行了未提供的查询、修改或退款操作。"
        )
        if self._skill_manager is not None:
            skill = self._skill_manager.prompt_for(req.message, agent_type.value)
            if skill:
                from memory.context_budget import clip_skills_prompt

                skill, _clipped = clip_skills_prompt(skill)
                if skill:
                    text += f"\n\n[动态 Skills]\n{skill}"
        return text

    async def run_agent(self, agent_type: AgentType, req: Request) -> AgentResponse:
        if agent_type is AgentType.ESCALATION:
            return self._escalation_response(req)

        t0 = time.monotonic()
        stats = self.stats[agent_type]
        stats.total += 1
        profile = PROFILES[agent_type]
        tools_used: List[str] = []
        tool_traces: List[Dict[str, Any]] = []
        token_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        try:
            model_name = self._model_name(agent_type)
            llm = build_chat_model(
                self._api_key,
                self._base_url,
                model_name,
                profile.temperature,
                profile.max_tokens,
            )
            lc_tools = tools_for_agent(agent_type.value, req, self._tool_manager)
            tool_map = {tool.name: tool for tool in lc_tools}
            bound = llm.bind_tools(lc_tools) if lc_tools else llm

            messages: List[Any] = [SystemMessage(content=self._system_prompt(agent_type, req))]
            # Shared CS pack from /chat, plus a role-focused entity card for specialists.
            if req.context:
                messages.append(HumanMessage(content=f"[背景信息]\n{req.context}"))
            if req.entities:
                try:
                    from memory.cs_context import _fmt_entities

                    focus = _fmt_entities(req.entities, agent_type.value, 220)
                    if focus:
                        messages.append(HumanMessage(content=f"[角色聚焦]\n{focus}"))
                except Exception:
                    messages.append(
                        HumanMessage(
                            content=f"[结构化实体]\n{json.dumps(req.entities, ensure_ascii=False)}"
                        )
                    )
            messages.append(HumanMessage(content=req.message))

            for _ in range(3):
                llm_t0 = time.monotonic()
                ai: AIMessage = await bound.ainvoke(messages)
                turn_usage = extract_message_usage(ai)
                token_usage = merge_token_usage(token_usage, turn_usage)
                tool_traces.append(
                    {
                        "kind": "llm",
                        "agent_type": agent_type.value,
                        "tool_name": "llm",
                        "tool_use_id": f"llm-{len(tool_traces)+1}",
                        "input": {"model": model_name},
                        "success": True,
                        "latency_ms": round((time.monotonic() - llm_t0) * 1000, 1),
                        "error": "",
                        "token_usage": turn_usage,
                    }
                )
                messages.append(ai)
                tool_calls = getattr(ai, "tool_calls", None) or []
                if not tool_calls:
                    content = ai.content if isinstance(ai.content, str) else str(ai.content or "")
                    ms = (time.monotonic() - t0) * 1000
                    stats.success += 1
                    stats.total_ms += ms
                    escalate = any(kw in content for kw in ("转人工", "人工客服", "escalate", "无法处理"))
                    return AgentResponse(
                        agent_type=agent_type,
                        content=content.strip() or "已处理完成。",
                        success=True,
                        latency_ms=ms,
                        escalate=escalate,
                        tools_used=tools_used,
                        tool_traces=tool_traces,
                        token_usage=token_usage,
                    )

                for call in tool_calls:
                    name = call.get("name")
                    args = call.get("args") or {}
                    call_id = call.get("id") or name
                    tool = tool_map.get(name)
                    tool_t0 = time.monotonic()
                    success = True
                    error = ""
                    result: Any
                    if tool is None:
                        success = False
                        result = {"success": False, "error": f"工具不在 {agent_type.value} 白名单"}
                        error = result["error"]
                    else:
                        try:
                            result = await tool.ainvoke(args)
                            tools_used.append(name)
                        except Exception as ex:
                            success = False
                            error = str(ex)
                            result = {"success": False, "error": error}
                    tool_traces.append(
                        {
                            "kind": "tool",
                            "agent_type": agent_type.value,
                            "tool_name": name,
                            "tool_use_id": call_id,
                            "input": dict(args),
                            "success": success,
                            "result_success": bool(result.get("success")) if isinstance(result, dict) and "success" in result else None,
                            "latency_ms": round((time.monotonic() - tool_t0) * 1000, 1),
                            "error": error,
                        }
                    )
                    messages.append(
                        ToolMessage(
                            content=json.dumps(result, ensure_ascii=False),
                            tool_call_id=call_id,
                        )
                    )

            raise RuntimeError(f"{agent_type.value} 工具调用超过最大轮数")
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            stats.total_ms += ms
            logger.error("%s LangChain 执行失败: %s", agent_type.value, ex)
            return AgentResponse(
                agent_type=agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                tools_used=tools_used,
                tool_traces=tool_traces,
                token_usage=token_usage,
            )

    def _escalation_response(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        stats = self.stats[AgentType.ESCALATION]
        stats.total += 1
        intent = req.intent.value if req.intent else "unknown"
        urgency = req.urgency.name if req.urgency else "UNKNOWN"
        entities = json.dumps(req.entities or {}, ensure_ascii=False)
        content = (
            "我已将这个问题标记为人工升级处理。\n\n"
            f"升级原因：意图={intent}，紧急度={urgency}\n"
            f"已记录信息：{entities}\n"
            "请不要发送密码、短信验证码或完整支付凭证；人工客服会根据会话记录继续核验。"
        )
        ms = (time.monotonic() - t0) * 1000
        stats.success += 1
        stats.total_ms += ms
        return AgentResponse(
            agent_type=AgentType.ESCALATION,
            content=content,
            success=True,
            latency_ms=ms,
            escalate=True,
        )

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [item for item in responses if item.success and item.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content

        evidence = "\n\n".join(
            f"[{item.agent_type.value} Agent 输出]\n{item.content}" for item in successful
        )
        prompt = (
            "你是客服 Response Composer，负责把多个专业 Agent 的结果合并成一条最终回复。"
            "以主 Agent 结论为主，去掉重复；不能补造订单/退款结果；只输出中文用户回复。\n\n"
            f"主 Agent：{successful[0].agent_type.value}\n"
            f"用户问题：{req.message}\n候选结果：\n{evidence}"
        )
        try:
            llm = build_chat_model(
                self._api_key,
                self._base_url,
                self._default_model,
                temperature=0.1,
                max_tokens=1000,
            )
            ai = await llm.ainvoke([HumanMessage(content=prompt)])
            content = ai.content if isinstance(ai.content, str) else str(ai.content or "")
            if content.strip():
                return content.strip()
        except Exception as ex:
            logger.warning("Composer 失败，使用确定性合并: %s", ex)

        return "\n\n".join(
            item.content if index == 0 else f"补充说明：\n{item.content}"
            for index, item in enumerate(successful)
        )


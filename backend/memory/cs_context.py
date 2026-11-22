"""Customer-service oriented context packing (not a generic code-agent compressor).

Design principles for this product:
1. Protect the current user turn and the routing/entity card — never drop them.
2. Intent-gate heavy layers: greetings/clarify get a thin pack; business intents get policy slots.
3. Prefer structured entities over raw chat length (order_id / amount / error_code).
4. Specialist agents see different slices (billing vs technical vs escalation).
5. Budget trim order is domain-specific, not "shrink everything evenly".
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mcp.text_chunker import estimate_tokens
from memory.context_budget import BudgetReport, _clip_to_tokens
from memory.models import MemoryContext, Message, MsgRole

_BUSINESS_GROUPS = {"billing", "technical", "account", "logistics", "order", "request", "complaint"}
_THIN_INTENTS = {"greeting", "feedback", "other", "human_handoff", "escalation"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(64, int(raw))
    except ValueError:
        return default


@dataclass
class CSContextConfig:
    total_tokens: int = 3200
    recent_turns_normal: int = 4
    recent_turns_thin: int = 2
    policy_max_tokens: int = 900
    summary_max_tokens: int = 350
    episodic_max_tokens: int = 400
    entities_max_tokens: int = 220

    @classmethod
    def from_env(cls) -> "CSContextConfig":
        return cls(
            total_tokens=_env_int("CS_CTX_TOTAL_TOKENS", 3200),
            recent_turns_normal=_env_int("CS_CTX_RECENT_TURNS", 4),
            recent_turns_thin=_env_int("CS_CTX_RECENT_TURNS_THIN", 2),
            policy_max_tokens=_env_int("CS_CTX_POLICY_TOKENS", 900),
            summary_max_tokens=_env_int("CS_CTX_SUMMARY_TOKENS", 350),
            episodic_max_tokens=_env_int("CS_CTX_EPISODIC_TOKENS", 400),
            entities_max_tokens=_env_int("CS_CTX_ENTITIES_TOKENS", 220),
        )


@dataclass
class CSPackInput:
    user_message: str
    intent: str = "other"
    intent_group: str = "other"
    urgency: str = "MEDIUM"
    confidence: float = 0.0
    entities: Dict[str, List[str]] = field(default_factory=dict)
    memory: Optional[MemoryContext] = None
    policy_snippets: Sequence[str] = field(default_factory=list)
    agent_role: str = "general"  # general | technical | billing | escalation
    mode: str = "normal"  # normal | clarify | escalation


def _fmt_recent(messages: Sequence[Message], turns: int) -> str:
    if not messages:
        return ""
    lines = ["[最近对话]"]
    for m in list(messages)[-turns:]:
        role = m.role.value if isinstance(m.role, MsgRole) else str(m.role)
        lines.append(f"{role}: {MemoryContext._clean(m.content)}")
    return "\n".join(lines)


def _fmt_entities(entities: Dict[str, List[str]], role: str, max_tokens: int) -> str:
    if not entities:
        return ""
    # Role preference: keep specialist keys first
    preferred = {
        "billing": ["order_id", "amount", "payment", "invoice", "product"],
        "technical": ["error_code", "device", "os", "app_version", "url"],
        "escalation": ["order_id", "amount", "error_code", "account"],
        "general": [],
    }.get(role, [])
    keys = list(preferred) + [k for k in entities.keys() if k not in preferred]
    payload = {}
    for key in keys:
        vals = entities.get(key) or []
        if vals:
            payload[key] = vals
    if not payload:
        return ""
    body = json.dumps(payload, ensure_ascii=False)
    body = _clip_to_tokens(body, max_tokens)
    return f"[结构化实体]\n{body}"


def _fmt_routing(inp: CSPackInput) -> str:
    card = {
        "intent": inp.intent,
        "intent_group": inp.intent_group,
        "urgency": inp.urgency,
        "confidence": round(float(inp.confidence or 0.0), 3),
        "agent_role": inp.agent_role,
        "mode": inp.mode,
    }
    return f"[路由卡片]\n{json.dumps(card, ensure_ascii=False)}"


def _fmt_policy(snippets: Sequence[str], role: str, max_tokens: int) -> str:
    items = [MemoryContext._clean(s) for s in snippets if s and str(s).strip()]
    if not items:
        return ""
    # Lightweight role filter on policy text
    if role == "billing":
        items = [s for s in items if re.search(r"退款|账单|扣款|发票|支付|金额", s)] or items
    elif role == "technical":
        items = [s for s in items if re.search(r"登录|故障|报错|401|崩溃|账号|密码", s)] or items
    block = "[政策依据]\n" + "\n".join(f"- {s}" for s in items[:4])
    return _clip_to_tokens(block, max_tokens)


def _fmt_summary(summary: str, max_tokens: int) -> str:
    summary = (summary or "").strip()
    if not summary:
        return ""
    return f"[会话摘要]\n{_clip_to_tokens(MemoryContext._clean(summary), max_tokens)}"


def _fmt_episodic(items: Sequence[str], max_tokens: int) -> str:
    cleaned = [MemoryContext._clean(h) for h in items if h and str(h).strip()][:3]
    if not cleaned:
        return ""
    block = "[相关历史]\n" + "\n".join(f"- {h}" for h in cleaned)
    return _clip_to_tokens(block, max_tokens)


def _needs_policy(inp: CSPackInput) -> bool:
    if inp.mode in {"clarify", "escalation"}:
        return False
    if inp.intent in _THIN_INTENTS and inp.intent_group not in _BUSINESS_GROUPS:
        return False
    return inp.intent_group in _BUSINESS_GROUPS or inp.intent not in _THIN_INTENTS


def _recent_turns(inp: CSPackInput, cfg: CSContextConfig) -> int:
    if inp.mode in {"clarify", "escalation"} or inp.intent in {"greeting", "feedback"}:
        return cfg.recent_turns_thin
    if inp.urgency in {"HIGH", "CRITICAL", "high", "critical"} or inp.intent_group == "complaint":
        return min(cfg.recent_turns_normal + 2, 8)
    return cfg.recent_turns_normal


def pack_customer_context(
    inp: CSPackInput,
    *,
    config: Optional[CSContextConfig] = None,
) -> Tuple[str, BudgetReport]:
    """Pack CS context layers under a token ceiling with domain trim order."""
    cfg = config or CSContextConfig.from_env()
    mem = inp.memory or MemoryContext([], [], {}, "")

    sections: Dict[str, str] = {
        "routing": _fmt_routing(inp),
        "entities": _fmt_entities(inp.entities or {}, inp.agent_role, cfg.entities_max_tokens),
        "policy": _fmt_policy(inp.policy_snippets, inp.agent_role, cfg.policy_max_tokens)
        if _needs_policy(inp)
        else "",
        "summary": _fmt_summary(mem.summary, cfg.summary_max_tokens)
        if inp.mode != "clarify"
        else "",
        "episodic": _fmt_episodic(mem.relevant_history, cfg.episodic_max_tokens)
        if inp.mode == "normal" and inp.intent not in {"greeting"}
        else "",
        "recent": _fmt_recent(mem.recent_messages, _recent_turns(inp, cfg)),
    }

    # Protected layers: routing + entities + recent (user continuity). Policy kept when business.
    # Trim order (drop/shrink first): episodic → summary → policy → older recent already limited
    order_keep = ("routing", "entities", "recent", "policy", "summary", "episodic")

    def packed() -> str:
        return "\n\n".join(sections[k] for k in order_keep if sections.get(k))

    before_parts = {k: v for k, v in sections.items() if v}
    before = estimate_tokens("\n\n".join(before_parts.values())) if before_parts else 0

    # Domain trim cascade
    if estimate_tokens(packed()) > cfg.total_tokens and sections["episodic"]:
        sections["episodic"] = ""
    if estimate_tokens(packed()) > cfg.total_tokens and sections["summary"]:
        sections["summary"] = _fmt_summary(mem.summary, max(80, cfg.summary_max_tokens // 2))
    if estimate_tokens(packed()) > cfg.total_tokens and sections["summary"]:
        sections["summary"] = ""
    if estimate_tokens(packed()) > cfg.total_tokens and sections["policy"]:
        sections["policy"] = _fmt_policy(
            inp.policy_snippets, inp.agent_role, max(120, cfg.policy_max_tokens // 2)
        )
    if estimate_tokens(packed()) > cfg.total_tokens and sections["policy"]:
        # Keep at most one short policy line for business intents
        if inp.policy_snippets:
            one = _clip_to_tokens(str(inp.policy_snippets[0]), 160)
            sections["policy"] = f"[政策依据]\n- {one}"
        else:
            sections["policy"] = ""
    if estimate_tokens(packed()) > cfg.total_tokens and sections["recent"]:
        sections["recent"] = _fmt_recent(mem.recent_messages, 2)

    text = packed()
    if estimate_tokens(text) > cfg.total_tokens:
        # Last resort: never drop routing/entities; clip recent body
        protected = "\n\n".join(
            x for x in (sections["routing"], sections["entities"]) if x
        )
        room = max(120, cfg.total_tokens - estimate_tokens(protected) - 20)
        tail = _clip_to_tokens(
            "\n\n".join(x for x in (sections["recent"], sections["policy"]) if x),
            room,
        )
        text = "\n\n".join(x for x in (protected, tail) if x)

    after = estimate_tokens(text)
    return text, BudgetReport(
        before_tokens=before,
        after_tokens=after,
        truncated=after < before,
        sections={k: estimate_tokens(v) for k, v in sections.items() if v},
    )


def specialize_for_agent(
    inp: CSPackInput,
    agent_role: str,
    *,
    config: Optional[CSContextConfig] = None,
) -> Tuple[str, BudgetReport]:
    """Re-pack the same facts for a specialist agent view."""
    specialized = CSPackInput(
        user_message=inp.user_message,
        intent=inp.intent,
        intent_group=inp.intent_group,
        urgency=inp.urgency,
        confidence=inp.confidence,
        entities=inp.entities,
        memory=inp.memory,
        policy_snippets=inp.policy_snippets,
        agent_role=agent_role,
        mode="escalation" if agent_role == "escalation" else inp.mode,
    )
    return pack_customer_context(specialized, config=config)


def pack_input_from_request(
    *,
    user_message: str,
    intent: Any,
    intent_group: str,
    urgency: Any,
    confidence: float,
    entities: Dict[str, List[str]],
    memory: MemoryContext,
    policy_snippets: Optional[Sequence[str]] = None,
    agent_role: str = "general",
    mode: str = "normal",
) -> CSPackInput:
    intent_v = getattr(intent, "value", intent) or "other"
    urgency_v = getattr(urgency, "name", None) or getattr(urgency, "value", urgency) or "MEDIUM"
    return CSPackInput(
        user_message=user_message,
        intent=str(intent_v),
        intent_group=str(intent_group or "other"),
        urgency=str(urgency_v),
        confidence=float(confidence or 0.0),
        entities=dict(entities or {}),
        memory=memory,
        policy_snippets=list(policy_snippets or []),
        agent_role=agent_role,
        mode=mode,
    )

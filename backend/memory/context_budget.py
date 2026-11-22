"""Prompt assembly budget: fit memory / skills into a token ceiling with priorities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from mcp.text_chunker import estimate_tokens
from memory.models import MemoryContext, Message, MsgRole


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


@dataclass
class BudgetConfig:
    """Token budgets for prompt sections (approximate tokens)."""

    total_memory_tokens: int = 2800
    skills_max_tokens: int = 800
    system_base_reserve: int = 600  # role contract etc. — informational
    # Fit order: higher priority kept first when trimming.
    # recent > summary > profile > history
    recent_max_messages: int = 8

    @classmethod
    def from_env(cls) -> "BudgetConfig":
        return cls(
            total_memory_tokens=_env_int("CTX_BUDGET_MEMORY_TOKENS", 2800),
            skills_max_tokens=_env_int("CTX_BUDGET_SKILLS_TOKENS", 800),
            system_base_reserve=_env_int("CTX_BUDGET_SYSTEM_RESERVE", 600),
            recent_max_messages=_env_int("CTX_BUDGET_RECENT_MESSAGES", 8),
        )


@dataclass
class BudgetReport:
    before_tokens: int
    after_tokens: int
    truncated: bool
    sections: Dict[str, int] = field(default_factory=dict)

    @property
    def saved_tokens(self) -> int:
        return max(0, self.before_tokens - self.after_tokens)

    @property
    def compression_ratio(self) -> float:
        if self.before_tokens <= 0:
            return 0.0
        return round(self.saved_tokens / self.before_tokens, 4)


def _clip_to_tokens(text: str, max_tokens: int) -> str:
    text = (text or "").strip()
    if not text or estimate_tokens(text) <= max_tokens:
        return text
    # Binary search on character length
    lo, hi = 0, len(text)
    best = ""
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cand = text[:mid].rstrip()
        if estimate_tokens(cand) <= max_tokens:
            best = cand
            lo = mid
        else:
            hi = mid - 1
    if best and not best.endswith("…"):
        best = best.rstrip("，,;； ") + "…"
    return best


def _format_recent(messages: Sequence[Message], limit: int) -> str:
    if not messages:
        return ""
    lines = ["[最近对话]"]
    for m in list(messages)[-limit:]:
        role = m.role.value if isinstance(m.role, MsgRole) else str(m.role)
        lines.append(f"{role}: {MemoryContext._clean(m.content)}")
    return "\n".join(lines)


def _format_history(items: Sequence[str], limit: int = 3) -> str:
    cleaned = [MemoryContext._clean(h) for h in items if h and str(h).strip()][:limit]
    if not cleaned:
        return ""
    return "[相关历史]\n" + "\n".join(f"- {h}" for h in cleaned)


def assemble_memory_prompt(
    ctx: MemoryContext,
    *,
    config: Optional[BudgetConfig] = None,
) -> Tuple[str, BudgetReport]:
    """
    Pack memory sections under a token ceiling.

    Priority when over budget: recent → summary → profile → history
    (drop/trim from the end of the priority list first).
    """
    cfg = config or BudgetConfig.from_env()
    summary = (ctx.summary or "").strip()
    history = list(ctx.relevant_history or [])
    profile = ctx.user_profile or {}
    recent = list(ctx.recent_messages or [])

    summary_block = f"[会话摘要]\n{MemoryContext._clean(summary)}" if summary else ""
    history_block = _format_history(history)
    profile_block = (
        f"[用户画像]\n{json.dumps(profile, ensure_ascii=False)}" if profile else ""
    )
    recent_block = _format_recent(recent, cfg.recent_max_messages)

    raw = "\n\n".join(b for b in (summary_block, history_block, profile_block, recent_block) if b)
    before = estimate_tokens(raw)

    # Start with full sections; trim low-priority first.
    sections = {
        "summary": summary_block,
        "history": history_block,
        "profile": profile_block,
        "recent": recent_block,
    }

    def packed() -> str:
        order = ("summary", "history", "profile", "recent")
        return "\n\n".join(sections[k] for k in order if sections[k])

    text = packed()
    if estimate_tokens(text) <= cfg.total_memory_tokens:
        after = estimate_tokens(text)
        return text, BudgetReport(
            before_tokens=before,
            after_tokens=after,
            truncated=before > after,
            sections={k: estimate_tokens(v) for k, v in sections.items() if v},
        )

    # 1) Drop history entirely if needed
    if estimate_tokens(packed()) > cfg.total_memory_tokens and sections["history"]:
        sections["history"] = ""

    # 2) Shrink profile
    if estimate_tokens(packed()) > cfg.total_memory_tokens and sections["profile"]:
        room = max(
            40,
            cfg.total_memory_tokens
            - estimate_tokens(sections["summary"])
            - estimate_tokens(sections["recent"]),
        )
        body = _clip_to_tokens(json.dumps(profile, ensure_ascii=False), max(40, room - 10))
        sections["profile"] = f"[用户画像]\n{body}" if body else ""

    # 3) Shrink summary
    if estimate_tokens(packed()) > cfg.total_memory_tokens and sections["summary"]:
        room = max(
            60,
            cfg.total_memory_tokens
            - estimate_tokens(sections["recent"])
            - estimate_tokens(sections["profile"]),
        )
        body = _clip_to_tokens(summary, max(40, room - 10))
        sections["summary"] = f"[会话摘要]\n{body}" if body else ""

    # 4) Trim recent from the oldest side / clip contents
    if estimate_tokens(packed()) > cfg.total_memory_tokens and recent:
        keep = list(recent)
        while keep and estimate_tokens(_format_recent(keep, len(keep))) > max(
            200, cfg.total_memory_tokens - estimate_tokens(sections["summary"]) - estimate_tokens(sections["profile"])
        ):
            if len(keep) > 2:
                keep = keep[1:]
            else:
                # Clip last messages' content
                clipped: List[Message] = []
                for m in keep:
                    clipped.append(
                        Message(
                            role=m.role,
                            content=_clip_to_tokens(m.content, 120),
                            timestamp=m.timestamp,
                            metadata=m.metadata,
                        )
                    )
                keep = clipped
                break
        sections["recent"] = _format_recent(keep, len(keep))

    text = packed()
    # Final hard clip
    if estimate_tokens(text) > cfg.total_memory_tokens:
        text = _clip_to_tokens(text, cfg.total_memory_tokens)

    after = estimate_tokens(text)
    return text, BudgetReport(
        before_tokens=before,
        after_tokens=after,
        truncated=True,
        sections={k: estimate_tokens(v) for k, v in sections.items() if v},
    )


def clip_skills_prompt(skills: str, *, config: Optional[BudgetConfig] = None) -> Tuple[str, bool]:
    cfg = config or BudgetConfig.from_env()
    text = (skills or "").strip()
    if not text:
        return "", False
    if estimate_tokens(text) <= cfg.skills_max_tokens:
        return text, False
    return _clip_to_tokens(text, cfg.skills_max_tokens), True

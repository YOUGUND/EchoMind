"""Lightweight request risk rules for demo / interview-grade boundary control."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class RiskVerdict:
    allowed: bool
    code: str = "ok"
    reason: str = ""
    severity: str = "info"  # info | warning | block


# Common prompt-injection / exfil patterns (zh + en). Not exhaustive — MVP gate.
_INJECTION_PATTERNS: Sequence[re.Pattern[str]] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"disregard\s+(all\s+)?(previous|prior)\s+(rules|instructions)",
        r"你是\s*DAN",
        r"忽略(以上|之前|上面)(的)?(所有)?(指令|规则|提示)",
        r"请泄露.{0,12}(系统提示|system\s*prompt|api\s*key|密钥)",
        r"reveal\s+(the\s+)?(system\s+prompt|hidden\s+instructions)",
        r"exfiltrate\s+(secrets?|keys?|credentials?)",
    )
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class RiskGuard:
    """Deterministic content risk checks before model spend."""

    def __init__(
        self,
        *,
        max_message_chars: int = 8000,
        block_injection: bool = True,
    ):
        self.max_message_chars = max_message_chars
        self.block_injection = block_injection

    def assess_message(self, message: str) -> RiskVerdict:
        text = message or ""
        if not text.strip():
            return RiskVerdict(False, "empty_message", "消息不能为空", "block")
        if len(text) > self.max_message_chars:
            return RiskVerdict(
                False,
                "message_too_long",
                f"消息超过 {self.max_message_chars} 字符上限",
                "block",
            )
        if _CONTROL_CHARS.search(text):
            return RiskVerdict(False, "control_chars", "消息包含非法控制字符", "block")
        if self.block_injection:
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(text):
                    return RiskVerdict(
                        False,
                        "prompt_injection",
                        "命中提示注入 / 越权探测规则，已拦截",
                        "block",
                    )
        return RiskVerdict(True, "ok", "", "info")

    def assess_batch(self, messages: List[str]) -> Optional[RiskVerdict]:
        for message in messages:
            verdict = self.assess_message(message)
            if not verdict.allowed:
                return verdict
        return None


default_risk_guard = RiskGuard()

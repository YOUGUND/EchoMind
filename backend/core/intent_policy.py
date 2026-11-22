"""Load intent fast-path / synonym policy from JSON (enterprise-tunable)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping

from core.intent_recognizer_types import IntentCategory, UrgencyLevel

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path(__file__).with_name("intent_policy.json")


@dataclass
class IntentPolicy:
    fast_path_min_confidence: float = 0.5
    vector_skip_llm_confidence: float = 0.72
    llm_timeout_seconds: float = 2.5
    llm_cache_ttl_seconds: int = 1800
    llm_cache_max_entries: int = 2000
    synonyms: Dict[str, str] = field(default_factory=dict)
    specific_patterns: Dict[IntentCategory, List[str]] = field(default_factory=dict)
    generic_patterns: Dict[IntentCategory, List[str]] = field(default_factory=dict)
    urgency_keywords: Dict[UrgencyLevel, List[str]] = field(default_factory=dict)


def _parse_intent_map(raw: Mapping[str, List[str]]) -> Dict[IntentCategory, List[str]]:
    out: Dict[IntentCategory, List[str]] = {}
    for key, values in (raw or {}).items():
        try:
            out[IntentCategory(key)] = list(values or [])
        except ValueError:
            logger.warning("intent_policy: unknown intent key %s", key)
    return out


def _parse_urgency_map(raw: Mapping[str, List[str]]) -> Dict[UrgencyLevel, List[str]]:
    mapping = {
        "critical": UrgencyLevel.CRITICAL,
        "high": UrgencyLevel.HIGH,
        "medium": UrgencyLevel.MEDIUM,
        "low": UrgencyLevel.LOW,
    }
    out: Dict[UrgencyLevel, List[str]] = {}
    for key, values in (raw or {}).items():
        level = mapping.get(str(key).lower())
        if level is None:
            logger.warning("intent_policy: unknown urgency key %s", key)
            continue
        out[level] = list(values or [])
    return out


def normalize_text(message: str, synonyms: Mapping[str, str]) -> str:
    """Apply synonym rewrites then light cleanup for matching / cache keys."""
    text = (message or "").strip()
    if not text:
        return ""
    # Longer keys first to avoid partial stomping.
    for src in sorted(synonyms.keys(), key=len, reverse=True):
        dst = synonyms[src]
        if src and src in text:
            text = text.replace(src, dst)
    return " ".join(text.split())


def load_intent_policy(path: str | Path | None = None) -> IntentPolicy:
    policy_path = Path(
        path
        or os.getenv("ECHOMIND_INTENT_POLICY_PATH", "")
        or _DEFAULT_POLICY_PATH
    )
    data: dict = {}
    if policy_path.exists():
        try:
            data = json.loads(policy_path.read_text(encoding="utf-8"))
        except Exception as ex:
            logger.warning("failed to load intent policy %s: %s", policy_path, ex)
    else:
        logger.warning("intent policy missing at %s; using defaults", policy_path)

    return IntentPolicy(
        fast_path_min_confidence=float(data.get("fast_path_min_confidence", 0.5)),
        vector_skip_llm_confidence=float(data.get("vector_skip_llm_confidence", 0.72)),
        llm_timeout_seconds=float(data.get("llm_timeout_seconds", 2.5)),
        llm_cache_ttl_seconds=int(data.get("llm_cache_ttl_seconds", 1800)),
        llm_cache_max_entries=int(data.get("llm_cache_max_entries", 2000)),
        synonyms={str(k): str(v) for k, v in (data.get("synonyms") or {}).items()},
        specific_patterns=_parse_intent_map(data.get("specific_patterns") or {}),
        generic_patterns=_parse_intent_map(data.get("generic_patterns") or {}),
        urgency_keywords=_parse_urgency_map(data.get("urgency_keywords") or {}),
    )

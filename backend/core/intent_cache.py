"""LLM-result cache: in-memory TTL, optional Redis when available."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class IntentResultCodec(Protocol):
    def dumps(self, value: Any) -> str: ...
    def loads(self, raw: str) -> Any: ...


class IntentLLMCache:
    """Only expensive LLM classifications should be stored here."""

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_entries: int = 2000,
        redis_url: Optional[str] = None,
        key_prefix: str = "echomind:intent:llm:",
    ):
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_entries = max(16, int(max_entries))
        self.key_prefix = key_prefix
        self._memory: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._redis = None
        self.hits = 0
        self.misses = 0

        url = redis_url or os.getenv("INTENT_CACHE_REDIS_URL") or os.getenv("REDIS_URL")
        # Free/memory profile should not require Redis.
        if url and os.getenv("STORAGE_MODE", "memory") != "memory":
            try:
                import redis.asyncio as redis

                self._redis = redis.from_url(url, decode_responses=True)
                logger.info("intent LLM cache: redis enabled")
            except Exception as ex:
                logger.warning("intent LLM cache: redis unavailable (%s), memory only", ex)

    async def get(self, key: str, codec: IntentResultCodec) -> Any | None:
        full = self.key_prefix + key
        if self._redis is not None:
            try:
                raw = await self._redis.get(full)
                if raw:
                    self.hits += 1
                    return codec.loads(raw)
            except Exception as ex:
                logger.warning("intent redis get failed: %s", ex)

        item = self._memory.get(full)
        if not item:
            self.misses += 1
            return None
        expires_at, raw = item
        if expires_at < time.time():
            self._memory.pop(full, None)
            self.misses += 1
            return None
        self._memory.move_to_end(full)
        self.hits += 1
        return codec.loads(raw)

    async def set(self, key: str, value: Any, codec: IntentResultCodec) -> None:
        full = self.key_prefix + key
        raw = codec.dumps(value)
        if self._redis is not None:
            try:
                await self._redis.setex(full, self.ttl_seconds, raw)
            except Exception as ex:
                logger.warning("intent redis set failed: %s", ex)

        self._memory[full] = (time.time() + self.ttl_seconds, raw)
        self._memory.move_to_end(full)
        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)

    def clear_memory(self) -> None:
        self._memory.clear()

    async def clear(self) -> None:
        self.clear_memory()
        if self._redis is None:
            return
        try:
            async for key in self._redis.scan_iter(match=self.key_prefix + "*", count=100):
                await self._redis.delete(key)
        except Exception as ex:
            logger.warning("intent redis clear failed: %s", ex)

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None

    @property
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": "redis+memory" if self._redis is not None else "memory",
            "size": len(self._memory),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "ttl_seconds": self.ttl_seconds,
        }


class IntentResultJSONCodec:
    """Serialize IntentResult without importing the recognizer module at import time."""

    def __init__(self, result_cls, category_cls, urgency_cls):
        self._result_cls = result_cls
        self._category_cls = category_cls
        self._urgency_cls = urgency_cls

    def dumps(self, value: Any) -> str:
        payload = {
            "intent": value.intent.value,
            "confidence": value.confidence,
            "urgency": value.urgency.name,
            "intent_group": value.intent_group,
            "entities": value.entities,
            "reasoning": value.reasoning,
            "latency_ms": value.latency_ms,
            "source_scores": value.source_scores,
        }
        return json.dumps(payload, ensure_ascii=False)

    def loads(self, raw: str) -> Any:
        data = json.loads(raw)
        return self._result_cls(
            intent=self._category_cls(data["intent"]),
            confidence=float(data.get("confidence") or 0.0),
            urgency=self._urgency_cls[data.get("urgency", "LOW")],
            intent_group=data.get("intent_group") or "other",
            entities=dict(data.get("entities") or {}),
            reasoning=data.get("reasoning") or "",
            latency_ms=float(data.get("latency_ms") or 0.0),
            source_scores=dict(data.get("source_scores") or {}),
        )

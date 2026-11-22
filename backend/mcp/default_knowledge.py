"""Shared default knowledge corpus loaded from config/default_knowledge.json."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "default_knowledge.json"


@lru_cache(maxsize=1)
def load_default_knowledge_docs() -> tuple:
    """Return immutable tuple of {title, content} dicts."""
    if not _DEFAULT_PATH.exists():
        logger.warning("默认知识库文件不存在: %s", _DEFAULT_PATH)
        return tuple()
    raw = json.loads(_DEFAULT_PATH.read_text(encoding="utf-8"))
    docs: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()
        if title and content:
            docs.append({"title": title, "content": content})
    logger.info("已加载默认知识库配置: %s 篇 (%s)", len(docs), _DEFAULT_PATH.name)
    return tuple(docs)


def default_knowledge_titles() -> List[str]:
    return [doc["title"] for doc in load_default_knowledge_docs()]

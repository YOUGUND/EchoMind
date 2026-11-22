"""LangGraph checkpointer factory (memory / sqlite)."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def create_checkpointer() -> Any:
    """
    CHECKPOINT_BACKEND=
      - memory (default): InMemorySaver，进程内，适合免费演示
      - sqlite: 本地文件持久化，支持重启后续跑/回溯
    """
    backend = os.getenv("CHECKPOINT_BACKEND", "memory").strip().lower()
    if backend == "sqlite":
        path = os.getenv("CHECKPOINT_SQLITE_PATH", "./data/checkpoints.sqlite")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            saver = AsyncSqliteSaver.from_conn_string(path)
            logger.info("Checkpointer: AsyncSqliteSaver (%s)", path)
            return saver
        except Exception as ex:
            logger.warning("Sqlite checkpointer 不可用，回退 memory: %s", ex)

    from langgraph.checkpoint.memory import InMemorySaver

    logger.info("Checkpointer: InMemorySaver")
    return InMemorySaver()

"""Async SQLAlchemy engine / session helpers."""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from storage.models import Base

logger = logging.getLogger(__name__)

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://epochflow:epochflow@127.0.0.1:5432/epochflow",
    )


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("数据库未初始化，请先调用 init_db()")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("数据库未初始化，请先调用 init_db()")
    return _session_factory


def _ensure_legacy_columns(conn) -> None:
    """Best-effort ADD COLUMN for DBs created before summary/archive fields."""
    from sqlalchemy import inspect, text

    inspector = inspect(conn)
    tables = set(inspector.get_table_names())
    if "conversations" in tables:
        cols = {c["name"] for c in inspector.get_columns("conversations")}
        if "summary" not in cols:
            conn.execute(text("ALTER TABLE conversations ADD COLUMN summary TEXT DEFAULT ''"))
    if "messages" in tables:
        cols = {c["name"] for c in inspector.get_columns("messages")}
        if "archived" not in cols:
            conn.execute(text("ALTER TABLE messages ADD COLUMN archived VARCHAR(8) DEFAULT '0'"))


async def init_db(url: Optional[str] = None) -> AsyncEngine:
    global _engine, _session_factory
    db_url = url or database_url()
    kwargs = {"echo": os.getenv("SQL_ECHO", "").lower() in {"1", "true", "yes"}}
    # SQLite needs special connect args for multi-thread test use.
    if db_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    _engine = create_async_engine(db_url, **kwargs)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_legacy_columns)
    logger.info("PostgreSQL/SQLAlchemy 已就绪: %s", db_url.split("@")[-1])
    return _engine


async def close_db() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def session_scope() -> AsyncIterator[AsyncSession]:
    factory = get_session_factory()
    async with factory() as session:
        yield session

"""PostgreSQL/SQLite multi memory: compress + episodic recall + budget packing."""

from __future__ import annotations

import asyncio
import importlib

import pytest

from memory.models import MsgRole
from storage.db import close_db, init_db


@pytest.fixture
def pg_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("PG_MEMORY_COMPRESS_AT", "6")
    monkeypatch.setenv("PG_MEMORY_KEEP_RECENT", "2")
    monkeypatch.setenv("PG_MEMORY_WORKING_MAX", "20")

    import memory.pg_memory as pg_mod

    importlib.reload(pg_mod)
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'mem.db'}"

    async def setup():
        await init_db(db_url)
        return pg_mod.MemoryManager(api_key="", model="test")

    mgr = asyncio.run(setup())
    yield mgr
    asyncio.run(close_db())


def test_compress_creates_summary_and_archives(pg_memory):
    async def exercise():
        user = "u-mem"
        conv = "c-mem"
        # 6 messages triggers compress (COMPRESS_AT=6)
        for i in range(6):
            await pg_memory.add_message(user, conv, MsgRole.USER if i % 2 == 0 else MsgRole.ASSISTANT, f"轮次{i} 退款审核需要多久订单号")

        ctx = await pg_memory.get_context(user, conv, query="退款")
        assert ctx.summary, "summary should be filled after compress"
        assert len(ctx.recent_messages) <= pg_memory.KEEP_RECENT
        assert ctx.relevant_history or ctx.summary
        text, report = ctx.to_prompt_text_with_budget()
        assert text
        assert report.after_tokens >= 1

    asyncio.run(exercise())

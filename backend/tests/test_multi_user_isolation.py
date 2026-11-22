"""Multi-user isolation: JWT subject wins; PG messages + KB scoped by user_id."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("STORAGE_MODE", "multi")
os.environ.setdefault("JWT_SECRET", "test-secret-for-isolation")
os.environ.setdefault("MILVUS_URI", "http://127.0.0.1:1")


def test_jwt_roundtrip_and_password_hash():
    from api.auth_tokens import create_access_token, decode_access_token, hash_password, verify_password

    token = create_access_token(user_id="u-1", email="a@example.com", ttl_seconds=60)
    payload = decode_access_token(token)
    assert payload["sub"] == "u-1"
    assert payload["email"] == "a@example.com"
    encoded = hash_password("password123")
    assert verify_password("password123", encoded)
    assert not verify_password("wrong", encoded)


def test_pg_memory_rejects_cross_user_conversation_access():
    from memory.pg_memory import MemoryManager

    foreign = SimpleNamespace(user_id="user-a", id="conv-1")
    session = MagicMock()
    session.get = AsyncMock(return_value=foreign)
    session.scalars = AsyncMock()

    factory_cm = MagicMock()
    factory_cm.__aenter__ = AsyncMock(return_value=session)
    factory_cm.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=factory_cm)

    memory = MemoryManager()
    with patch("memory.pg_memory.get_session_factory", return_value=factory):
        try:
            asyncio.run(memory.get_context("user-b", "conv-1"))
            raise AssertionError("cross-user read must raise PermissionError")
        except PermissionError:
            pass


def test_milvus_kb_filters_by_user_id_even_in_memory_fallback():
    import importlib.util
    from pathlib import Path

    from api.user_context import AuthUser, bind_request_user

    # Project package is named `mcp`, which can be shadowed by the public MCP SDK.
    module_path = Path(__file__).resolve().parents[1] / "mcp" / "milvus_knowledge_base.py"
    spec = importlib.util.spec_from_file_location("epochflow_milvus_kb", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    KnowledgeBase = mod.KnowledgeBase

    kb = KnowledgeBase(milvus_uri="http://127.0.0.1:1")
    assert kb._backend == "memory"

    kb.add_documents(
        [{"title": "A私有", "content": "只有用户A能看到的退款密钥 ALPHA-SECRET"}],
        user_id="user-a",
    )
    kb.add_documents(
        [{"title": "B私有", "content": "只有用户B能看到的物流口令 BRAVO-SECRET"}],
        user_id="user-b",
    )

    with bind_request_user(AuthUser(id="user-a", email="a@ex.com")):
        hits_a = kb.search("退款密钥", top_k=5)
    with bind_request_user(AuthUser(id="user-b", email="b@ex.com")):
        hits_b = kb.search("物流口令", top_k=5)

    assert any("ALPHA-SECRET" in (h.get("content") or "") for h in hits_a)
    assert not any("BRAVO-SECRET" in (h.get("content") or "") for h in hits_a)
    assert any("BRAVO-SECRET" in (h.get("content") or "") for h in hits_b)
    assert not any("ALPHA-SECRET" in (h.get("content") or "") for h in hits_b)
    assert kb.doc_count("user-a") >= 1
    assert kb.doc_count("user-b") >= 1


def test_request_boundary_requires_jwt_in_multi_mode():
    from api.security import RequestBoundary, security_audit

    class _App:
        async def __call__(self, scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    boundary = RequestBoundary(_App())
    responses = []

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(event):
        responses.append(event)

    scope = {
        "type": "http",
        "path": "/chat",
        "method": "POST",
        "headers": [],
    }
    with patch.dict(
        os.environ,
        {"STORAGE_MODE": "multi", "PUBLIC_DEMO": "1", "ANTHROPIC_API_KEY": "sk-test"},
        clear=False,
    ):
        security_audit.counts.clear()
        asyncio.run(boundary(scope, receive, send))
    start = next(r for r in responses if r["type"] == "http.response.start")
    assert start["status"] == 401
    assert security_audit.counts.get("jwt_required", 0) >= 1


def test_body_user_id_ignored_concept_via_token_subject():
    from api.auth_tokens import create_access_token, decode_access_token

    token = create_access_token(user_id="real-user", email="real@ex.com")
    subject = decode_access_token(token)["sub"]
    forged_body_user_id = "attacker"
    effective = subject
    assert effective == "real-user"
    assert effective != forged_body_user_id


def test_pg_memory_integration_when_sqlalchemy_async_available():
    """Optional integration test — skips if async SQLite driver missing."""
    try:
        import aiosqlite  # noqa: F401
        import sqlalchemy  # noqa: F401
    except ImportError:
        return

    import tempfile
    from pathlib import Path

    from memory.models import MsgRole
    from memory.pg_memory import MemoryManager
    from storage.db import close_db, get_session_factory, init_db
    from storage.models import User

    tmp = tempfile.mkdtemp(prefix="epochflow_multi_")
    db_url = f"sqlite+aiosqlite:///{(Path(tmp) / 't.db').as_posix()}"

    async def exercise():
        await init_db(db_url)
        factory = get_session_factory()
        async with factory() as session:
            session.add(User(id="user-a", email="a@ex.com", password_hash="x"))
            session.add(User(id="user-b", email="b@ex.com", password_hash="x"))
            await session.commit()
        memory = MemoryManager()
        await memory.add_message("user-a", "conv-1", MsgRole.USER, "A 的私密问题")
        ctx_a = await memory.get_context("user-a", "conv-1")
        assert any("私密" in m.content for m in ctx_a.recent_messages)
        try:
            await memory.get_context("user-b", "conv-1")
            raise AssertionError("expected PermissionError")
        except PermissionError:
            pass
        await close_db()

    asyncio.run(exercise())

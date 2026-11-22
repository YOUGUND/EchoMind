"""Tool manager failure paths: missing tool, validation, timeout, circuit, fallback."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from mcp.tool_manager import CircuitBreaker, CircuitState, MCPToolManager, Tool


def _manager() -> MCPToolManager:
    with patch("mcp.tool_manager.create_async_client", return_value=MagicMock()):
        return MCPToolManager(api_key="test-key", model="test-model")


def test_missing_tool_returns_structured_failure():
    manager = _manager()

    async def exercise():
        result = await manager.call("no_such_tool", {})
        assert result.success is False
        assert "不存在" in (result.error or "")

    asyncio.run(exercise())


def test_param_validation_failure_triggers_stats_and_breaker_count():
    manager = _manager()

    async def handler(params, context):
        return {"ok": True}

    manager.register(
        Tool(
            name="need_query",
            description="requires query",
            handler=handler,
            schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        )
    )

    async def exercise():
        result = await manager.call("need_query", {})
        assert result.success is False
        assert "缺少必需参数" in (result.error or "")
        stats = manager.get_stats()["need_query"]
        assert stats["total"] == 1
        assert stats["consecutive_fails"] == 1

    asyncio.run(exercise())


def test_timeout_uses_fallback_when_configured():
    manager = _manager()

    async def slow(params, context):
        await asyncio.sleep(0.2)
        return {"ok": True}

    def fallback(params, context, error):
        return {"degraded": True, "reason": error}

    manager.register(
        Tool(
            name="slow_tool",
            description="times out",
            handler=slow,
            schema={"type": "object", "properties": {}},
            timeout_s=0.05,
            fallback=fallback,
        )
    )

    async def exercise():
        result = await manager.call("slow_tool", {})
        assert result.success is True
        assert result.data["degraded"] is True
        assert "超时" in (result.error or "")

    asyncio.run(exercise())


def test_circuit_opens_after_threshold_and_serves_fallback():
    manager = _manager()
    calls = {"n": 0}

    async def always_fail(params, context):
        calls["n"] += 1
        raise RuntimeError("boom")

    def fallback(params, context, error):
        return {"fallback": True, "error": error}

    tool = Tool(
        name="flaky",
        description="fails",
        handler=always_fail,
        schema={"type": "object", "properties": {}},
        fallback=fallback,
    )
    tool.breaker = CircuitBreaker(failure_threshold=3, recovery_s=60.0)
    manager.register(tool)

    async def exercise():
        for _ in range(3):
            result = await manager.call("flaky", {}, use_cache=False)
            assert result.data["fallback"] is True
        assert tool.breaker.state is CircuitState.OPEN
        before = calls["n"]
        blocked = await manager.call("flaky", {}, use_cache=False)
        assert blocked.data["fallback"] is True
        assert "熔断" in (blocked.error or "")
        assert calls["n"] == before  # handler not invoked while open

    asyncio.run(exercise())


def test_circuit_breaker_unit_transitions():
    breaker = CircuitBreaker(failure_threshold=2, recovery_s=0.01)
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow() is False
    # wait for half-open
    import time

    time.sleep(0.02)
    assert breaker.allow() is True
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED

"""Enterprise intent path: synonyms, vector skip, LLM timeout degrade, feedback."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.intent_feedback import append_correction, feedback_path, load_corrections
from core.intent_policy import load_intent_policy, normalize_text
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


def test_synonym_normalize_maps_login_variants():
    policy = load_intent_policy()
    assert "登录" in normalize_text("用户登陆失败", policy.synonyms)


def test_synonym_enables_fast_path_without_llm():
    async def exercise():
        recognizer = IntentRecognizer(api_key="test-key")

        async def model_must_not_run(*args, **kwargs):
            raise AssertionError("synonym fast-path must not call LLM/embedding")

        recognizer._llm_recognize = model_must_not_run
        recognizer._embedding_recognize = model_must_not_run
        try:
            result = await recognizer.recognize("用户登陆失败，出现 401")
        finally:
            await recognizer.aclose()

        assert result.intent is IntentCategory.TECHNICAL_LOGIN
        assert result.source_scores.get("fast_path") == 1.0

    asyncio.run(exercise())


def test_vector_high_confidence_skips_llm():
    async def exercise():
        recognizer = IntentRecognizer(api_key="test-key")
        recognizer.policy.vector_skip_llm_confidence = 0.7

        async def fake_emb(message):
            return {"intent": IntentCategory.REFUND, "confidence": 0.91}

        async def llm_must_not_run(*args, **kwargs):
            raise AssertionError("vector skip must not call LLM")

        recognizer._embedding_recognize = fake_emb
        recognizer._llm_recognize = llm_must_not_run
        # Force past fast-path with ambiguous wording that won't hit specific patterns strongly
        # Use a message without refund keywords but we'll bypass pattern by stubbing pattern.
        recognizer._pattern_recognize = lambda message: {
            "intent": IntentCategory.OTHER,
            "confidence": 0.0,
        }
        try:
            result = await recognizer.recognize("处理一下这个事情")
        finally:
            await recognizer.aclose()

        assert result.intent is IntentCategory.REFUND
        assert result.source_scores.get("vector_skip_llm") == 1.0

    asyncio.run(exercise())


def test_llm_timeout_degrades_to_embedding_or_pattern():
    async def exercise():
        recognizer = IntentRecognizer(api_key="test-key")
        recognizer.policy.llm_timeout_seconds = 0.01
        recognizer.policy.vector_skip_llm_confidence = 0.99  # force LLM path

        async def slow_llm(message, history=None):
            await asyncio.sleep(0.05)
            return {"intent": IntentCategory.QUERY, "confidence": 0.9, "reasoning": "late"}

        async def emb(message):
            return {"intent": IntentCategory.TECHNICAL_CRASH, "confidence": 0.66}

        recognizer._llm_recognize = slow_llm
        recognizer._embedding_recognize = emb
        recognizer._pattern_recognize = lambda message: {
            "intent": IntentCategory.OTHER,
            "confidence": 0.0,
        }
        try:
            result = await recognizer.recognize("系统异常需要排查一下根因")
        finally:
            await recognizer.aclose()

        assert result.intent is IntentCategory.TECHNICAL_CRASH
        assert result.source_scores.get("llm_timeout_degraded") == 1.0

    asyncio.run(exercise())


def test_learn_persists_feedback(tmp_path, monkeypatch):
    path = tmp_path / "intent_feedback.jsonl"
    monkeypatch.setenv("ECHOMIND_INTENT_FEEDBACK_PATH", str(path))

    recognizer = IntentRecognizer(api_key="test-key")
    recognizer.learn("这笔钱怎么退回来", IntentCategory.REFUND)

    rows = load_corrections()
    assert any(msg == "这笔钱怎么退回来" and intent is IntentCategory.REFUND for msg, intent in rows)
    assert path.exists()
    line = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert line["intent"] == "refund"

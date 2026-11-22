"""Eval platform MVP: suites, store, gate (no live LLM required)."""

from __future__ import annotations

from evaluation.evaluator import EvalReport, EvalResult
from evaluation.store import (
    append_regression_cases,
    evaluate_gate,
    get_run,
    list_runs,
    save_run,
)
from evaluation.suite_loader import list_suites, load_suite


def test_list_and_load_builtin_suites():
    items = list_suites()
    ids = {item["id"] for item in items}
    assert "platform_smoke" in ids
    assert "intent_core" in ids
    assert "intent_full" in ids
    assert "dialog_core" in ids
    core = next(item for item in items if item["id"] == "intent_core")
    assert core["in_catalog"] is True
    assert core["fail_under"] == 0.75
    assert "intent" in core["tags"]

    suite = load_suite("intent_core")
    assert len(suite.intent_cases) >= 20
    assert suite.config is not None
    assert suite.config.fail_under == 0.75

    full = load_suite("intent_full")
    assert len(full.intent_cases) >= 35


def test_suite_rejects_invalid_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOMIND_EVAL_ROOT", str(tmp_path))
    suites = tmp_path / "suites"
    suites.mkdir(parents=True)
    bad = {
        "id": "bad_suite",
        "name": "bad",
        "version": "1",
        "intent_cases": [{"message": "hi", "expected_intent": "not_a_real_intent"}],
        "dialog_cases": [],
    }
    (suites / "bad_suite.json").write_text(
        __import__("json").dumps(bad, ensure_ascii=False), encoding="utf-8"
    )
    try:
        load_suite("bad_suite")
        assert False, "expected ValueError"
    except ValueError as ex:
        assert "非法" in str(ex)


def test_save_list_get_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOMIND_EVAL_ROOT", str(tmp_path))
    report = EvalReport(
        timestamp="t",
        total=1,
        passed=1,
        pass_rate=1.0,
        avg_scores={"intent_accuracy": 1.0},
        regressions=[],
        recommendations=["ok"],
        results=[
            EvalResult(test_id="intent_recognition", passed=True, scores={"accuracy": 1.0})
        ],
    )
    saved = save_run(report, suite_id="platform_smoke", suite_version="1", gate={"passed": True})
    assert saved["run_id"]
    items = list_runs(limit=5)
    assert items[0]["run_id"] == saved["run_id"]
    detail = get_run(saved["run_id"])
    assert detail["report"]["pass_rate"] == 1.0


def test_evaluate_gate_and_regression_回流(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOMIND_EVAL_ROOT", str(tmp_path))
    report = EvalReport(
        timestamp="t",
        total=1,
        passed=0,
        pass_rate=0.4,
        avg_scores={"intent_accuracy": 0.4},
        regressions=[],
        recommendations=[],
        results=[
            EvalResult(
                test_id="intent_recognition",
                passed=False,
                scores={"accuracy": 0.4},
                metadata={
                    "cases": [
                        {
                            "message": "登录一直报401",
                            "expected": "technical_login",
                            "predicted": "other",
                        }
                    ]
                },
            )
        ],
    )
    gate = evaluate_gate(report, 0.75)
    assert gate["enabled"] is True
    assert gate["passed"] is False
    added = append_regression_cases(report)
    assert added == 1
    suite_path = tmp_path / "suites" / "regressions.json"
    assert suite_path.exists()

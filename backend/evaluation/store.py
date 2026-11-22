"""Persist eval runs and optional baseline for regression gates."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.evaluator import EvalReport

_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "eval"


def eval_root() -> Path:
    override = os.getenv("ECHOMIND_EVAL_ROOT", "").strip()
    return Path(override) if override else _DEFAULT_ROOT


def runs_dir() -> Path:
    path = eval_root() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def baseline_path() -> Path:
    override = os.getenv("EVAL_BASELINE_PATH", "").strip()
    if override:
        return Path(override)
    return eval_root() / "baselines" / "current.json"


def new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def save_run(
    report: EvalReport,
    *,
    suite_id: Optional[str] = None,
    suite_version: Optional[str] = None,
    gate: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_id = new_run_id()
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "suite_id": suite_id,
        "suite_version": suite_version,
        "created_at": datetime.now().isoformat(),
        "gate": gate or {},
        "report": asdict(report),
    }
    if extra:
        payload["extra"] = extra
    path = runs_dir() / f"{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    files = sorted(runs_dir().glob("*.json"), reverse=True)
    items: List[Dict[str, Any]] = []
    for path in files[: max(1, limit)]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        report = data.get("report") or {}
        items.append(
            {
                "run_id": data.get("run_id") or path.stem,
                "suite_id": data.get("suite_id"),
                "suite_version": data.get("suite_version"),
                "created_at": data.get("created_at"),
                "pass_rate": report.get("pass_rate"),
                "total": report.get("total"),
                "passed": report.get("passed"),
                "avg_scores": report.get("avg_scores"),
                "regressions": report.get("regressions"),
                "gate": data.get("gate") or {},
                "path": str(path),
            }
        )
    return items


def get_run(run_id: str) -> Dict[str, Any]:
    path = runs_dir() / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"评测跑次不存在: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline_report(report: EvalReport) -> Path:
    path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def append_regression_cases(report: EvalReport) -> int:
    """Append failed intent/dialog snippets into regressions suite for回流."""
    path = eval_root() / "suites" / "regressions.json"
    existing: Dict[str, Any]
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    else:
        existing = {
            "id": "regressions",
            "name": "自动回流坏案例",
            "version": "1",
            "description": "由评测失败 case 自动追加，供回归套件使用",
            "intent_cases": [],
            "dialog_cases": [],
        }

    intent_cases: List[Dict[str, Any]] = list(existing.get("intent_cases") or [])
    dialog_cases: List[Dict[str, Any]] = list(existing.get("dialog_cases") or [])
    intent_keys = {(c.get("message"), c.get("expected_intent")) for c in intent_cases}
    dialog_keys = {json.dumps(c, ensure_ascii=False, sort_keys=True) for c in dialog_cases}
    added = 0

    for result in report.results:
        if result.passed:
            continue
        meta = result.metadata or {}
        if result.test_id == "intent_recognition":
            for case in meta.get("cases") or []:
                if case.get("expected") == case.get("predicted"):
                    continue
                row = {
                    "message": case.get("message"),
                    "expected_intent": case.get("expected"),
                }
                key = (row["message"], row["expected_intent"])
                if row["message"] and key not in intent_keys:
                    intent_cases.append(row)
                    intent_keys.add(key)
                    added += 1
            continue

        question = meta.get("question")
        if question:
            row = {"question": question}
            key = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if key not in dialog_keys:
                dialog_cases.append(row)
                dialog_keys.add(key)
                added += 1

    # Cap growth for demo safety.
    existing["intent_cases"] = intent_cases[-100:]
    existing["dialog_cases"] = dialog_cases[-50:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return added


def evaluate_gate(report: EvalReport, fail_under: Optional[float]) -> Dict[str, Any]:
    threshold = fail_under
    if threshold is None:
        raw = os.getenv("EVAL_FAIL_UNDER", "").strip()
        threshold = float(raw) if raw else None
    if threshold is None:
        return {"enabled": False, "passed": True, "fail_under": None, "pass_rate": report.pass_rate}

    passed = report.pass_rate >= float(threshold)
    return {
        "enabled": True,
        "passed": passed,
        "fail_under": float(threshold),
        "pass_rate": report.pass_rate,
        "reason": None if passed else f"pass_rate {report.pass_rate:.3f} < fail_under {float(threshold):.3f}",
    }

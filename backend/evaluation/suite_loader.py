"""Load versioned eval suites + suite_catalog configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.intent_recognizer_types import IntentCategory
from evaluation.evaluator import IntentTestCase

_DEFAULT_EVAL_ROOT = Path(__file__).resolve().parents[1] / "data" / "eval"
_VALID_INTENTS: Set[str] = {item.value for item in IntentCategory}


def eval_data_root() -> Path:
    override = os.getenv("ECHOMIND_EVAL_ROOT", "").strip()
    return Path(override) if override else _DEFAULT_EVAL_ROOT


def suites_dir() -> Path:
    override = os.getenv("ECHOMIND_EVAL_SUITES_DIR", "").strip()
    return Path(override) if override else eval_data_root() / "suites"


def catalog_path() -> Path:
    override = os.getenv("ECHOMIND_EVAL_CATALOG", "").strip()
    return Path(override) if override else eval_data_root() / "suite_catalog.json"


@dataclass
class SuiteConfig:
    id: str
    tags: List[str] = field(default_factory=list)
    fail_under: Optional[float] = None
    owner: str = ""
    purpose: str = ""


@dataclass
class EvalSuite:
    id: str
    name: str
    version: str
    description: str
    intent_cases: List[IntentTestCase]
    dialog_cases: List[Dict[str, Any]]
    path: Path
    config: Optional[SuiteConfig] = None

    @property
    def case_count(self) -> int:
        dialog_turns = 0
        for case in self.dialog_cases:
            turns = case.get("turns")
            if isinstance(turns, list) and turns:
                dialog_turns += len(turns)
            elif case.get("question"):
                dialog_turns += 1
        return len(self.intent_cases) + dialog_turns


def load_catalog() -> Dict[str, Any]:
    path = catalog_path()
    if not path.exists():
        return {"version": "0", "default_suite": "platform_smoke", "suites": []}
    return json.loads(path.read_text(encoding="utf-8"))


def catalog_configs() -> Dict[str, SuiteConfig]:
    data = load_catalog()
    out: Dict[str, SuiteConfig] = {}
    for row in data.get("suites") or []:
        suite_id = str(row.get("id") or "").strip()
        if not suite_id:
            continue
        out[suite_id] = SuiteConfig(
            id=suite_id,
            tags=list(row.get("tags") or []),
            fail_under=float(row["fail_under"]) if row.get("fail_under") is not None else None,
            owner=str(row.get("owner") or ""),
            purpose=str(row.get("purpose") or ""),
        )
    return out


def default_suite_id() -> str:
    return str(load_catalog().get("default_suite") or "intent_core")


def _validate_intent_cases(cases: List[Dict[str, Any]], suite_id: str) -> None:
    for idx, item in enumerate(cases):
        expected = str(item.get("expected_intent") or "").strip()
        if expected not in _VALID_INTENTS:
            raise ValueError(
                f"套件 {suite_id} 第 {idx} 条 expected_intent 非法: {expected!r}；"
                f"可选值={sorted(_VALID_INTENTS)}"
            )
        if not str(item.get("message") or "").strip():
            raise ValueError(f"套件 {suite_id} 第 {idx} 条 message 为空")


def _validate_dialog_cases(cases: List[Dict[str, Any]], suite_id: str) -> None:
    for idx, item in enumerate(cases):
        turns = item.get("turns")
        question = item.get("question")
        if isinstance(turns, list) and any(str(t).strip() for t in turns):
            continue
        if question and str(question).strip():
            continue
        raise ValueError(f"套件 {suite_id} 对话用例 {idx} 缺少 question 或 turns")


def list_suites() -> List[Dict[str, Any]]:
    root = suites_dir()
    if not root.exists():
        return []
    configs = catalog_configs()
    catalog_order = [cfg.id for cfg in configs.values()]
    disk_ids = [path.stem for path in sorted(root.glob("*.json"))]
    ordered_ids = list(dict.fromkeys([*catalog_order, *disk_ids]))

    items: List[Dict[str, Any]] = []
    for suite_id in ordered_ids:
        path = root / f"{suite_id}.json"
        if not path.exists():
            continue
        try:
            suite = load_suite(suite_id)
        except Exception:
            continue
        cfg = suite.config
        items.append(
            {
                "id": suite.id,
                "name": suite.name,
                "version": suite.version,
                "description": suite.description,
                "intent_cases": len(suite.intent_cases),
                "dialog_cases": len(suite.dialog_cases),
                "approx_units": suite.case_count,
                "path": str(suite.path),
                "tags": list(cfg.tags) if cfg else [],
                "fail_under": cfg.fail_under if cfg else None,
                "owner": cfg.owner if cfg else "",
                "purpose": cfg.purpose if cfg else "",
                "in_catalog": suite_id in configs,
            }
        )
    return items


def load_suite(suite_id: str) -> EvalSuite:
    path = suites_dir() / f"{suite_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"评测套件不存在: {suite_id} ({path})")
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_intent = list(data.get("intent_cases") or [])
    raw_dialog = list(data.get("dialog_cases") or [])
    _validate_intent_cases(raw_intent, suite_id)
    _validate_dialog_cases(raw_dialog, suite_id)

    intent_cases = [
        IntentTestCase(
            message=str(item["message"]),
            expected_intent=str(item["expected_intent"]),
            context=item.get("context"),
        )
        for item in raw_intent
    ]
    cfg = catalog_configs().get(suite_id)
    return EvalSuite(
        id=str(data.get("id") or suite_id),
        name=str(data.get("name") or suite_id),
        version=str(data.get("version") or "1"),
        description=str(data.get("description") or (cfg.purpose if cfg else "")),
        intent_cases=intent_cases,
        dialog_cases=raw_dialog,
        path=path,
        config=cfg,
    )

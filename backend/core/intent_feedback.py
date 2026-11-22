"""Persist human intent corrections for online template enrichment / later training."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from core.intent_recognizer_types import IntentCategory

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "intent_feedback.jsonl"


def feedback_path() -> Path:
    override = os.getenv("ECHOMIND_INTENT_FEEDBACK_PATH", "").strip()
    return Path(override) if override else _DEFAULT_PATH


def append_correction(message: str, correct: IntentCategory, source: str = "learn") -> None:
    path = feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.time(),
        "message": message,
        "intent": correct.value,
        "source": source,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_corrections(limit: int = 5000) -> List[Tuple[str, IntentCategory]]:
    path = feedback_path()
    if not path.exists():
        return []
    rows: List[Tuple[str, IntentCategory]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as ex:
        logger.warning("failed reading intent feedback: %s", ex)
        return []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            intent = IntentCategory(data["intent"])
            message = str(data.get("message") or "").strip()
            if message:
                rows.append((message, intent))
        except Exception:
            continue
    return rows


def merge_into_templates(
    templates: Dict[IntentCategory, List[str]],
    corrections: Iterable[Tuple[str, IntentCategory]],
    per_intent_cap: int = 40,
) -> int:
    """Append unique corrections into the in-memory few-shot / embedding template bank."""
    added = 0
    for message, intent in corrections:
        bucket = templates.setdefault(intent, [])
        if message in bucket:
            continue
        bucket.append(message)
        added += 1
        if len(bucket) > per_intent_cap:
            # Keep newest corrections.
            del bucket[: len(bucket) - per_intent_cap]
    return added

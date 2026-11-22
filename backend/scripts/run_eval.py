"""
评测中台 CLI 门禁：

  cd backend
  python -m scripts.run_eval --suite platform_smoke --fail-under 0.0

意图套件不需要完整 Orchestrator；含 dialog 的套件请走已启动的 API /eval/run。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EpochFlow eval suite with optional gate")
    parser.add_argument("--suite", default="platform_smoke", help="suite id under data/eval/suites")
    parser.add_argument("--fail-under", type=float, default=None, help="fail if pass_rate < threshold")
    parser.add_argument("--update-baseline", action="store_true", help="write baselines/current.json")
    parser.add_argument("--no-save", action="store_true", help="do not persist run json")
    parser.add_argument("--no-regressions", action="store_true", help="skip bad-case回流")
    return parser.parse_args()


async def _amain(args: argparse.Namespace) -> int:
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from evaluation.store import (
        append_regression_cases,
        baseline_path,
        evaluate_gate,
        save_baseline_report,
        save_run,
    )
    from evaluation.suite_loader import load_suite

    suite = load_suite(args.suite)
    if suite.dialog_cases:
        print(
            "该套件含 dialog_cases，CLI 冒烟仅支持意图套件。\n"
            "请使用: POST /eval/run {\"suite_id\": \"%s\"}" % suite.id,
            file=sys.stderr,
        )
        return 2

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip() or "not-configured-local-client"
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()

    recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
    evaluator = EndToEndEvaluator(
        orchestrator=None,
        recognizer=recognizer,
        api_key=api_key,
        base_url=base_url,
        model=model,
        baseline_path=str(baseline_path()),
    )
    try:
        report = await evaluator.run(intent_cases=suite.intent_cases, dialog_cases=[])
    finally:
        await recognizer.aclose()
        try:
            await evaluator._judge._client.close()
        except Exception:
            pass

    gate = evaluate_gate(report, args.fail_under)
    run_id = None
    if not args.no_save:
        saved = save_run(
            report,
            suite_id=suite.id,
            suite_version=suite.version,
            gate=gate,
            extra={"source": "scripts.run_eval"},
        )
        run_id = saved.get("run_id")
    if args.update_baseline:
        save_baseline_report(report)
    regressions_added = 0
    if not args.no_regressions:
        regressions_added = append_regression_cases(report)

    print(
        f"suite={suite.id} v{suite.version} run_id={run_id} "
        f"pass_rate={report.pass_rate:.3f} total={report.total} passed={report.passed}"
    )
    if report.regressions:
        print("regressions:")
        for item in report.regressions:
            print(f"  - {item}")
    if regressions_added:
        print(f"regressions_added={regressions_added}")
    if gate.get("enabled"):
        print(f"gate fail_under={gate.get('fail_under')} passed={gate.get('passed')}")
        if not gate.get("passed"):
            print(gate.get("reason") or "gate failed", file=sys.stderr)
            return 1
    return 0


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()

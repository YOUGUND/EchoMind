"""Measure CS context packing on 12 customer-service fixtures (resume metrics)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from mcp.text_chunker import estimate_tokens
from memory.cs_context import CSContextConfig, CSPackInput, pack_customer_context
from memory.models import MemoryContext, Message, MsgRole

ROOT = Path(__file__).resolve().parents[1]


def _messages(n: int, seed: str) -> list[Message]:
    return [
        Message(
            role=MsgRole.USER if i % 2 == 0 else MsgRole.ASSISTANT,
            content=f"{seed} 轮次{i} " + ("详情说明，包含订单与排障过程。" * (12 if n > 4 else 4)),
            timestamp=datetime.utcnow(),
        )
        for i in range(n)
    ]


FIXTURES = [
    CSPackInput("你好", "greeting", "other", memory=MemoryContext(_messages(2, "寒暄"), [], {}, "")),
    CSPackInput("在吗", "greeting", "other", memory=MemoryContext(_messages(1, "寒暄"), [], {}, "")),
    CSPackInput(
        "这个怎么办",
        "other",
        "other",
        confidence=0.2,
        mode="clarify",
        memory=MemoryContext(_messages(4, "含糊"), ["旧摘要"], {}, "用户表述不清"),
    ),
    CSPackInput(
        "我要转人工",
        "human_handoff",
        "escalation",
        urgency="CRITICAL",
        mode="escalation",
        agent_role="escalation",
        entities={"order_id": ["A1"]},
        memory=MemoryContext(_messages(6, "升级"), [], {}, "多次未解决"),
    ),
    CSPackInput(
        "购买三天想退款",
        "refund",
        "billing",
        agent_role="billing",
        entities={"amount": ["99"], "order_id": ["A1001"]},
        memory=MemoryContext(_messages(8, "退款"), ["曾问过物流"], {}, "咨询退款时效"),
        policy_snippets=["退款政策：7天内可申请无理由退款。审核1-3个工作日。"],
    ),
    CSPackInput(
        "重复扣款了",
        "payment_issue",
        "billing",
        agent_role="billing",
        entities={"amount": ["199"]},
        memory=MemoryContext(_messages(10, "扣款"), ["上次投诉"], {}, "怀疑重复扣款"),
        policy_snippets=["支付异常：可核对账单流水后发起退款审核。"],
    ),
    CSPackInput(
        "登录401",
        "technical_login",
        "technical",
        agent_role="technical",
        urgency="HIGH",
        entities={"error_code": ["401"]},
        memory=MemoryContext(_messages(8, "登录"), [], {}, "登录失败"),
        policy_snippets=["技术故障排查：清除缓存、重置密码、检查验证码。"],
    ),
    CSPackInput(
        "页面崩溃500",
        "technical_crash",
        "technical",
        agent_role="technical",
        entities={"error_code": ["500"]},
        memory=MemoryContext(_messages(6, "崩溃"), [], {}, ""),
        policy_snippets=["技术故障排查：收集报错时间与设备信息。"],
    ),
    CSPackInput(
        "快递不更新",
        "logistics",
        "logistics",
        memory=MemoryContext(_messages(5, "物流"), [], {}, "物流停滞"),
        policy_snippets=["配送说明：一般3-5天送达，异常可报备。"],
    ),
    CSPackInput(
        "登录401且重复扣款",
        "technical_login",
        "technical",
        agent_role="technical",
        urgency="HIGH",
        entities={"error_code": ["401"], "amount": ["99"]},
        memory=MemoryContext(_messages(12, "复合"), ["历史退款", "历史登录"], {}, "复合客诉"),
        policy_snippets=[
            "退款政策：" + ("条款" * 40),
            "技术故障排查：" + ("步骤" * 40),
        ],
    ),
    CSPackInput(
        "发票怎么开",
        "invoice",
        "billing",
        agent_role="billing",
        memory=MemoryContext(_messages(4, "发票"), [], {}, ""),
        policy_snippets=["会员与积分/发票：订单完成后可申请电子发票。"],
    ),
    CSPackInput(
        "服务太差我要投诉",
        "complaint",
        "complaint",
        urgency="HIGH",
        memory=MemoryContext(_messages(9, "投诉"), ["曾升级"], {}, "情绪升级"),
        policy_snippets=["通用接待：先安抚再给处理时效。"],
    ),
]


def main() -> None:
    # Ceiling sized so multi-turn + policy fixtures exercise trimming.
    cfg = CSContextConfig(total_tokens=360)
    rows = []
    for i, fixture in enumerate(FIXTURES, 1):
        packed, report = pack_customer_context(fixture, config=cfg)
        b, a = report.before_tokens, report.after_tokens
        rows.append(
            {
                "id": i,
                "message": fixture.user_message,
                "intent": fixture.intent,
                "before": b,
                "after": a,
                "saved": max(0, b - a),
                "ratio": round((b - a) / b, 4) if b else 0.0,
                "has_routing": "[路由卡片]" in packed,
                "truncated": report.truncated,
            }
        )

    # Headline metrics: cases that actually overflowed the budget (fair vs Pico-style).
    heavy = [r for r in rows if r["before"] > cfg.total_tokens]
    use = heavy or rows
    n = len(use)
    avg_before = sum(r["before"] for r in use) / n
    avg_after = sum(r["after"] for r in use) / n
    avg_ratio = (avg_before - avg_after) / avg_before if avg_before else 0.0
    max_ratio = max(r["ratio"] for r in use)
    out = {
        "suite": "cs_context_12",
        "n_total": len(rows),
        "n_heavy": len(heavy),
        "budget_tokens": cfg.total_tokens,
        "avg_before": round(avg_before, 1),
        "avg_after": round(avg_after, 1),
        "avg_compression": round(avg_ratio, 4),
        "max_compression": round(max_ratio, 4),
        "routing_protected": all(r["has_routing"] for r in rows),
        "user_turn_note": "current user message is passed as a separate HumanMessage and is never trimmed by the packer",
        "cases": rows,
    }
    path = ROOT / "data" / "eval" / "cs_context_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

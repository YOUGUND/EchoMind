"""Customer-service context packing: intent gate + protected user continuity."""

from datetime import datetime

from mcp.text_chunker import estimate_tokens
from memory.cs_context import CSContextConfig, CSPackInput, pack_customer_context
from memory.models import MemoryContext, Message, MsgRole


def _mem(n: int = 6) -> MemoryContext:
    msgs = [
        Message(
            role=MsgRole.USER if i % 2 == 0 else MsgRole.ASSISTANT,
            content=f"轮次{i} 内容关于退款和登录 " + ("详述" * 20),
            timestamp=datetime.utcnow(),
        )
        for i in range(n)
    ]
    return MemoryContext(
        recent_messages=msgs,
        relevant_history=["更早一次沟通提到订单号 A1001 与重复扣款"] * 3,
        user_profile={},
        summary=("用户多次咨询退款时效与登录 401。" * 15),
    )


def test_greeting_gets_thin_pack_without_policy():
    text, report = pack_customer_context(
        CSPackInput(
            user_message="你好",
            intent="greeting",
            intent_group="other",
            memory=_mem(),
            policy_snippets=["退款政策：7天无理由"],
            mode="normal",
        ),
        config=CSContextConfig(total_tokens=2500),
    )
    assert "[路由卡片]" in text
    assert "[政策依据]" not in text


def test_billing_keeps_entities_and_can_include_policy():
    text, _ = pack_customer_context(
        CSPackInput(
            user_message="购买三天了想退款",
            intent="refund",
            intent_group="billing",
            entities={"order_id": ["A1001"], "amount": ["99"]},
            memory=_mem(),
            policy_snippets=["退款政策：购买后7天内可申请", "登录故障排查：清除缓存"],
            agent_role="billing",
        ),
        config=CSContextConfig(total_tokens=2500),
    )
    assert "order_id" in text
    assert "[政策依据]" in text
    assert "退款" in text


def test_budget_protects_routing_card():
    text, report = pack_customer_context(
        CSPackInput(
            user_message="登录401而且重复扣款",
            intent="technical_login",
            intent_group="technical",
            urgency="HIGH",
            entities={"error_code": ["401"], "amount": ["99"]},
            memory=_mem(10),
            policy_snippets=["排障手册" + ("长文本" * 200), "退款政策" + ("细则" * 200)],
            agent_role="technical",
        ),
        config=CSContextConfig(total_tokens=500),
    )
    assert "[路由卡片]" in text
    assert estimate_tokens(text) <= 520
    assert report.after_tokens <= report.before_tokens

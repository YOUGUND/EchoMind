"""Context budget assembler + skills clip."""

from datetime import datetime

from mcp.text_chunker import estimate_tokens
from memory.context_budget import BudgetConfig, assemble_memory_prompt, clip_skills_prompt
from memory.models import MemoryContext, Message, MsgRole


def test_assemble_trims_when_over_budget():
    recent = [
        Message(role=MsgRole.USER, content=("退款问题详细描述，" * 40), timestamp=datetime.utcnow())
        for _ in range(8)
    ]
    ctx = MemoryContext(
        recent_messages=recent,
        relevant_history=["历史片段A" * 30, "历史片段B" * 30, "历史片段C" * 30],
        user_profile={"preferences": ["要发票", "晚间联系"] * 10},
        summary=("用户多次咨询退款与物流。" * 40),
    )
    cfg = BudgetConfig(total_memory_tokens=400, recent_max_messages=8)
    raw = ctx.to_prompt_text(apply_budget=False)
    text, report = assemble_memory_prompt(ctx, config=cfg)
    assert estimate_tokens(text) <= 420
    assert report.after_tokens < report.before_tokens
    assert report.after_tokens < estimate_tokens(raw)
    assert report.truncated is True
    assert report.saved_tokens > 0


def test_clip_skills_prompt():
    long_skill = "客服规范条款。" * 200
    clipped, changed = clip_skills_prompt(long_skill, config=BudgetConfig(skills_max_tokens=80))
    assert changed is True
    assert estimate_tokens(clipped) <= 90

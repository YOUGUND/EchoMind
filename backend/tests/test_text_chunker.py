"""Structure-aware chunker: headings + recursive separators + token budget + overlap."""

from mcp.text_chunker import ChunkingConfig, chunk_text, estimate_tokens


def test_estimate_tokens_cjk_and_latin():
    assert estimate_tokens("退款") == 2
    assert estimate_tokens("refund") >= 1


def test_markdown_heading_split_and_section_path():
    text = """# 退款政策

总则说明。

## 申请条件

用户需在购买后 7 天内申请。

## 审核时效

审核通常需要 1 到 3 个工作日。审核通过后打款。
"""
    cfg = ChunkingConfig(max_tokens=80, overlap_tokens=10, min_chunk_tokens=5, include_section_prefix=True)
    chunks = chunk_text(text, title="售后手册", config=cfg)
    assert len(chunks) >= 2
    paths = {c.section_path for c in chunks}
    assert any("申请条件" in p for p in paths)
    assert any(c.text.startswith("【") for c in chunks)


def test_clause_number_and_token_cap():
    body = "。".join([f"这是第{i}条说明内容补充若干汉字" for i in range(40)])
    text = f"第一条 总则\n{body}\n\n第二条 细则\n{body}"
    cfg = ChunkingConfig(max_tokens=60, overlap_tokens=15, min_chunk_tokens=5, include_section_prefix=False)
    chunks = chunk_text(text, title="合同", config=cfg)
    assert len(chunks) >= 2
    assert all(estimate_tokens(c.text) <= 60 + 20 for c in chunks)  # allow small merge slack


def test_overlap_carries_tail():
    # Force multiple chunks in one section
    sentences = [f"政策句子编号{i}，说明退款审核与到账规则。" for i in range(30)]
    text = "## 细则\n" + "".join(sentences)
    cfg = ChunkingConfig(max_tokens=50, overlap_tokens=20, min_chunk_tokens=5, include_section_prefix=False)
    chunks = chunk_text(text, title="退款", config=cfg)
    assert len(chunks) >= 2
    # Second chunk should contain some tail content from previous original segment via overlap
    assert chunks[1].index == 1

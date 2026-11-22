"""Structure-aware text chunking for knowledge ingestion.

Pipeline:
1. Split by heading / clause hierarchy (Markdown H1-H6, 章/节/条, 条款编号)
2. Recursively split oversized sections with separator cascade
3. Enforce approximate token budget
4. Apply character/token overlap between adjacent chunks
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

# Markdown ATX headings
_MD_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")

# 第X章 / 第X节 / 第X条（支持中文数字与阿拉伯数字）
_CN_SECTION = re.compile(
    r"(?m)^(\s*(?:第[一二三四五六七八九十百千零〇两\d]+[章节条款项]|[（(][一二三四五六七八九十\d]+[）)]))\.?\s*(.*)$"
)

# 1. / 1、 / （1）/ (1) 条款编号（行首）
_NUMBERED = re.compile(r"(?m)^(\s*(?:\d{1,3}[、.．]|[（(]\d{1,3}[）)]))\s*(.*)$")

# Recursive separators: larger structure first
_DEFAULT_SEPARATORS: Tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ";",
    "，",
    ",",
    " ",
    "",
)


@dataclass(frozen=True)
class TextChunk:
    text: str
    index: int
    section_path: str = ""
    heading_level: int = 0

    def with_section_prefix(self) -> str:
        if not self.section_path:
            return self.text
        return f"【{self.section_path}】\n{self.text}"


def estimate_tokens(text: str) -> int:
    """Approximate token count without a vendor tokenizer.

    Heuristic tuned for mixed CN/EN policy docs:
    - CJK ideograph ≈ 1 token
    - Latin run ≈ ceil(chars / 4)
    """
    if not text:
        return 0
    cjk = 0
    latin = 0
    for ch in text:
        o = ord(ch)
        if (
            0x4E00 <= o <= 0x9FFF
            or 0x3400 <= o <= 0x4DBF
            or 0xF900 <= o <= 0xFAFF
            or 0x3000 <= o <= 0x303F
            or 0xFF00 <= o <= 0xFFEF
        ):
            cjk += 1
        elif ch.isascii() and (ch.isalnum() or ch in "'-"):
            latin += 1
        elif ch.isspace():
            continue
        else:
            cjk += 1  # punctuation-ish budget
    return cjk + max(1, (latin + 3) // 4) if latin else cjk


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


@dataclass
class ChunkingConfig:
    max_tokens: int = 400
    overlap_tokens: int = 60
    min_chunk_tokens: int = 20
    include_section_prefix: bool = True
    separators: Sequence[str] = _DEFAULT_SEPARATORS

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        return cls(
            max_tokens=_env_int("KB_CHUNK_MAX_TOKENS", 400),
            overlap_tokens=_env_int("KB_CHUNK_OVERLAP_TOKENS", 60),
            min_chunk_tokens=_env_int("KB_CHUNK_MIN_TOKENS", 20),
            include_section_prefix=os.getenv("KB_CHUNK_SECTION_PREFIX", "true").lower()
            not in {"0", "false", "no"},
        )


def chunk_text(
    text: str,
    *,
    title: str = "",
    config: Optional[ChunkingConfig] = None,
) -> List[TextChunk]:
    """Public entry: structure → recursive split → overlap."""
    cfg = config or ChunkingConfig.from_env()
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []

    sections = _split_by_structure(cleaned, root_title=title.strip())
    raw_parts: List[Tuple[str, str, int]] = []  # text, path, level
    for section_text, path, level in sections:
        for piece in _recursive_split(section_text, cfg.max_tokens, cfg.separators):
            piece = piece.strip()
            if not piece:
                continue
            if estimate_tokens(piece) < cfg.min_chunk_tokens and raw_parts:
                # Merge tiny trailing crumbs into previous piece when same section
                prev_text, prev_path, prev_level = raw_parts[-1]
                if prev_path == path:
                    raw_parts[-1] = (f"{prev_text}\n{piece}".strip(), prev_path, prev_level)
                    continue
            raw_parts.append((piece, path, level))

    if not raw_parts:
        return []

    overlapped = _apply_overlap(raw_parts, cfg.overlap_tokens)
    chunks: List[TextChunk] = []
    for i, (body, path, level) in enumerate(overlapped):
        chunk = TextChunk(text=body, index=i, section_path=path, heading_level=level)
        if cfg.include_section_prefix:
            chunk = TextChunk(
                text=chunk.with_section_prefix(),
                index=i,
                section_path=path,
                heading_level=level,
            )
        chunks.append(chunk)
    return chunks


def chunk_texts(text: str, **kwargs) -> List[str]:
    """Convenience: return plain strings (backward compatible with old _chunk_text)."""
    return [c.text for c in chunk_text(text, **kwargs)]


def _split_by_structure(text: str, root_title: str = "") -> List[Tuple[str, str, int]]:
    """Split document into (section_body, section_path, heading_level)."""
    matches: List[Tuple[int, int, str, int]] = []  # start, end_of_heading_line, title, level

    for m in _MD_HEADING.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        matches.append((m.start(), m.end(), title, level))

    if not matches:
        for m in _CN_SECTION.finditer(text):
            title = (m.group(1) + " " + (m.group(2) or "")).strip()
            # 章=1 节=2 条/款/项=3
            marker = m.group(1)
            if "章" in marker:
                level = 1
            elif "节" in marker:
                level = 2
            else:
                level = 3
            matches.append((m.start(), m.end(), title, level))

    if not matches:
        for m in _NUMBERED.finditer(text):
            title = (m.group(1) + " " + (m.group(2) or "")).strip()
            matches.append((m.start(), m.end(), title, 4))

    if not matches:
        path = root_title or ""
        return [(text, path, 0)]

    matches.sort(key=lambda x: x[0])
    # Drop nested duplicates at same position
    dedup: List[Tuple[int, int, str, int]] = []
    seen_starts = set()
    for item in matches:
        if item[0] in seen_starts:
            continue
        seen_starts.add(item[0])
        dedup.append(item)
    matches = dedup

    stack: List[Tuple[int, str]] = []  # level, title
    sections: List[Tuple[str, str, int]] = []

    # Preamble before first heading
    first_start = matches[0][0]
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append((preamble, root_title or "前言", 0))

    for i, (start, head_end, title, level) in enumerate(matches):
        body_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        body = text[head_end:body_end].strip()
        # Keep heading line in body for context
        heading_line = text[start:head_end].strip()
        block = f"{heading_line}\n{body}".strip() if body else heading_line

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path_parts = [root_title] if root_title else []
        path_parts.extend(t for _, t in stack)
        path = " > ".join(p for p in path_parts if p)
        sections.append((block, path, level))

    return sections


def _join_parts(parts: Sequence[str], sep: str) -> str:
    if sep in {"。", "！", "？", "；", ";", "，", ","}:
        # Punctuation already kept on the left piece.
        return "".join(parts)
    return sep.join(parts)


def _recursive_split(
    text: str,
    max_tokens: int,
    separators: Sequence[str],
) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    seps = list(separators) if separators else [""]
    sep = seps[0]
    rest = seps[1:] if len(seps) > 1 else [""]

    if sep == "":
        return _hard_split_by_tokens(text, max_tokens)

    parts = _split_with_separator(text, sep)
    if len(parts) <= 1:
        return _recursive_split(text, max_tokens, rest)

    chunks: List[str] = []
    current: List[str] = []
    for part in parts:
        trial = _join_parts(current + [part], sep) if current else part
        if estimate_tokens(trial) <= max_tokens:
            current.append(part)
            continue
        if current:
            chunks.append(_join_parts(current, sep).strip())
            current = []
        if estimate_tokens(part) > max_tokens:
            chunks.extend(_recursive_split(part, max_tokens, rest))
        else:
            current = [part]

    if current:
        chunks.append(_join_parts(current, sep).strip())
    return [c for c in chunks if c]


def _split_with_separator(text: str, sep: str) -> List[str]:
    if sep == "":
        return list(text)
    if sep in {"。", "！", "？", "；", ";", "，", ","}:
        # Keep punctuation on the left piece
        parts: List[str] = []
        buf = ""
        for ch in text:
            buf += ch
            if ch == sep:
                parts.append(buf)
                buf = ""
        if buf:
            parts.append(buf)
        return [p for p in parts if p.strip()]
    return [p for p in text.split(sep) if p != "" or sep == ""]


def _hard_split_by_tokens(text: str, max_tokens: int) -> List[str]:
    """Last-resort windowing by estimated tokens."""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    chunks: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if estimate_tokens("".join(buf)) >= max_tokens:
            chunks.append("".join(buf).strip())
            buf = []
    if buf:
        chunks.append("".join(buf).strip())
    return [c for c in chunks if c]


def _apply_overlap(
    parts: Sequence[Tuple[str, str, int]],
    overlap_tokens: int,
) -> List[Tuple[str, str, int]]:
    if overlap_tokens <= 0 or len(parts) <= 1:
        return list(parts)

    out: List[Tuple[str, str, int]] = []
    prev_text = ""
    prev_path = None
    for i, (text, path, level) in enumerate(parts):
        if i == 0 or not prev_text or path != prev_path:
            out.append((text, path, level))
            prev_text = text
            prev_path = path
            continue
        prefix = _tail_by_tokens(prev_text, overlap_tokens)
        merged = f"{prefix}\n{text}".strip() if prefix and not text.startswith(prefix) else text
        out.append((merged, path, level))
        prev_text = text  # overlap from original section body, not already-overlapped text
        prev_path = path
    return out


def _tail_by_tokens(text: str, token_budget: int) -> str:
    if estimate_tokens(text) <= token_budget:
        return text.strip()
    # Walk backward in coarse steps
    chars = list(text)
    lo, hi = 0, len(chars)
    best = ""
    while lo < hi:
        mid = (lo + hi) // 2
        cand = "".join(chars[mid:]).strip()
        if estimate_tokens(cand) <= token_budget:
            best = cand
            hi = mid
        else:
            lo = mid + 1
    # Prefer breaking at punctuation if possible
    for sep in ("。", "！", "？", "\n", "；", "，", " "):
        idx = best.find(sep)
        if idx != -1 and idx + 1 < len(best):
            return best[idx + 1 :].strip()
    return best


def chunk_documents(
    documents: Iterable[dict],
    *,
    config: Optional[ChunkingConfig] = None,
) -> List[Tuple[str, str, TextChunk]]:
    """Expand docs into (title, content_for_embed, chunk) triples."""
    cfg = config or ChunkingConfig.from_env()
    rows: List[Tuple[str, str, TextChunk]] = []
    for doc in documents:
        title = str(doc.get("title") or "")
        content = str(doc.get("content") or "")
        for chunk in chunk_text(content, title=title, config=cfg):
            rows.append((title, chunk.text, chunk))
    return rows

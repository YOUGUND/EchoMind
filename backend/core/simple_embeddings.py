"""Deterministic embeddings for Milvus multi-user KB (no external model required)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence


def embed_texts(texts: Sequence[str], dim: int = 384) -> List[List[float]]:
    return [_embed_one(text or "", dim) for text in texts]


def _embed_one(text: str, dim: int) -> List[float]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    vec = [0.0] * dim
    if not normalized:
        return vec
    tokens = set()
    for n in (2, 3, 4):
        if len(normalized) >= n:
            tokens.update(normalized[i : i + n] for i in range(len(normalized) - n + 1))
    if not tokens:
        tokens.add(normalized)
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

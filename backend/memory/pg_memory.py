"""PostgreSQL working memory for STORAGE_MODE=multi (+ summary compress / light episodic)."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import or_, select

from memory.models import MemoryContext, Message, MsgRole
from storage.db import get_session_factory
from storage.models import Conversation, MemoryEpisode
from storage.models import Message as DbMessage

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


class MemoryManager:
    """Persists conversation turns in PostgreSQL, scoped by user_id.

    Long-context path (multi):
      - working set: non-archived recent messages
      - compress when len >= COMPRESS_AT: LLM/heuristic summary → Conversation.summary
      - archive older turns + store MemoryEpisode for light recall
      - get_context: summary + keyword episodic hits + recent working messages
    """

    WORKING_MAX = _env_int("PG_MEMORY_WORKING_MAX", 20)
    COMPRESS_AT = _env_int("PG_MEMORY_COMPRESS_AT", 12)
    KEEP_RECENT = _env_int("PG_MEMORY_KEEP_RECENT", 5)
    SUMMARY_MAX_CHARS = _env_int("PG_MEMORY_SUMMARY_MAX_CHARS", 600)
    EPISODE_TOP_K = _env_int("PG_MEMORY_EPISODE_TOP_K", 3)

    def __init__(self, **kwargs: Any):
        self._kwargs = kwargs
        self._client = kwargs.get("client")
        self._model = kwargs.get("model") or os.getenv("ANTHROPIC_MODEL", "deepseek-chat")
        if self._client is None:
            api_key = (kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or "").strip()
            if api_key:
                try:
                    from core.llm_utils import create_async_client

                    self._client = create_async_client(
                        api_key=api_key,
                        base_url=kwargs.get("base_url") or os.getenv("ANTHROPIC_BASE_URL") or None,
                    )
                except Exception as ex:
                    logger.warning("PG Memory 摘要客户端不可用: %s", ex)

    async def _ensure_conversation(self, session, user_id: str, conv_id: str) -> Conversation:
        row = await session.get(Conversation, conv_id)
        if row is None:
            row = Conversation(id=conv_id, user_id=user_id, title="", summary="")
            session.add(row)
            await session.flush()
            return row
        if row.user_id != user_id:
            raise PermissionError("无权访问该会话")
        return row

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role: MsgRole,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        del metadata  # MVP: not persisted
        factory = get_session_factory()
        async with factory() as session:
            await self._ensure_conversation(session, user_id, conv_id)
            session.add(
                DbMessage(
                    conversation_id=conv_id,
                    user_id=user_id,
                    role=role.value if isinstance(role, MsgRole) else str(role),
                    content=(content or "")[:8000],
                    archived="0",
                )
            )
            conv = await session.get(Conversation, conv_id)
            if conv is not None:
                conv.updated_at = datetime.utcnow()
            await session.commit()

        # Compress outside the write transaction to keep commits short.
        try:
            await self._maybe_compress(user_id, conv_id)
        except Exception as ex:
            logger.warning("工作记忆压缩失败 user=%s conv=%s: %s", user_id, conv_id, ex)

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        factory = get_session_factory()
        async with factory() as session:
            conv = await session.get(Conversation, conv_id)
            if conv is not None and conv.user_id != user_id:
                raise PermissionError("无权访问该会话")
            summary = (conv.summary if conv is not None else "") or ""

            result = await session.scalars(
                select(DbMessage)
                .where(
                    DbMessage.user_id == user_id,
                    DbMessage.conversation_id == conv_id,
                    or_(DbMessage.archived == "0", DbMessage.archived.is_(None)),
                )
                .order_by(DbMessage.created_at.asc())
            )
            rows = list(result.all())[-self.WORKING_MAX :]
            messages = [
                Message(
                    role=MsgRole(row.role) if row.role in MsgRole._value2member_map_ else MsgRole.USER,
                    content=row.content,
                    timestamp=row.created_at or datetime.utcnow(),
                )
                for row in rows
            ]
            episodes = await self._recall_episodes(session, user_id, conv_id, query or "")
            return MemoryContext(
                recent_messages=messages,
                relevant_history=episodes,
                user_profile={},
                summary=summary,
            )

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        # Profiles remain out of scope for multi MVP (budget reserved in assembler).
        return None

    async def close(self) -> None:
        return None

    async def _maybe_compress(self, user_id: str, conv_id: str) -> None:
        factory = get_session_factory()
        async with factory() as session:
            result = await session.scalars(
                select(DbMessage)
                .where(
                    DbMessage.user_id == user_id,
                    DbMessage.conversation_id == conv_id,
                    or_(DbMessage.archived == "0", DbMessage.archived.is_(None)),
                )
                .order_by(DbMessage.created_at.asc())
            )
            rows = list(result.all())
            if len(rows) < self.COMPRESS_AT:
                return

            keep_n = min(self.KEEP_RECENT, len(rows) - 1)
            to_archive = rows[:-keep_n]
            keep = rows[-keep_n:]
            if not to_archive:
                return

            transcript = "\n".join(f"{r.role}: {r.content}" for r in to_archive)
            piece = await self._summarize(transcript)
            conv = await self._ensure_conversation(session, user_id, conv_id)
            merged = await self._merge_summary(conv.summary or "", piece)
            conv.summary = merged[: self.SUMMARY_MAX_CHARS * 2]
            conv.updated_at = datetime.utcnow()

            session.add(
                MemoryEpisode(
                    user_id=user_id,
                    conversation_id=conv_id,
                    summary=piece[:1000],
                    content=transcript[:12000],
                )
            )
            for row in to_archive:
                row.archived = "1"

            # Touch keep rows so they stay active working set
            for row in keep:
                row.archived = "0"

            await session.commit()
            logger.info(
                "PG 工作记忆已压缩 user=%s conv=%s archived=%d keep=%d summary_len=%d",
                user_id,
                conv_id,
                len(to_archive),
                len(keep),
                len(conv.summary or ""),
            )

    async def _summarize(self, transcript: str) -> str:
        text = (transcript or "").strip()
        if not text:
            return ""
        # Prefer Anthropic-compatible client when injected / configured.
        client = self._client
        if client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if api_key:
                try:
                    from core.llm_utils import create_async_client

                    client = create_async_client(
                        api_key=api_key,
                        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
                    )
                except Exception as ex:
                    logger.warning("创建摘要客户端失败，改用启发式: %s", ex)
                    client = None
        if client is not None:
            try:
                from core.llm_utils import extract_text_content

                prompt = f"用 2-3 句话总结以下客服对话的关键事实、未解决问题与用户诉求：\n{text[:6000]}"
                resp = await client.messages.create(
                    model=self._model,
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = extract_text_content(resp.content).strip()
                if summary:
                    return summary[: self.SUMMARY_MAX_CHARS]
            except Exception as ex:
                logger.warning("LLM 摘要失败，改用启发式: %s", ex)
        # Heuristic fallback: keep head+tail snippets
        flat = " ".join(text.split())
        if len(flat) <= self.SUMMARY_MAX_CHARS:
            return flat
        head = flat[: self.SUMMARY_MAX_CHARS // 2]
        tail = flat[-(self.SUMMARY_MAX_CHARS // 2) :]
        return f"{head} … {tail}"

    async def _merge_summary(self, old: str, new: str) -> str:
        old = (old or "").strip()
        new = (new or "").strip()
        if not old:
            return new
        if not new:
            return old
        client = self._client
        if client is not None:
            try:
                from core.llm_utils import extract_text_content

                prompt = (
                    f"将下面两段客服会话摘要合并为不超过 {self.SUMMARY_MAX_CHARS} 字的一段中文摘要，"
                    "只输出摘要正文。\n\n旧摘要:\n"
                    f"{old}\n\n新增摘要:\n{new}"
                )
                resp = await client.messages.create(
                    model=self._model,
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                merged = extract_text_content(resp.content).strip()
                if merged:
                    return merged[: self.SUMMARY_MAX_CHARS]
            except Exception as ex:
                logger.warning("合并摘要失败: %s", ex)
        joined = f"{old}；{new}"
        return joined[: self.SUMMARY_MAX_CHARS]

    async def _recall_episodes(
        self,
        session,
        user_id: str,
        conv_id: str,
        query: str,
    ) -> List[str]:
        result = await session.scalars(
            select(MemoryEpisode)
            .where(
                MemoryEpisode.user_id == user_id,
                MemoryEpisode.conversation_id == conv_id,
            )
            .order_by(MemoryEpisode.created_at.desc())
            .limit(20)
        )
        rows = list(result.all())
        if not rows:
            return []
        q = (query or "").strip().lower()
        scored = []
        for row in rows:
            blob = f"{row.summary}\n{row.content}".lower()
            score = 0
            if q:
                for token in [t for t in q.replace("，", " ").replace(",", " ").split() if len(t) >= 2]:
                    if token in blob:
                        score += 2
                # char bigrams for CN
                for i in range(max(0, len(q) - 1)):
                    gram = q[i : i + 2]
                    if gram.strip() and gram in blob:
                        score += 1
            scored.append((score, row))
        scored.sort(key=lambda x: (x[0], x[1].created_at), reverse=True)
        out: List[str] = []
        for score, row in scored[: self.EPISODE_TOP_K]:
            text = (row.summary or "").strip() or (row.content or "")[:240]
            if text:
                out.append(text)
        return out

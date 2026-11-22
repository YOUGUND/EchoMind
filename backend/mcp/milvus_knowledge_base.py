"""Milvus-backed knowledge base with per-user isolation (STORAGE_MODE=multi)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from api.user_context import get_request_user
from core.simple_embeddings import embed_texts
from mcp.default_knowledge import default_knowledge_titles, load_default_knowledge_docs
from mcp.text_chunker import chunk_text

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_chunks"
EMBED_DIM = 384


class _InMemoryStore:
    """Process-local fallback when Milvus is unavailable (dev / tests)."""

    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def add(self, *, ids, documents, user_ids, titles, chunk_indexes, vectors):
        for i, doc_id in enumerate(ids):
            self.rows.append(
                {
                    "id": doc_id,
                    "document": documents[i],
                    "user_id": user_ids[i],
                    "title": titles[i],
                    "chunk_index": chunk_indexes[i],
                    "vector": vectors[i],
                }
            )

    def search(self, *, user_id: str, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        scored = []
        for row in self.rows:
            if row["user_id"] != user_id:
                continue
            score = _dot(row["vector"], query_vector)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        items = []
        for score, row in scored[:top_k]:
            items.append(
                {
                    "title": row["title"],
                    "content": row["document"],
                    "score": round(float(score), 4),
                    "chunk": row["chunk_index"],
                }
            )
        return items

    def count(self, user_id: Optional[str] = None) -> int:
        if user_id is None:
            return len(self.rows)
        return sum(1 for row in self.rows if row["user_id"] == user_id)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class KnowledgeBase:
    """
    Per-user RAG store.

    Prefer Milvus with scalar filter `user_id == current`.
    Falls back to in-memory vectors if Milvus cannot be reached.
    """

    def __init__(
        self,
        milvus_uri: Optional[str] = None,
        **_: Any,
    ):
        self._uri = (milvus_uri or os.getenv("MILVUS_URI") or "http://127.0.0.1:19530").strip()
        self._backend = "memory"
        self._memory = _InMemoryStore()
        self._collection = None
        self._connect()

    def _connect(self) -> None:
        try:
            from pymilvus import (  # type: ignore
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )

            connections.connect(alias="default", uri=self._uri, timeout=3)
            if not utility.has_collection(COLLECTION_NAME):
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="chunk_index", dtype=DataType.INT64),
                    FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=4000),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
                ]
                schema = CollectionSchema(fields, description="EpochFlow per-user knowledge chunks")
                collection = Collection(name=COLLECTION_NAME, schema=schema)
                collection.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "IP",
                        "params": {"nlist": 128},
                    },
                )
                # Optional partition-key style isolation via scalar index on user_id.
                try:
                    collection.create_index(field_name="user_id", index_name="idx_user_id")
                except Exception:
                    pass
            self._collection = Collection(COLLECTION_NAME)
            self._collection.load()
            self._backend = "milvus"
            logger.info("知识库已连接 Milvus: %s collection=%s", self._uri, COLLECTION_NAME)
        except Exception as ex:
            self._collection = None
            self._backend = "memory"
            logger.warning("Milvus 不可用，使用进程内向量回退（仍按 user_id 隔离）: %s", ex)

    def _require_user_id(self, user_id: Optional[str]) -> str:
        if user_id:
            return str(user_id)
        user = get_request_user()
        if user is None:
            raise PermissionError("知识库操作需要登录用户")
        return user.id

    def add_documents(
        self,
        documents: List[Dict[str, str]],
        user_id: Optional[str] = None,
    ) -> int:
        uid = self._require_user_id(user_id)
        ids: List[str] = []
        docs: List[str] = []
        titles: List[str] = []
        chunk_indexes: List[int] = []
        user_ids: List[str] = []

        for doc in documents:
            title = doc.get("title", "") or ""
            content = doc.get("content", "") or ""
            for piece in chunk_text(content, title=title):
                body = piece.text[:3900]
                doc_id = hashlib.md5(
                    f"{uid}:{title}:{piece.index}:{piece.section_path}:{body[:80]}".encode()
                ).hexdigest()
                ids.append(doc_id)
                docs.append(body)
                titles.append(title[:480])
                chunk_indexes.append(piece.index)
                user_ids.append(uid)

        if not ids:
            return 0

        vectors = embed_texts(docs, dim=EMBED_DIM)
        if self._backend == "milvus" and self._collection is not None:
            self._collection.insert([ids, user_ids, titles, chunk_indexes, docs, vectors])
            self._collection.flush()
        else:
            self._memory.add(
                ids=ids,
                documents=docs,
                user_ids=user_ids,
                titles=titles,
                chunk_indexes=chunk_indexes,
                vectors=vectors,
            )
        logger.info("用户 %s 导入 %d 个知识片段 (%s)", uid, len(ids), self._backend)
        return len(ids)

    async def add_documents_async(
        self,
        documents: List[Dict[str, str]],
        user_id: Optional[str] = None,
    ) -> int:
        return await asyncio.to_thread(self.add_documents, documents, user_id)

    def search(self, query: str, top_k: int = 5, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        uid = self._require_user_id(user_id)
        vector = embed_texts([query], dim=EMBED_DIM)[0]
        if self._backend == "milvus" and self._collection is not None:
            # Always filter by user_id — this is the isolation boundary.
            expr = f'user_id == "{uid}"'
            results = self._collection.search(
                data=[vector],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=max(1, top_k),
                expr=expr,
                output_fields=["title", "document", "chunk_index", "user_id"],
            )
            items: List[Dict[str, Any]] = []
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    # Defense in depth: ignore any hit that slipped without matching user.
                    if str(entity.get("user_id")) != uid:
                        continue
                    items.append(
                        {
                            "title": entity.get("title") or "",
                            "content": entity.get("document") or "",
                            "score": round(float(hit.distance), 4),
                            "chunk": int(entity.get("chunk_index") or 0),
                        }
                    )
            return items
        return self._memory.search(user_id=uid, query_vector=vector, top_k=top_k)

    async def search_async(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.search, query, top_k, user_id)

    def doc_count(self, user_id: Optional[str] = None) -> int:
        uid = user_id
        if uid is None:
            user = get_request_user()
            uid = user.id if user else None
        if self._backend == "milvus" and self._collection is not None and uid:
            # Query by filter count via query iterator would be heavier; approximate with query.
            rows = self._collection.query(expr=f'user_id == "{uid}"', output_fields=["id"], limit=16384)
            return len(rows)
        return self._memory.count(uid)

    async def doc_count_async(self, user_id: Optional[str] = None) -> int:
        return await asyncio.to_thread(self.doc_count, user_id)

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        query = params.get("query", "")
        top_k = int(params.get("top_k") or 5)
        # user_id comes from request ContextVar set by API layer — never from tool args.
        return await self.search_async(query, top_k=top_k)

    def _delete_titles(self, user_id: str, titles: List[str]) -> int:
        """Remove chunks for the given titles so default corpus can be upserted."""
        if not titles:
            return 0
        removed = 0
        if self._backend == "milvus" and self._collection is not None:
            # Escape quotes in titles for Milvus expr.
            safe = [t.replace('\\', '\\\\').replace('"', '\\"') for t in titles]
            title_list = ", ".join(f'"{t}"' for t in safe)
            expr = f'user_id == "{user_id}" and title in [{title_list}]'
            try:
                self._collection.delete(expr)
                self._collection.flush()
                removed = 1  # Milvus delete does not always return count
            except Exception as ex:
                logger.warning("删除默认知识标题失败 user=%s: %s", user_id, ex)
        else:
            before = len(self._memory.rows)
            title_set = set(titles)
            self._memory.rows = [
                row
                for row in self._memory.rows
                if not (row.get("user_id") == user_id and row.get("title") in title_set)
            ]
            removed = before - len(self._memory.rows)
        return removed

    def ensure_default_docs(self, user_id: str, *, force: bool = False) -> int:
        """
        Upsert config/default_knowledge.json into the user's private KB.

        - force=False: skip when user already has all default titles
        - force=True / missing titles: delete those titles then re-import full corpus
        """
        docs = [dict(item) for item in load_default_knowledge_docs()]
        if not docs:
            return 0
        titles = [doc["title"] for doc in docs]
        existing = set(self._titles_for_user(user_id))
        missing = [t for t in titles if t not in existing]
        if not force and not missing and existing.issuperset(titles):
            return 0
        self._delete_titles(user_id, titles)
        return self.add_documents(docs, user_id=user_id)

    def _titles_for_user(self, user_id: str) -> List[str]:
        if self._backend == "milvus" and self._collection is not None:
            try:
                rows = self._collection.query(
                    expr=f'user_id == "{user_id}"',
                    output_fields=["title"],
                    limit=16384,
                )
                return sorted({str(r.get("title") or "") for r in rows if r.get("title")})
            except Exception as ex:
                logger.warning("列举用户知识标题失败: %s", ex)
                return []
        return sorted(
            {
                str(row.get("title") or "")
                for row in self._memory.rows
                if row.get("user_id") == user_id and row.get("title")
            }
        )

    async def ensure_default_docs_async(self, user_id: str, *, force: bool = False) -> int:
        return await asyncio.to_thread(self.ensure_default_docs, user_id, force=force)

    def _rows_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        if self._backend == "milvus" and self._collection is not None:
            try:
                return list(
                    self._collection.query(
                        expr=f'user_id == "{user_id}"',
                        output_fields=["id", "title", "document", "chunk_index"],
                        limit=16384,
                    )
                )
            except Exception as ex:
                logger.warning("读取用户知识片段失败: %s", ex)
                return []
        return [row for row in self._memory.rows if row.get("user_id") == user_id]

    def list_documents(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Group chunks by title for the knowledge UI (same shape as free_knowledge)."""
        uid = self._require_user_id(user_id)
        seed_titles = set(default_knowledge_titles())
        groups: Dict[str, Dict[str, Any]] = {}
        for row in self._rows_for_user(uid):
            title = str(row.get("title") or "")
            content = str(row.get("document") or row.get("content") or "")
            doc_id = hashlib.sha256(f"{uid}\0{title}".encode()).hexdigest()[:24]
            group = groups.setdefault(
                doc_id,
                {
                    "id": doc_id,
                    "title": title,
                    "chunks": 0,
                    "content": "",
                    "seed": title in seed_titles,
                },
            )
            group["chunks"] += 1
            if content:
                group["content"] += content + "\n"
        return list(groups.values())

    def delete_document(self, doc_id: str, user_id: Optional[str] = None) -> int:
        uid = self._require_user_id(user_id)
        target = None
        for doc in self.list_documents(user_id=uid):
            if doc.get("id") == doc_id:
                target = doc
                break
        if target is None:
            return 0
        title = target.get("title") or ""
        before = self.doc_count(uid)
        self._delete_titles(uid, [title])
        after = self.doc_count(uid)
        removed = max(0, before - after)
        return removed or (1 if before else 0)

    async def list_documents_async(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.list_documents, user_id)

    async def delete_document_async(self, doc_id: str, user_id: Optional[str] = None) -> int:
        return await asyncio.to_thread(self.delete_document, doc_id, user_id)

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Backward-compatible wrapper; prefers structure-aware token chunker."""
        from mcp.text_chunker import ChunkingConfig, chunk_texts

        # Map legacy char budget ≈ token budget for callers that still pass chunk_size.
        cfg = ChunkingConfig.from_env()
        cfg.max_tokens = max(32, int(chunk_size))
        return chunk_texts(text, config=cfg)



"""
端到端意图识别（企业落地版）

分流顺序：
  1. 同义词规范化 + 配置化关键词快路径
  2. 仅缓存「走过 LLM」的结果（内存 TTL；full 模式可 Redis）
  3. 向量分类高置信 → 跳过 LLM（扛大部分流量）
  4. LLM（超时降级到向量+关键词）
  5. 人工纠错写入 jsonl，启动时回流到模板库
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from core.intent_cache import IntentLLMCache, IntentResultJSONCodec
from core.intent_feedback import append_correction, load_corrections, merge_into_templates
from core.intent_policy import IntentPolicy, load_intent_policy, normalize_text
from core.intent_recognizer_types import IntentCategory, IntentResult, UrgencyLevel
from core.llm_utils import create_async_client, extract_text_content

logger = logging.getLogger(__name__)

# Backward-compatible re-exports
__all__ = [
    "IntentCategory",
    "UrgencyLevel",
    "IntentResult",
    "IntentRecognizer",
]


# ── Few-shot / Embedding 模板（运行期可被 learn 与回流扩充）────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY: ["我的订单状态是什么？", "如何重置密码？", "快递什么时候到？"],
    IntentCategory.COMPLAINT: ["等了好几个小时！", "服务太差了！", "一直没人处理！"],
    IntentCategory.REQUEST: ["帮我取消订单", "我需要修改地址", "请协助退款"],
    IntentCategory.GREETING: ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉！", "转人工客服", "找你们经理"],
    IntentCategory.TECHNICAL: ["应用一直崩溃", "无法登录", "出现500错误"],
    IntentCategory.BILLING: ["为什么扣了两次款？", "申请退款", "发票问题"],
    IntentCategory.ACCOUNT: ["修改邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK: ["服务很棒！", "非常满意", "给个好评"],
    IntentCategory.ORDER_STATUS: ["我的订单现在是什么状态？", "订单有没有发货？", "订单处理到哪一步了？"],
    IntentCategory.LOGISTICS: ["快递什么时候到？", "物流一直不更新", "配送要多久？"],
    IntentCategory.REFUND: ["我要申请退款", "退货退款怎么处理？", "退款多久到账？"],
    IntentCategory.INVOICE: ["帮我开发票", "发票抬头怎么改？", "电子发票在哪里？"],
    IntentCategory.PAYMENT_ISSUE: ["为什么重复扣款？", "支付失败怎么办？", "这个月多扣了钱"],
    IntentCategory.ACCOUNT_SECURITY: ["账户被盗了", "发现异常登录", "我要重置密码"],
    IntentCategory.TECHNICAL_LOGIN: ["登录一直报401", "验证码收不到", "无法登录账号"],
    IntentCategory.TECHNICAL_CRASH: ["应用一直崩溃", "页面报500错误", "系统闪退"],
    IntentCategory.HUMAN_HANDOFF: ["转人工客服", "我要找人工", "请升级处理"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.ORDER_STATUS,
    IntentCategory.LOGISTICS,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.ACCOUNT_SECURITY,
    IntentCategory.TECHNICAL_LOGIN,
    IntentCategory.TECHNICAL_CRASH,
    IntentCategory.HUMAN_HANDOFF,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.BILLING,
    IntentCategory.TECHNICAL,
    IntentCategory.ACCOUNT,
    IntentCategory.ESCALATION,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.ORDER_STATUS: IntentCategory.QUERY,
    IntentCategory.LOGISTICS: IntentCategory.QUERY,
    IntentCategory.REFUND: IntentCategory.BILLING,
    IntentCategory.INVOICE: IntentCategory.BILLING,
    IntentCategory.PAYMENT_ISSUE: IntentCategory.BILLING,
    IntentCategory.ACCOUNT_SECURITY: IntentCategory.ACCOUNT,
    IntentCategory.TECHNICAL_LOGIN: IntentCategory.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: IntentCategory.TECHNICAL,
    IntentCategory.HUMAN_HANDOFF: IntentCategory.ESCALATION,
}


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """配置化快路径 + 向量先行 + LLM 超时降级 + 贵路径缓存。"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
        policy: Optional[IntentPolicy] = None,
        redis_url: Optional[str] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = create_async_client(**kwargs)
        self.model = model
        self.threshold = confidence_threshold
        self.policy = policy or load_intent_policy()
        self._embedding_enabled = True
        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._llm_cache = IntentLLMCache(
            ttl_seconds=self.policy.llm_cache_ttl_seconds,
            max_entries=self.policy.llm_cache_max_entries,
            redis_url=redis_url,
        )
        self._codec = IntentResultJSONCodec(IntentResult, IntentCategory, UrgencyLevel)
        # 兼容旧字段（仅统计；真正缓存已迁到 _llm_cache）
        self.cache_hits = 0
        self.cache_misses = 0

        merged = merge_into_templates(_TEMPLATES, load_corrections())
        if merged:
            logger.info("intent feedback: merged %s corrections into templates", merged)

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        t0 = time.monotonic()
        normalized = normalize_text(self._clean_text(message), self.policy.synonyms)
        if not normalized:
            return self._pack(
                IntentCategory.OTHER,
                0.0,
                normalized or message,
                {},
                "空消息",
                t0,
                {"fast_path": 0.0},
            )

        # 1) 配置化快路径
        pat = self._pattern_recognize(normalized)
        fast_enabled = os.getenv("ECHOMIND_FAST_INTENT", "true").strip().lower() not in {
            "0", "false", "no", "off",
        }
        fast_intent = pat.get("intent", IntentCategory.OTHER)
        fast_confidence = float(pat.get("confidence", 0.0) or 0.0)
        if (
            fast_enabled
            and fast_intent in _SPECIFIC_INTENTS
            and fast_confidence >= self.policy.fast_path_min_confidence
        ):
            return self._pack(
                fast_intent,
                fast_confidence,
                normalized,
                {
                    "llm": 0.0,
                    "embedding": 0.0,
                    "pattern": fast_confidence,
                    "fast_path": 1.0,
                },
                "明确业务关键词命中，使用本地快速路由",
                t0,
            )

        cache_key = self._cache_key(normalized, history)

        # 2) 仅贵路径缓存
        cached = await self._llm_cache.get(cache_key, self._codec)
        if cached is not None:
            self.cache_hits += 1
            cached.latency_ms = (time.monotonic() - t0) * 1000
            cached.source_scores = {
                **dict(cached.source_scores or {}),
                "llm_cache_hit": 1.0,
            }
            return cached
        self.cache_misses += 1

        # 3) 向量分类先行（可跳过 LLM）
        emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}
        if self._embedding_enabled:
            emb = await self._embedding_recognize(normalized)
            emb_conf = float(emb.get("confidence", 0.0) or 0.0)
            if (
                emb.get("intent") in _SPECIFIC_INTENTS
                and emb_conf >= self.policy.vector_skip_llm_confidence
            ):
                # Do not dilute with a fake LLM vote; vector path is authoritative here.
                pat_conf = float(pat.get("confidence", 0.0) or 0.0)
                return self._pack(
                    emb["intent"],
                    emb_conf,
                    normalized,
                    {
                        "llm": 0.0,
                        "embedding": emb_conf,
                        "pattern": pat_conf,
                        "vector_skip_llm": 1.0,
                    },
                    "向量高置信跳过 LLM",
                    t0,
                )

        # 4) LLM（超时降级）
        llm = await self._llm_recognize_with_timeout(normalized, history)
        intent, confidence, source_scores = self._vote(llm, emb, pat)
        if llm.get("timed_out"):
            source_scores["llm_timeout_degraded"] = 1.0
        if llm.get("failed"):
            source_scores["llm_failed"] = 1.0

        result = self._pack(
            intent,
            confidence,
            normalized,
            source_scores,
            llm.get("reasoning", ""),
            t0,
        )

        # 只缓存真正调用过 LLM 的结果（含超时降级后的融合结果，避免打爆上游）
        if not llm.get("skipped"):
            to_store = IntentResult(
                intent=result.intent,
                confidence=result.confidence,
                urgency=result.urgency,
                intent_group=result.intent_group,
                entities=result.entities,
                reasoning=result.reasoning,
                latency_ms=result.latency_ms,
                source_scores={**result.source_scores, "stored_llm_cache": 1.0},
            )
            await self._llm_cache.set(cache_key, to_store, self._codec)
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线纠错：更新模板 + 持久化回流样本。"""
        normalized = normalize_text(self._clean_text(message), self.policy.synonyms)
        tpls = _TEMPLATES.setdefault(correct, [])
        if normalized and normalized not in tpls:
            tpls.append(normalized)
            self._tpl_embeddings.pop(correct, None)
        try:
            append_correction(normalized or message, correct)
        except Exception as ex:
            logger.warning("persist intent correction failed: %s", ex)
        # 模板变更后清空贵路径缓存，避免脏读
        self._llm_cache.clear_memory()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._llm_cache.clear())
        except RuntimeError:
            pass
        logger.info("学习新样本 → %s: %s", correct.value, (normalized or message)[:40])

    async def aclose(self) -> None:
        await self._llm_cache.close()
        try:
            await self.client.close()
        except Exception:
            pass

    # ── 内部策略 ─────────────────────────────────────────────────────────────

    def _pack(
        self,
        intent: IntentCategory,
        confidence: float,
        message: str,
        source_scores: Dict[str, float],
        reasoning: str,
        t0: float,
    ) -> IntentResult:
        return IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=self._urgency(message, intent),
            intent_group=self._intent_group(intent),
            entities=self._extract_entities(message),
            reasoning=reasoning,
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
        )

    async def _llm_recognize_with_timeout(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self._llm_recognize(message, history),
                timeout=self.policy.llm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("LLM intent timed out after %.2ss", self.policy.llm_timeout_seconds)
            return {
                "intent": IntentCategory.OTHER,
                "confidence": 0.0,
                "reasoning": "LLM 超时降级",
                "failed": True,
                "timed_out": True,
            }

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]
        )
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是客服意图分析专家。根据示例判断用户意图，返回 JSON。
如果用户问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如退款优先返回 refund，发票优先返回 invoice，登录故障优先返回 technical_login。

示例:
{examples}
{ctx}
用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning("LLM 识别失败: %s", ex)
            return {
                "intent": IntentCategory.OTHER,
                "confidence": 0.0,
                "reasoning": "LLM 失败",
                "failed": True,
            }

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)
            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat
            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning("Embedding 识别失败: %s", ex)
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        msg = message.lower()
        specific = self.policy.specific_patterns or {}
        generic = self.policy.generic_patterns or {}

        best_cat, best_score = self._best_pattern_match(msg, specific)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}
        best_cat, best_score = self._best_pattern_match(msg, generic)
        return {"intent": best_cat, "confidence": best_score}

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed") or llm.get("timed_out"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best]
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        if best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS and pat_conf >= 0.5 and best_score < 0.8:
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(best_score, pat_conf), source_scores
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        message = self._clean_text(message)
        return {
            "order_id": self._unique(
                re.findall(
                    r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})",
                    message,
                    re.I,
                )
            ),
            "product": [],
            "date": self._unique(
                re.findall(
                    r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)",
                    message,
                )
            ),
            "amount": self._unique(
                re.findall(
                    r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))",
                    message,
                    re.I,
                )
            ),
            "error_code": self._unique(
                re.findall(
                    r"(?:error(?:_code)?|错误码|状态码|http)\s*[:：#]?\s*([45]\d{2})\b",
                    message,
                    re.I,
                )
                + re.findall(r"\b([45]\d{2})\b", message)
            ),
        }

    async def _load_template_embeddings(self) -> None:
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return
        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning("远端 Embedding 失败，使用本地向量兜底: %s", ex)
        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        import hashlib as _hashlib

        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i : i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)
        for token in tokens:
            digest = _hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        configured = self.policy.urgency_keywords or {}
        # CRITICAL → HIGH → MEDIUM order
        for level in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH, UrgencyLevel.MEDIUM):
            kws = configured.get(level) or []
            if any(kw.lower() in msg for kw in kws):
                return level
        if intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        from core.model_scope import selected_model

        payload = {
            "message": self._clean_text(message)[:200],
            "model": selected_model.get() or self.model,
        }
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", ""))[:160],
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw.lower() in message)
            if not hits:
                continue
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        stats = self._llm_cache.stats
        return {
            **stats,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses)
                else 0.0
            ),
        }

"""Enterprise routing policy: hard rules → intent map → evidence → collaborate.

Design notes (ops-friendly):
- INTENT_ROUTING is the single source of truth for intent → agent.
- Intent weight scales with confidence; low confidence leans on keywords/entities.
- Supporting agents require independent evidence (not just sharing the primary intent).
- At most one supporting specialist to control cost/latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from agents.contracts import AgentType, Request, RoutingDecision
from core.intent_recognizer import IntentCategory, UrgencyLevel


# ---------------------------------------------------------------------------
# Tunables (keep together for product / ops review)
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE = 0.75
MID_CONFIDENCE = 0.50
CLARIFY_CONFIDENCE = 0.50

INTENT_WEIGHT_HIGH = 0.80
INTENT_WEIGHT_MID = 0.50
INTENT_WEIGHT_LOW = 0.20

KEYWORD_HIT = 0.16
KEYWORD_CAP = 0.40
ENTITY_WEIGHTS: Mapping[str, Tuple[AgentType, float]] = {
    "error_code": (AgentType.TECHNICAL, 0.22),
    "amount": (AgentType.BILLING, 0.18),
    "order_id": (AgentType.GENERAL, 0.12),
}

# Supporting: need real cross-domain evidence, not a weak keyword splash.
# Compared against evidence only — primary total is inflated by intent weight.
SUPPORT_EVIDENCE_MIN = 0.28
MAX_SUPPORTING_AGENTS = 1

# When high-confidence intent maps to a specialist, another domain must beat
# this margin on evidence to steal primary.
PRIMARY_OVERRIDE_MARGIN = 0.15

PRIOR: Mapping[AgentType, float] = {
    AgentType.GENERAL: 0.05,
    AgentType.TECHNICAL: 0.0,
    AgentType.BILLING: 0.0,
}

INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
    IntentCategory.TECHNICAL: AgentType.TECHNICAL,
    IntentCategory.TECHNICAL_LOGIN: AgentType.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: AgentType.TECHNICAL,
    IntentCategory.BILLING: AgentType.BILLING,
    IntentCategory.REFUND: AgentType.BILLING,
    IntentCategory.INVOICE: AgentType.BILLING,
    IntentCategory.PAYMENT_ISSUE: AgentType.BILLING,
    IntentCategory.ACCOUNT: AgentType.BILLING,
    IntentCategory.ACCOUNT_SECURITY: AgentType.BILLING,
    IntentCategory.ESCALATION: AgentType.ESCALATION,
    IntentCategory.HUMAN_HANDOFF: AgentType.ESCALATION,
    IntentCategory.QUERY: AgentType.GENERAL,
    IntentCategory.ORDER_STATUS: AgentType.GENERAL,
    IntentCategory.LOGISTICS: AgentType.GENERAL,
    IntentCategory.REQUEST: AgentType.GENERAL,
    IntentCategory.COMPLAINT: AgentType.GENERAL,
    IntentCategory.GREETING: AgentType.GENERAL,
    IntentCategory.FEEDBACK: AgentType.GENERAL,
    IntentCategory.OTHER: AgentType.GENERAL,
}

DOMAIN_KEYWORDS: Mapping[AgentType, Sequence[str]] = {
    AgentType.TECHNICAL: (
        "崩溃",
        "报错",
        "error",
        "crash",
        "无法登录",
        "登录失败",
        "验证码",
        "500",
        "401",
        "403",
        "超时",
        "白屏",
    ),
    AgentType.BILLING: (
        "退款",
        "退货",
        "扣款",
        "发票",
        "账单",
        "支付",
        "订阅",
        "refund",
        "invoice",
        "多扣",
        "重复扣",
    ),
    AgentType.GENERAL: (
        "订单",
        "物流",
        "快递",
        "配送",
        "会员",
        "积分",
        "咨询",
        "帮助",
    ),
}

SPECIALISTS = (AgentType.TECHNICAL, AgentType.BILLING)


@dataclass(frozen=True)
class ScoreComponents:
    total: Dict[AgentType, float]
    intent: Dict[AgentType, float]
    evidence: Dict[AgentType, float]
    intent_weight: float
    mapped_agent: AgentType | None


def needs_clarification(req: Request) -> bool:
    if req.intent not in (None, IntentCategory.OTHER):
        return False
    text = (req.message or "").strip()
    if len(text) <= 2:
        return False
    return float(req.intent_confidence or 0.0) < CLARIFY_CONFIDENCE


def _intent_weight(confidence: float) -> float:
    if confidence >= HIGH_CONFIDENCE:
        return INTENT_WEIGHT_HIGH
    if confidence >= MID_CONFIDENCE:
        return INTENT_WEIGHT_MID
    return INTENT_WEIGHT_LOW


def _empty_buckets() -> Dict[AgentType, float]:
    return {
        AgentType.GENERAL: 0.0,
        AgentType.TECHNICAL: 0.0,
        AgentType.BILLING: 0.0,
    }


def _keyword_evidence(message: str) -> Dict[AgentType, float]:
    msg = (message or "").lower()
    scores = _empty_buckets()
    for agent_type, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in msg)
        if hits:
            scores[agent_type] = min(KEYWORD_CAP, hits * KEYWORD_HIT)
    return scores


def _entity_evidence(entities: Mapping[str, object] | None) -> Dict[AgentType, float]:
    scores = _empty_buckets()
    if not entities:
        return scores
    for key, (agent_type, weight) in ENTITY_WEIGHTS.items():
        values = entities.get(key)
        if values:
            scores[agent_type] += weight
    return scores


def _merge_scores(*parts: Mapping[AgentType, float]) -> Dict[AgentType, float]:
    merged = _empty_buckets()
    for part in parts:
        for agent_type, value in part.items():
            if agent_type in merged:
                merged[agent_type] += float(value)
    return merged


def score_components(req: Request) -> ScoreComponents:
    """Split intent contribution vs lexical/entity evidence for explainability."""
    confidence = float(req.intent_confidence or 0.0)
    weight = _intent_weight(confidence)
    mapped = INTENT_ROUTING.get(req.intent) if req.intent else None

    intent_scores = _empty_buckets()
    if mapped in intent_scores:
        # OTHER is a weak default — do not overweight uncertain catch-all.
        applied = weight * (0.5 if req.intent is IntentCategory.OTHER else 1.0)
        intent_scores[mapped] = applied
    elif mapped is AgentType.ESCALATION:
        # Escalation is handled as a hard rule in route_decision; keep scores neutral.
        pass

    evidence = _merge_scores(_keyword_evidence(req.message), _entity_evidence(req.entities))
    prior = {agent: float(PRIOR.get(agent, 0.0)) for agent in _empty_buckets()}
    total = _merge_scores(prior, intent_scores, evidence)
    return ScoreComponents(
        total={k: round(v, 3) for k, v in total.items()},
        intent={k: round(v, 3) for k, v in intent_scores.items()},
        evidence={k: round(v, 3) for k, v in evidence.items()},
        intent_weight=round(weight, 3),
        mapped_agent=mapped if mapped in (AgentType.GENERAL, *SPECIALISTS) else None,
    )


def domain_scores(req: Request) -> Dict[AgentType, float]:
    return score_components(req).total


def collaboration_targets(req: Request) -> List[AgentType]:
    """Specialists with independent evidence (used by supporting selection)."""
    components = score_components(req)
    targets = [
        agent
        for agent in SPECIALISTS
        if components.evidence.get(agent, 0.0) >= SUPPORT_EVIDENCE_MIN
    ]
    return list(dict.fromkeys(targets))


def _pick_primary(components: ScoreComponents) -> Tuple[AgentType, float]:
    ordered = sorted(components.total.items(), key=lambda item: item[1], reverse=True)
    top_agent, top_score = ordered[0]

    mapped = components.mapped_agent
    if mapped is None:
        return top_agent, top_score

    mapped_score = components.total.get(mapped, 0.0)
    # High-confidence mapped specialist: keep unless another domain clearly dominates.
    if (
        components.intent_weight >= INTENT_WEIGHT_HIGH
        and mapped in SPECIALISTS
    ):
        rival_evidence = max(
            (components.evidence.get(agent, 0.0) for agent in SPECIALISTS if agent != mapped),
            default=0.0,
        )
        intent_part = components.intent.get(mapped, 0.0)
        if rival_evidence > intent_part + PRIMARY_OVERRIDE_MARGIN:
            rival = max(
                ((a, components.total[a]) for a in SPECIALISTS if a != mapped),
                key=lambda item: item[1],
            )
            return rival
        return mapped, mapped_score

    return top_agent, top_score


def _pick_supporting(
    components: ScoreComponents,
    primary: AgentType,
) -> List[AgentType]:
    candidates: List[Tuple[AgentType, float]] = []
    for agent in SPECIALISTS:
        if agent == primary:
            continue
        evidence = components.evidence.get(agent, 0.0)
        if evidence < SUPPORT_EVIDENCE_MIN:
            continue
        candidates.append((agent, evidence))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [agent for agent, _ in candidates[:MAX_SUPPORTING_AGENTS]]


def routing_reason(
    req: Request,
    scores: Dict[AgentType, float],
    primary_agent: AgentType,
    supporting_agents: List[AgentType],
    components: ScoreComponents | None = None,
) -> str:
    score_text = ", ".join(
        f"{agent_type.value}={score:.2f}"
        for agent_type, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    )
    support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
    intent = req.intent.value if req.intent else "unknown"
    extra = ""
    if components is not None:
        evidence_text = ", ".join(
            f"{agent.value}={components.evidence.get(agent, 0.0):.2f}"
            for agent in (AgentType.TECHNICAL, AgentType.BILLING, AgentType.GENERAL)
        )
        extra = (
            f", intent_weight={components.intent_weight:.2f}, "
            f"mapped={components.mapped_agent.value if components.mapped_agent else 'none'}, "
            f"evidence=[{evidence_text}]"
        )
    return (
        f"intent={intent}, group={req.intent_group or 'unknown'}, "
        f"primary={primary_agent.value}, supporting={support_text}, "
        f"scores=[{score_text}]{extra}"
    )


def route_decision(req: Request) -> RoutingDecision:
    if req.urgency == UrgencyLevel.CRITICAL:
        return RoutingDecision(
            primary_agent=AgentType.ESCALATION,
            reason="紧急度为 CRITICAL，触发升级路由",
            confidence=1.0,
        )

    if req.intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
        return RoutingDecision(
            primary_agent=AgentType.ESCALATION,
            reason=f"意图为 {req.intent.value if req.intent else 'unknown'}，触发升级路由",
            confidence=max(float(req.intent_confidence or 0.0), 0.8),
        )

    components = score_components(req)
    primary_agent, primary_score = _pick_primary(components)
    supporting_agents = _pick_supporting(components, primary_agent)

    # Confidence blends model intent confidence with routing margin over runner-up.
    ordered = sorted(components.total.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
    confidence = min(
        1.0,
        0.55 * float(req.intent_confidence or 0.0) + 0.45 * min(primary_score, 1.0) + 0.1 * margin,
    )

    return RoutingDecision(
        primary_agent=primary_agent,
        supporting_agents=supporting_agents,
        reason=routing_reason(
            req, components.total, primary_agent, supporting_agents, components
        ),
        confidence=round(confidence, 3),
    )

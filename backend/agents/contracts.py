"""Shared agent contracts used by LangGraph runtime and tools."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.intent_recognizer import IntentCategory, UrgencyLevel


class AgentType(Enum):
    GENERAL = "general"
    TECHNICAL = "technical"
    BILLING = "billing"
    ESCALATION = "escalation"


@dataclass(frozen=True)
class AgentProfile:
    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024
    system_prompt: str = ""


@dataclass
class AgentStats:
    total: int = 0
    success: int = 0
    total_ms: float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type: AgentType
    content: str
    success: bool
    confidence: float = 1.0
    latency_ms: float = 0.0
    escalate: bool = False
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class Request:
    message: str
    user_id: str
    conv_id: str
    context: str = ""
    history: Optional[List[Dict[str, str]]] = None
    entities: Dict[str, List[str]] = field(default_factory=dict)
    intent: Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency: Optional[UrgencyLevel] = None
    intent_confidence: float = 1.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    thread_id: Optional[str] = None


@dataclass
class OrchestratorResult:
    request_id: str
    response: str
    agent_type: AgentType
    intent: Optional[IntentCategory]
    escalated: bool = False
    latency_ms: float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    thread_id: str = ""
    checkpoint_id: Optional[str] = None
    interrupted: bool = False
    interrupt_payload: Optional[Dict[str, Any]] = None


@dataclass
class RoutingDecision:
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


GENERAL_PROFILE = AgentProfile(
    role="通用客服分诊与首轮接待",
    mission="快速回答基础问题，澄清不完整需求，并识别是否需要专业 Agent 或人工处理。",
    workflow=("复述诉求", "判断业务范围", "直接回答或补充必要信息", "给出下一步"),
    input_contract=("对话历史", "用户画像", "意图与紧急度", "知识库上下文"),
    output_contract=("先回应核心问题", "信息不足时只询问必要字段", "明确下一步和边界"),
    handoff_conditions=("涉及权限、资金、隐私或复杂投诉", "用户明确要求人工"),
    tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields"),
    temperature=0.3,
    max_tokens=900,
    system_prompt=(
        "你是 EpochFlow 智能客服。友好、简洁地回答用户问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业客服。"
    ),
)

TECHNICAL_PROFILE = AgentProfile(
    role="技术故障诊断与排障",
    mission="基于错误码、环境和复现信息缩小根因范围，给出低风险、可验证的排查步骤。",
    workflow=("确认现象", "判断影响范围", "按网络/权限/配置/依赖排查", "给出验证方式", "判断升级条件"),
    input_contract=("错误码", "问题发生时间", "运行环境", "影响范围", "最近变更", "知识库上下文"),
    output_contract=("现象复述", "可能原因", "编号排查步骤", "验证结果", "需要补充的信息"),
    handoff_conditions=("生产大面积不可用", "数据丢失或权限异常", "需要后台日志、数据库或人工操作"),
    tool_scope=("search_knowledge_base", "lookup_error_code", "build_diagnostic_plan"),
    temperature=0.1,
    max_tokens=1200,
    system_prompt=(
        "你是技术支持专家。专注于：故障排查、错误诊断、系统配置。"
        "提供清晰的步骤化解决方案。遇到需要后台操作的问题，说明需要升级处理。"
    ),
)

BILLING_PROFILE = AgentProfile(
    role="账单核验与售后处理",
    mission="区分扣款、退款、发票、订阅等资金场景，解释可判断事实，并明确核验和人工审核边界。",
    workflow=("确认账单场景", "收集必要核验字段", "区分订单/实付/退款金额", "说明处理路径与时效", "判断是否升级"),
    input_contract=("订单号", "金额与币种", "支付时间", "支付渠道", "用户期望", "知识库上下文"),
    output_contract=("需要核验的信息", "当前可判断内容", "下一步处理路径", "时效边界"),
    handoff_conditions=("实际退款或补偿", "重复扣款或支付成功但订单未生效", "发票作废/重开", "企业合同或大额订单"),
    tool_scope=("search_knowledge_base", "check_billing_fields", "compare_amounts"),
    temperature=0.0,
    max_tokens=1100,
    system_prompt=(
        "你是账单服务专家。专注于：账单查询、退款申请、发票问题、订阅管理。"
        "对财务问题保持准确和专业。涉及实际退款操作时，说明需要人工审核。"
    ),
)

ESCALATION_PROFILE = AgentProfile(
    role="人工升级与交接",
    mission="确认升级原因，整理已知上下文，告知用户下一步，不执行未经授权的业务操作。",
    workflow=("确认升级原因", "整理已知信息", "标记优先级", "生成交接摘要"),
    input_contract=("用户消息", "意图", "紧急度", "结构化实体", "对话背景"),
    output_contract=("升级原因", "已知信息摘要", "还需补充的信息", "保守的后续说明"),
    handoff_conditions=("用户明确要求人工", "紧急或高风险场景"),
    tool_scope=("search_knowledge_base", "create_handoff_summary"),
    temperature=0.0,
    max_tokens=500,
    system_prompt="你负责客服人工升级交接，不要继续模拟已完成的后台操作。",
)

PROFILES: Dict[AgentType, AgentProfile] = {
    AgentType.GENERAL: GENERAL_PROFILE,
    AgentType.TECHNICAL: TECHNICAL_PROFILE,
    AgentType.BILLING: BILLING_PROFILE,
    AgentType.ESCALATION: ESCALATION_PROFILE,
}

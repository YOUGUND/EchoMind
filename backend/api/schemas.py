from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    model: Optional[str] = Field(default=None, max_length=200)
    message:     str = Field(min_length=1, max_length=8000)
    user_id:     str = Field(default="anonymous", max_length=128)
    conv_id:     Optional[str] = Field(default=None, max_length=128)
    thread_id:   Optional[str] = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    model: str = ""
    conv_id:     str
    request_id:  str = ""
    response:    str
    intent:      str
    intent_group: str = "other"
    agent_type:  str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    thread_id: str = ""
    checkpoint_id: Optional[str] = None
    interrupted: bool = False
    interrupt_payload: Dict[str, Any] = Field(default_factory=dict)


class ResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    resume: Any = Field(description="澄清文本，或 escalate approve/reject")
    request_id: Optional[str] = Field(default=None, max_length=64)


class ForkRequest(BaseModel):
    source_thread_id: str = Field(min_length=1, max_length=128)
    checkpoint_id: str = Field(min_length=1, max_length=128)
    new_thread_id: Optional[str] = Field(default=None, max_length=128)


class ToolTraceResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)



class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=100000)


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput] = Field(min_length=1, max_length=100)


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str = Field(min_length=1, max_length=8000)
    expected_intent: str = Field(min_length=1, max_length=100)
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = Field(default=None, max_length=8000)
    turns: Optional[List[str]] = Field(default=None, max_length=3)
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。

    - 不传 suite_id：工作台实验模式（可内联少量 case，默认限流）
    - 传 suite_id：中台套件模式（从 data/eval/suites 加载）
    """
    suite_id: Optional[str] = Field(default=None, max_length=100)
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None
    fail_under: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    save_run: bool = True
    update_baseline: bool = False
    collect_regressions: bool = True



from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class RelatedPage(BaseModel):
    """A navigable page suggestion returned with assistant responses."""
    label: str
    route: str


class ContextInfo(BaseModel):
    leave: Optional[Dict[str, Any]] = Field(default_factory=dict)
    requests: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tasks: Optional[Dict[str, Any]] = Field(default_factory=dict)
    routes: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    role: str
    question: str
    pageContext: Optional[str] = None
    context: Optional[ContextInfo] = Field(default_factory=ContextInfo)


class ChatResponse(BaseModel):
    answer: str
    reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    relatedPages: List[RelatedPage] = Field(default_factory=list)
    disclaimer: str = "The assistant provides guidance only. Final decisions remain with authorized users."
    aiGenerated: bool = True
    # V2 traceability: identifies which pipeline step produced this response.
    # Values: "local_rules" | "external_ai" | "fallback" | "refusal"
    source: Optional[str] = None
    # V2 Phase 3: drafting assistant — populated only when a draft was generated.
    # None for all non-drafting responses (backward-compatible).
    draft: Optional[str] = None

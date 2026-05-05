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
    # V2 Phase 3.1: structured draft extraction.
    # draftType  — identifies the request type being drafted.
    #              Values: "LEAVE_REQUEST" | "LOAN_REQUEST" | "AUTHORIZATION_REQUEST"
    #                      | "DOCUMENT_REQUEST" | "IMPROVE_TEXT"
    #              None for all non-drafting responses (backward-compatible).
    # draftFields — extracted field values keyed by field name.
    #              Stable shape per draftType; null values for unextracted fields.
    #              None only for IMPROVE_TEXT (no structured fields apply).
    # missingFields — list of field names that could not be extracted from the
    #              user's input. Empty list for non-drafting and IMPROVE_TEXT.
    #              Spring Boot is responsible for final validation.
    draftType: Optional[str] = None
    draftFields: Optional[dict] = None
    missingFields: List[str] = Field(default_factory=list)

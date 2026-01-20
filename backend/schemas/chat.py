"""Pydantic models for the /api/v1/chat endpoint."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    search_method: Optional[Literal["vector", "keyword", "hybrid"]] = "hybrid"
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    language: Optional[Literal["en", "fr"]] = "en"
    answer_style: Optional[Literal["concise", "detailed", "formal"]] = "concise"
    include_citations: Optional[bool] = True


class SourceReference(BaseModel):
    source_type: Literal["pdf", "html", "unknown"] = "unknown"
    source_url: Optional[str] = None
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    score: Optional[float] = None
    similarity_score: Optional[float] = None
    content_preview: Optional[str] = None
    extra: Optional[dict[str, Any]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    retrieved_count: int = 0
    latency: float = 0.0



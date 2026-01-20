"""Optional feedback endpoint.

This is intentionally simple for a portfolio project:
- accept feedback payload
- log it (in production you might store to DB / analytics)
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    answer: str = Field(..., min_length=1, max_length=20000)
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=2000)


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> dict[str, str]:
    logger.info(
        "Feedback received (rating=%s, comment=%s)",
        req.rating,
        (req.comment[:120] + "…") if req.comment and len(req.comment) > 120 else (req.comment or ""),
    )
    return {"status": "ok"}


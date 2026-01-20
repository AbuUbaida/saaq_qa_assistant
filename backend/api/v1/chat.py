"""Chat endpoint (search-style RAG).

Frontend calls:
  POST /api/v1/chat
with a JSON body matching `ChatRequest`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.config import get_settings
from ...core.logging import get_logger
from ...schemas.chat import ChatRequest, ChatResponse
from ...services.rag_pipeline import answer_question

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    settings = get_settings()

    try:
        result = answer_question(
            question=req.question,
            search_method=req.search_method or "hybrid",
            top_k=req.top_k or settings.default_top_k,
            language=req.language or "en",
            answer_style=req.answer_style or "concise",
            include_citations=bool(req.include_citations) if req.include_citations is not None else True,
        )
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e

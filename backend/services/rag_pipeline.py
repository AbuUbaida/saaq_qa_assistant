"""RAG pipeline: retrieve -> build prompt -> LLM."""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.documents import Document

from ..core.config import get_settings
from ..core.logging import get_logger
from ..db.vector_store import WeaviateStore
from .document_retriever import retrieve_documents
from .llm_service import create_llm, generate_response
from .rag_prompt_builder import build_prompt

logger = get_logger(__name__)


def _preview(text: str, max_chars: int = 240) -> str:
    if not text:
        return ""
    t = " ".join(text.split())
    return (t[: max_chars - 1] + "…") if len(t) > max_chars else t


def answer_question(
    *,
    question: str,
    search_method: Literal["vector", "keyword", "hybrid"] = "hybrid",
    top_k: int = 5,
    language: str = "en",
    answer_style: str = "concise",
    include_citations: bool = True,
) -> dict[str, Any]:
    """Answer a question using the RAG pipeline."""
    settings = get_settings()
    start = time.perf_counter()

    store = WeaviateStore(
        use_local=settings.weaviate_use_local,
        url=settings.weaviate_cloud_url,
        api_key=settings.weaviate_api_key,
        collection_name=settings.weaviate_collection,
    )
    try:
        docs: list[Document] = retrieve_documents(
            weaviate_client=store.client,
            query=question,
            search_method=search_method,
            top_k=top_k,
            collection_name=settings.weaviate_collection,
            embedding_model=settings.embedding_model,
            api_key=settings.hf_api_key,
        )

        if not docs:
            answer = (
                "I couldn’t find anything relevant in the knowledge base for that question. "
                "Try rephrasing, or ingest more SAAQ documents."
            )
            latency = time.perf_counter() - start
            return {"answer": answer, "sources": [], "retrieved_count": 0, "latency": latency}

        prompt = build_prompt(
            question=question,
            context_documents=docs,
            language=language,
            answer_style=answer_style,
            include_citations=include_citations,
        )

        llm = create_llm(
            provider=settings.llm_provider,
            model_name=settings.hf_llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.hf_api_key,
        )
        answer = generate_response(llm, prompt)

        sources = []
        for doc in docs:
            md = doc.metadata or {}
            sources.append(
                {
                    "source_type": md.get("source_type", "unknown") or "unknown",
                    "source_url": md.get("source_url"),
                    "source_file": md.get("source_file") or md.get("source_path"),
                    "page_number": md.get("page_number"),
                    "score": md.get("score"),
                    "similarity_score": md.get("similarity_score"),
                    "content_preview": _preview(doc.page_content),
                    "extra": {
                        k: v
                        for k, v in md.items()
                        if k not in {"source_url", "source_file", "source_path"}
                    },
                }
            )

        latency = time.perf_counter() - start
        logger.info("Answered question (docs=%d, latency=%.2fs)", len(docs), latency)
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(docs),
            "latency": latency,
        }
    finally:
        store.close()

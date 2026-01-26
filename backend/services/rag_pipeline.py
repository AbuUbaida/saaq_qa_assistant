"""RAG pipeline: retrieve -> build prompt -> LLM."""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.documents import Document

from opentelemetry.trace import Status, StatusCode
from opentelemetry import trace

from ..core.config import get_settings
from ..core.logging import get_logger
from ..db.vector_store import WeaviateStore
from .document_retriever import retrieve_documents
from .llm_service import create_llm, generate_response
from .rag_prompt_builder import build_prompt

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


def _preview(text: str, max_chars: int = 240) -> str:
    if not text:
        return ""
    t = " ".join(text.split())
    return (t[: max_chars - 1] + "…") if len(t) > max_chars else t


def _prompt_to_text(prompt: list[Any], max_chars: int = 4000) -> tuple[str, int]:
    parts: list[str] = []
    for message in prompt:
        content = getattr(message, "content", None)
        parts.append(str(content) if content is not None else str(message))
    
    text = "\n\n".join(parts)
    truncated = len(text) > max_chars
    text_display = text[:max_chars] + "…" if truncated else text

    return text_display, len(text)


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
        with tracer.start_as_current_span("rag.pipeline") as pipeline_span:
            pipeline_span.add_event("start_pipeline")
            pipeline_span.set_attribute("input", question)

            with tracer.start_as_current_span("rag.retrieve") as span:
                span.add_event("start_retrieval")
                span.set_attribute("input.query", question)
                span.set_attribute("input.search_method", search_method)
                span.set_attribute("input.top_k", top_k)
                span.set_attribute("input.embedding_model", settings.embedding_model)
                retrieve_start = time.perf_counter()

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
                    span.set_attribute("retrieved_count", 0)
                    span.set_attribute("latency_ms", int((time.perf_counter() - retrieve_start) * 1000))
                    span.set_status(Status(StatusCode.OK))
                    span.add_event("end_retrieval")

                    pipeline_span.set_attribute("latency_ms", int(latency * 1000))
                    pipeline_span.set_status(Status(StatusCode.OK))
                    pipeline_span.add_event("end_pipeline")
                    return {"answer": answer, "sources": [], "retrieved_count": 0, "latency": latency}

                sources = []
                for i, doc in enumerate(docs):
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

                    span.set_attribute(f"retrieval.documents.{i}.document.id", i)
                    span.set_attribute(f"retrieval.documents.{i}.document.content", doc.page_content)
                    span.set_attribute(f"retrieval.documents.{i}.document.metadata", md)

                span.set_attribute("retrieved_count", len(docs))
                span.set_attribute("latency_ms", int((time.perf_counter() - retrieve_start) * 1000))
                span.set_status(Status(StatusCode.OK))
                span.add_event("end_retrieval")

            with tracer.start_as_current_span("rag.prompt") as span:
                span.add_event("start_prompt_build")
                prompt_start = time.perf_counter()

                prompt = build_prompt(
                    question=question,
                    context_documents=docs,
                    language=language,
                    answer_style=answer_style,
                    include_citations=include_citations,
                )
                
                prompt_text, prompt_length = _prompt_to_text(prompt, max_chars=1000)
                span.set_attribute("output.prompt", prompt_text)
                span.set_attribute("output.prompt_length", prompt_length)
                span.set_attribute("latency_ms", int((time.perf_counter() - prompt_start) * 1000))
                span.set_status(Status(StatusCode.OK))
                span.add_event("end_prompt_build")

            with tracer.start_as_current_span("rag.generate") as span:
                span.add_event("start_response_generation")
                span.set_attribute("input.llm_model", settings.hf_llm_model)
                llm_start = time.perf_counter()

                llm = create_llm(
                    provider=settings.llm_provider,
                    model_name=settings.hf_llm_model,
                    temperature=settings.llm_temperature,
                    api_key=settings.hf_api_key,
                )
                answer = generate_response(llm, prompt)

                span.set_attribute("latency_ms", int((time.perf_counter() - llm_start) * 1000))
                span.set_attribute("output.answer", answer)
                span.set_status(Status(StatusCode.OK))
                span.add_event("end_response_generation")

            latency = time.perf_counter() - start
            pipeline_span.set_attribute("output", answer)
            pipeline_span.set_attribute("latency_ms", int(latency * 1000))
            pipeline_span.set_status(Status(StatusCode.OK))
            pipeline_span.add_event("end_pipeline")

            logger.info("Answered question (docs=%d, latency=%.2fs)", len(docs), latency)
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_count": len(docs),
                "latency": latency,
            }
    finally:
        store.close()

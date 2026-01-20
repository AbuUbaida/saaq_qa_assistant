"""Retrieve relevant documents from Weaviate (vector / keyword / hybrid)."""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

import weaviate
from langchain_core.documents import Document

from ..core.logging import get_logger
from ..db.vector_store import DEFAULT_COLLECTION_NAME, WeaviateStore
from ..ingest.embedder import DEFAULT_EMBEDDING_MODEL, embed_query

logger = get_logger(__name__)


def _parse_metadata_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _result_to_document(result: Any) -> Document:
    props = getattr(result, "properties", {}) or {}

    md: dict[str, Any] = {}
    md.update(_parse_metadata_json(props.get("metadata_json")))

    md.update(
        {
            "source_url": props.get("source_url") or "",
            "source_file": props.get("source_file") or "",
            "source_path": props.get("source_path") or "",
            "source_type": props.get("source_type") or "",
            "embedding_model": props.get("embedding_model") or "",
            "page_number": props.get("page_number") or 0,
            "chunk_index": props.get("chunk_index") or 0,
            "document_index": props.get("document_index") or 0,
            "start_index": props.get("start_index") or 0,
        }
    )

    meta = getattr(result, "metadata", None)
    if meta is not None:
        if hasattr(meta, "distance") and meta.distance is not None:
            md["similarity_score"] = 1.0 - float(meta.distance)
        if hasattr(meta, "score") and meta.score is not None:
            md["score"] = float(meta.score)

    return Document(page_content=props.get("text", "") or "", metadata=md)


def retrieve_documents(
    *,
    weaviate_client: weaviate.WeaviateClient,
    query: str,
    search_method: Literal["vector", "keyword", "hybrid"] = "hybrid",
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
) -> list[Document]:
    if search_method not in ("vector", "keyword", "hybrid"):
        raise ValueError("search_method must be: 'vector', 'keyword', or 'hybrid'")

    store = WeaviateStore(client=weaviate_client, collection_name=collection_name)

    query_vector = None
    if search_method in ("vector", "hybrid"):
        query_vector = embed_query(query, model_name=embedding_model, api_key=api_key)

    if search_method == "vector":
        results = store.search_by_vector(query_vector=query_vector, collection_name=collection_name, top_k=top_k)
    elif search_method == "keyword":
        results = store.search_by_keyword(query_keyword=query, collection_name=collection_name, top_k=top_k)
    else:
        results = store.hybrid_search(
            query_keyword=query,
            query_vector=query_vector,
            collection_name=collection_name,
            top_k=top_k,
        )

    docs = [_result_to_document(r) for r in (results or [])]
    logger.info("Retrieved %d document(s)", len(docs))
    return docs

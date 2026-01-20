"""Index embeddings JSONL into Weaviate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.config import get_settings
from ..core.logging import get_logger
from ..db.vector_store import WeaviateStore
from .embedder import load_embeddings_from_jsonl

logger = get_logger(__name__)


def index_embeddings_jsonl(
    jsonl_path: Path | str,
    *,
    collection_name: Optional[str] = None,
    use_local: Optional[bool] = None,
    weaviate_url: Optional[str] = None,
    weaviate_api_key: Optional[str] = None,
) -> int:
    """Load embeddings from JSONL and upsert into Weaviate."""
    settings = get_settings()
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings JSONL not found: {path}")

    documents, embeddings = load_embeddings_from_jsonl(path)

    store = WeaviateStore(
        use_local=settings.weaviate_use_local if use_local is None else use_local,
        url=weaviate_url or settings.weaviate_cloud_url,
        api_key=weaviate_api_key or settings.weaviate_api_key,
        collection_name=collection_name or settings.weaviate_collection,
    )
    try:
        inserted = store.upsert(
            documents=documents,
            embeddings=embeddings,
            collection_name=collection_name or settings.weaviate_collection,
            embedding_model=(documents[0].metadata or {}).get("embedding_model"),
        )
        logger.info("Indexed %d embedding(s) from %s", inserted, path)
        return inserted
    finally:
        store.close()

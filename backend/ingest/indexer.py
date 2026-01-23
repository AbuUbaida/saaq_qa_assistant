"""Index embeddings JSONL into Weaviate."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..core.config import get_settings
from ..core.logging import get_logger
from ..db.vector_store import WeaviateStore
from .embedder import load_embeddings_from_jsonl

logger = get_logger(__name__)


def _load_all_embeddings(jsonl_paths: Iterable[Path]) -> tuple[list, list, Optional[str]]:
    all_docs = []
    all_embs = []
    embedding_model: Optional[str] = None
    for path in jsonl_paths:
        documents, embeddings = load_embeddings_from_jsonl(path)
        if documents and embedding_model is None:
            embedding_model = (documents[0].metadata or {}).get("embedding_model")
        all_docs.extend(documents)
        all_embs.extend(embeddings)
    return all_docs, all_embs, embedding_model


def index_embeddings_files(
    jsonl_paths: Iterable[Path | str],
    *,
    collection_name: Optional[str] = None,
    use_local: Optional[bool] = None,
    weaviate_url: Optional[str] = None,
    weaviate_api_key: Optional[str] = None,
    batch_size: int = 200,
) -> int:
    """Load embeddings from multiple JSONL files and batch-index into Weaviate."""
    settings = get_settings()
    paths = [Path(p) for p in jsonl_paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Embeddings JSONL not found: {path}")

    documents, embeddings, embedding_model = _load_all_embeddings(paths)
    if not documents:
        logger.info("No embeddings found to index.")
        return 0

    store = WeaviateStore(
        use_local=settings.weaviate_use_local if use_local is None else use_local,
        url=weaviate_url or settings.weaviate_cloud_url,
        api_key=weaviate_api_key or settings.weaviate_api_key,
        collection_name=collection_name or settings.weaviate_collection,
    )
    try:
        inserted = store.batch_insert(
            documents=documents,
            embeddings=embeddings,
            collection_name=collection_name or settings.weaviate_collection,
            embedding_model=embedding_model,
            batch_size=batch_size,
        )
        logger.info("Indexed %d embedding(s) from %d file(s)", inserted, len(paths))
        return inserted
    finally:
        store.close()


def index_embeddings_jsonl(
    jsonl_path: Path | str,
    *,
    collection_name: Optional[str] = None,
    use_local: Optional[bool] = None,
    weaviate_url: Optional[str] = None,
    weaviate_api_key: Optional[str] = None,
    batch_size: int = 200,
) -> int:
    """Load embeddings from a JSONL file and batch-index into Weaviate."""
    return index_embeddings_files(
        [jsonl_path],
        collection_name=collection_name,
        use_local=use_local,
        weaviate_url=weaviate_url,
        weaviate_api_key=weaviate_api_key,
        batch_size=batch_size,
    )

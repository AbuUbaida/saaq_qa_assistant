"""Weaviate vector store wrapper.

Goal: keep Weaviate interactions in one place with a small surface area:
- connect (local or cloud)
- ensure collection schema exists
- batch insert documents
- search (vector / keyword / hybrid)
- delete collection
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import weaviate
from dotenv import load_dotenv
from langchain_core.documents import Document
from weaviate.classes.query import MetadataQuery 
from weaviate.classes.config import Configure, DataType, Property
from weaviate.util import generate_uuid5

# try:
#     # Weaviate v4 style (preferred)
#     from weaviate.classes.query import MetadataQuery  # type: ignore
# except Exception:  # pragma: no cover
#     MetadataQuery = None  # type: ignore

logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_COLLECTION_NAME = "SAAQDocuments"


def _iso_to_datetime(value: Any) -> datetime:
    """Best-effort conversion for ISO timestamps in metadata."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            # Accept both `...Z` and `...+00:00` styles.
            s = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def make_chunk_id(doc: Document) -> str:
    """Create a deterministic chunk id from doc metadata + content.

    This ID is used to create a deterministic UUID so re-running ingestion
    won't create duplicates in Weaviate.
    """
    md = doc.metadata or {}

    source_type = str(md.get("source_type", "") or "")
    source_url = str(md.get("source_url", "") or "")
    source_file = str(md.get("source_file", "") or "")
    source_path = str(md.get("source_path", "") or "")
    page_number = str(md.get("page_number", "") or "")

    # Stable chunk addressing
    chunk_index = str(md.get("chunk_index", "") or "")
    start_index = str(md.get("start_index", "") or "")

    source_key = source_url or source_path or source_file
    text_hash = _content_hash(doc.page_content or "")

    return "|".join(
        [
            source_type,
            source_key,
            page_number,
            chunk_index,
            start_index,
            text_hash,
        ]
    )


class WeaviateStore:
    """Small wrapper around a Weaviate collection."""

    def __init__(
        self,
        client: Optional[weaviate.WeaviateClient] = None,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        use_local: bool = True,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name
        self._client = client
        self._client_owner = False

        if self._client is None:
            self._client = self._connect(use_local=use_local, url=url, api_key=api_key)
            self._client_owner = True

    @staticmethod
    def _connect(*, use_local: bool, url: Optional[str], api_key: Optional[str]) -> weaviate.WeaviateClient:
        if use_local:
            host = os.getenv("WEAVIATE_HOST", "localhost")
            port_raw = os.getenv("WEAVIATE_PORT", "8080")
            try:
                port = int(port_raw)
            except Exception:
                port = 8080

            try:
                client = weaviate.connect_to_local(host=host, port=port)  # type: ignore[call-arg]
            except TypeError:
                # Older client versions don't accept host/port kwargs.
                client = weaviate.connect_to_local()
        else:
            url = url or os.getenv("WEAVIATE_CLOUD_URL")
            api_key = api_key or os.getenv("WEAVIATE_API_KEY")
            if not url:
                raise ValueError("Missing Weaviate URL (set WEAVIATE_CLOUD_URL or pass url=...)")
            if not api_key:
                raise ValueError("Missing Weaviate API key (set WEAVIATE_API_KEY or pass api_key=...)")
            client = weaviate.connect_to_weaviate_cloud(cluster_url=url, auth_credentials=api_key)

        if not client.is_ready():
            raise RuntimeError("Weaviate client is not ready")
        return client

    @property
    def client(self) -> weaviate.WeaviateClient:
        if self._client is None:
            raise RuntimeError("Weaviate client not initialized")
        return self._client

    def ensure_collection(self, *, collection_name: Optional[str] = None) -> None:
        """Create the collection if it doesn't exist."""
        name = collection_name or self.collection_name
        if self.client.collections.exists(name):
            return

        properties: List[Property] = [
            # Main content
            Property(name="text", data_type=DataType.TEXT),

            # Source metadata (flat, easy to query)
            Property(name="source_type", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="source_url", data_type=DataType.TEXT),
            Property(name="source_file", data_type=DataType.TEXT),
            Property(name="source_path", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="start_index", data_type=DataType.INT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="total_chunks", data_type=DataType.INT),
            Property(name="document_index", data_type=DataType.INT),
            Property(name="language", data_type=DataType.TEXT),
            Property(name="collected_at", data_type=DataType.DATE),
            Property(name="cleaned_at", data_type=DataType.DATE),
            Property(name="embedding_model", data_type=DataType.TEXT),
            Property(name="content_hash", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            # Full metadata as JSON (keeps ingestion flexible without changing schema every time)
            Property(name="metadata_json", data_type=DataType.TEXT),
        ]

        self.client.collections.create(
            name,
            properties=properties,
            vector_config=Configure.Vectors.self_provided(),
        )
        logger.info("Created Weaviate collection: %s", name)

    @staticmethod
    def _to_object(doc: Document, *, embedding_model: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        md = doc.metadata or {}
        chunk_id = str(md.get("chunk_id") or make_chunk_id(doc))

        def _as_int(v: Any) -> Optional[int]:
            if v is None:
                return None
            if v == "":
                return None
            try:
                return int(v)
            except Exception:
                return None

        props: Dict[str, Any] = {
            "text": doc.page_content,
            "source_type": md.get("source_type", ""),
            "source": md.get("source_url") or md.get("source_path") or md.get("source_file") or "",
            "source_url": md.get("source_url", ""),
            "source_file": md.get("source_file", ""),
            "source_path": md.get("source_path", ""),
            "page_number": _as_int(md.get("page_number")),
            "start_index": _as_int(md.get("start_index")),
            "chunk_index": _as_int(md.get("chunk_index")),
            "total_chunks": _as_int(md.get("total_chunks")),
            "document_index": _as_int(md.get("document_index")),
            "language": md.get("language", ""),
            "collected_at": _iso_to_datetime(md.get("collected_at")),
            "cleaned_at": _iso_to_datetime(md.get("cleaned_at")),
            "embedding_model": embedding_model or md.get("embedding_model", ""),
            "content_hash": md.get("content_hash") or _content_hash(doc.page_content or ""),
            "chunk_id": chunk_id,
            "metadata_json": json.dumps(md, ensure_ascii=False, sort_keys=True),
        }

        # UUID should be deterministic for idempotent ingestion.
        uuid = str(generate_uuid5(chunk_id))
        return props, uuid

    def batch_insert(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        *,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        batch_size: int = 200,
    ) -> int:
        """Batch-insert documents and their vectors into Weaviate."""
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")

        name = collection_name or self.collection_name
        self.ensure_collection(collection_name=name)
        collection = self.client.collections.get(name)

        inserted = 0

        # Batch insert using Weaviate's batch API.
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for doc, vec in zip(documents, embeddings):
                props, uuid = self._to_object(doc, embedding_model=embedding_model)
                batch.add_object(properties=props, uuid=uuid, vector=vec)
                inserted += 1

        logger.info("Batch-inserted %d object(s) into %s", inserted, name)
        return inserted

    def _collection(self, collection_name: Optional[str] = None):
        name = collection_name or self.collection_name
        return self.client.collections.get(name)

    def search_by_vector(
        self,
        *,
        query_vector: List[float],
        collection_name: Optional[str] = None,
        top_k: int = 5,
    ):
        """Vector search using self-provided vectors."""
        if not query_vector:
            return []
        col = self._collection(collection_name)
        meta = MetadataQuery(distance=True) if MetadataQuery else None
        res = col.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=meta,  # type: ignore[arg-type]
            return_properties=[
                "text",
                "source_type",
                "source_url",
                "source_file",
                "source_path",
                "page_number",
                "chunk_index",
                "document_index",
                "start_index",
                "embedding_model",
                "metadata_json",
            ],
        )
        return getattr(res, "objects", []) or []

    def search_by_keyword(
        self,
        *,
        query_keyword: str,
        collection_name: Optional[str] = None,
        top_k: int = 5,
    ):
        """Keyword search (BM25)."""
        if not query_keyword.strip():
            return []
        col = self._collection(collection_name)
        meta = MetadataQuery(score=True) if MetadataQuery else None
        res = col.query.bm25(
            query=query_keyword,
            query_properties=["text"],
            limit=top_k,
            return_metadata=meta,  # type: ignore[arg-type]
            return_properties=[
                "text",
                "source_type",
                "source_url",
                "source_file",
                "source_path",
                "page_number",
                "chunk_index",
                "document_index",
                "start_index",
                "embedding_model",
                "metadata_json",
            ],
        )
        return getattr(res, "objects", []) or []

    def hybrid_search(
        self,
        *,
        query_keyword: str,
        query_vector: Optional[List[float]] = None,
        collection_name: Optional[str] = None,
        top_k: int = 5,
    ):
        """Hybrid search (keyword + vector)."""
        if not query_keyword.strip() and not query_vector:
            return []
        col = self._collection(collection_name)
        meta = MetadataQuery(score=True) if MetadataQuery else None
        res = col.query.hybrid(
            query=query_keyword,
            vector=query_vector,
            query_properties=["text"],
            limit=top_k,
            return_metadata=meta,  # type: ignore[arg-type]
            return_properties=[
                "text",
                "source_type",
                "source_url",
                "source_file",
                "source_path",
                "page_number",
                "chunk_index",
                "document_index",
                "start_index",
                "embedding_model",
                "metadata_json",
            ],
        )
        return getattr(res, "objects", []) or []

    def delete_collection(self, *, collection_name: Optional[str] = None) -> bool:
        """Delete a collection if it exists."""
        name = collection_name or self.collection_name
        if not self.client.collections.exists(name):
            return False
        self.client.collections.delete(name)
        logger.info("Deleted Weaviate collection: %s", name)
        return True

    def collection_has_data(self, *, collection_name: Optional[str] = None) -> bool:
        """Return True if the collection exists and has at least one object."""
        name = collection_name or self.collection_name
        if not self.client.collections.exists(name):
            return False
        col = self.client.collections.get(name)
        try:
            res = col.query.fetch_objects(limit=1)
            return bool(getattr(res, "objects", []) or [])
        except Exception:
            try:
                res = col.aggregate.over_all(total_count=True)
                total = getattr(res, "total_count", None)
                return bool(total)
            except Exception:
                return False

    def close(self) -> None:
        if self._client_owner and self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.warning("Error closing Weaviate client: %s", exc)

    def __enter__(self) -> "WeaviateStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


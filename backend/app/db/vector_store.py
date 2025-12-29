"""Weaviate vector store operations.

This module is intentionally usable from *both* the backend runtime and
standalone scripts (e.g. `scripts/ingest_data.py`).
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
from weaviate.classes.config import Configure, DataType, Property
from weaviate.util import generate_uuid5

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


class WeaviateVectorStore:
    """Manages Weaviate operations for upserting Documents with self-provided vectors."""

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
            client = weaviate.connect_to_local()
        else:
            url = url or os.getenv("WEAVIATE_CLOUD_URL")
            api_key = api_key or os.getenv("WEAVIATE_API_KEY")
            if not url:
                raise ValueError("Missing Weaviate URL (set WEAVIATE_CLOUD_URL or pass url=...)")
            if not api_key:
                raise ValueError("Missing Weaviate API key (set WEAVIATE_API_KEY or pass api_key=...)")

            # Weaviate client versions differ slightly; try AuthApiKey if available.
            auth = None
            try:
                from weaviate.auth import AuthApiKey  # type: ignore

                auth = AuthApiKey(api_key)
                client = weaviate.connect_to_weaviate_cloud(cluster_url=url, auth_credentials=auth)
            except Exception:
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
            Property(name="text", data_type=DataType.TEXT),
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

    def upsert(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        *,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        batch_size: int = 200,
    ) -> int:
        """Upsert (idempotently) documents and their vectors into Weaviate."""
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")

        name = collection_name or self.collection_name
        self.ensure_collection(collection_name=name)
        collection = self.client.collections.get(name)

        inserted = 0

        # Prefer true upsert if the installed client supports it.
        has_upsert = hasattr(getattr(collection, "data", None), "upsert")
        if has_upsert:
            for doc, vec in zip(documents, embeddings):
                props, uuid = self._to_object(doc, embedding_model=embedding_model)
                collection.data.upsert(properties=props, uuid=uuid, vector=vec)  # type: ignore[attr-defined]
                inserted += 1
            logger.info("Upserted %d object(s) into %s", inserted, name)
            return inserted

        # Fallback: batch insert; if duplicates are rejected by your Weaviate version,
        # you may need to delete-by-uuid or switch to a client version that exposes upsert.
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for doc, vec in zip(documents, embeddings):
                props, uuid = self._to_object(doc, embedding_model=embedding_model)
                batch.add_object(properties=props, uuid=uuid, vector=vec)
                inserted += 1

        logger.info("Inserted %d object(s) into %s", inserted, name)
        return inserted

    def close(self) -> None:
        if self._client_owner and self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.warning("Error closing Weaviate client: %s", exc)

    def __enter__(self) -> "WeaviateVectorStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False




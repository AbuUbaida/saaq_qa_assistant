"""Deprecated: use `backend.app.db.vector_store` instead.

This file remains as a compatibility shim for the previous misspelled module name.
"""

from __future__ import annotations

from .vector_store import (  # noqa: F401
    DEFAULT_COLLECTION_NAME,
    WeaviateStore,
    WeaviateVectorStore,
    make_chunk_id,
)
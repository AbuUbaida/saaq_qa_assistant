"""FastAPI entrypoint for the backend.

Keep this file boring and obvious:
- create the app
- include the API router(s)
- configure CORS

Run locally:
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.logging import get_logger
from .api.v1.chat import router as chat_router
from .api.v1.feedback import router as feedback_router
from .db.vector_store import WeaviateStore
from .ingest.indexer import index_embeddings_files
from .observability.otel import setup_otel

logger = get_logger(__name__)


def _wait_for_weaviate(settings) -> None:
    """Block until Weaviate is ready or timeout expires."""
    deadline = time.monotonic() + settings.auto_index_wait_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            store = WeaviateStore(
                use_local=settings.weaviate_use_local,
                url=settings.weaviate_cloud_url,
                api_key=settings.weaviate_api_key,
                collection_name=settings.weaviate_collection,
            )
            store.close()
            return
        except Exception as exc:  # pragma: no cover - best effort wait
            last_error = exc
            logger.info("Waiting for Weaviate to be ready...")
            time.sleep(2)
    raise RuntimeError("Weaviate not ready before timeout") from last_error


def _index_on_startup(settings) -> None:
    if not settings.auto_index_on_startup:
        return

    _wait_for_weaviate(settings)

    store = WeaviateStore(
        use_local=settings.weaviate_use_local,
        url=settings.weaviate_cloud_url,
        api_key=settings.weaviate_api_key,
        collection_name=settings.weaviate_collection,
    )
    try:
        store.ensure_collection(collection_name=settings.weaviate_collection)
        if store.collection_has_data(collection_name=settings.weaviate_collection):
            logger.info("Collection already has data; skipping auto-indexing.")
            return
    finally:
        store.close()

    html_dir = Path(settings.auto_index_html_dir)
    pdf_dir = Path(settings.auto_index_pdf_dir)

    html_files = sorted(html_dir.glob("*.jsonl")) if html_dir.exists() else []
    pdf_files = sorted(pdf_dir.glob("*.jsonl")) if pdf_dir.exists() else []

    if not html_files and not pdf_files:
        logger.warning(
            "No embeddings found in expected directories. Checked: %s and %s",
            html_dir,
            pdf_dir,
        )
        return

    if html_files:
        logger.info("Indexing %d HTML embeddings file(s)...", len(html_files))
        index_embeddings_files(
            html_files,
            collection_name=settings.weaviate_collection,
            use_local=settings.weaviate_use_local,
            weaviate_url=settings.weaviate_cloud_url,
            weaviate_api_key=settings.weaviate_api_key,
            batch_size=settings.auto_index_batch_size,
        )

    if pdf_files:
        logger.info("Indexing %d PDF embeddings file(s)...", len(pdf_files))
        index_embeddings_files(
            pdf_files,
            collection_name=settings.weaviate_collection,
            use_local=settings.weaviate_use_local,
            weaviate_url=settings.weaviate_cloud_url,
            weaviate_api_key=settings.weaviate_api_key,
            batch_size=settings.auto_index_batch_size,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_otel(app)
    if settings.auto_index_on_startup:
        logger.info("Auto-index on startup enabled.")
        _index_on_startup(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    # CORS (simple + configurable via env var)
    origins_raw = (settings.cors_allow_origins or "*").strip()
    allow_origins = ["*"] if origins_raw == "*" else [o.strip() for o in origins_raw.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
    app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])

    logger.info("Backend started (cors_origins=%s)", allow_origins)
    return app


app = create_app()

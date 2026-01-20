"""FastAPI entrypoint for the backend.

Keep this file boring and obvious:
- create the app
- include the API router(s)
- configure CORS

Run locally:
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.logging import get_logger
from .api.v1.chat import router as chat_router
from .api.v1.feedback import router as feedback_router

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.app_name)

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

"""Backend configuration (small + easy to read).

We keep this intentionally simple for a portfolio project:
- read env vars (optionally via .env)
- expose a cached `get_settings()` function
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    # App
    app_name: str = "SAAQ QA Assistant"
    log_level: str = "INFO"
    cors_allow_origins: str = "*"  # comma-separated or "*"

    # Weaviate
    weaviate_use_local: bool = True
    weaviate_cloud_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_collection: str = "SAAQDocuments"

    # Embeddings / LLM (HuggingFace endpoints)
    hf_api_key: Optional[str] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "huggingface"
    hf_llm_model: str = "HuggingFaceH4/zephyr-7b-beta"
    llm_temperature: float = 0.0

    # Retrieval defaults
    default_top_k: int = 5

    # Startup indexing (optional)
    auto_index_on_startup: bool = False
    auto_index_html_dir: str = "/app/data/embeddings/html_documents"
    auto_index_pdf_dir: str = "/app/data/embeddings/pdf_documents"
    auto_index_batch_size: int = 200
    auto_index_wait_seconds: int = 120


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once (cached)."""
    load_dotenv()

    return Settings(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        cors_allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*"),
        weaviate_use_local=_env_bool("WEAVIATE_USE_LOCAL", True),
        weaviate_cloud_url=os.getenv("WEAVIATE_CLOUD_URL"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
        weaviate_collection=os.getenv("WEAVIATE_COLLECTION", "SAAQDocuments"),
        hf_api_key=os.getenv("HF_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        llm_provider=os.getenv("LLM_PROVIDER", "huggingface"),
        hf_llm_model=os.getenv("HF_LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        default_top_k=_env_int("TOP_K", 5),
        auto_index_on_startup=_env_bool("AUTO_INDEX_ON_STARTUP", False),
        auto_index_html_dir=os.getenv("AUTO_INDEX_HTML_DIR", "/app/data/embeddings/html_documents"),
        auto_index_pdf_dir=os.getenv("AUTO_INDEX_PDF_DIR", "/app/data/embeddings/pdf_documents"),
        auto_index_batch_size=_env_int("AUTO_INDEX_BATCH_SIZE", 200),
        auto_index_wait_seconds=_env_int("AUTO_INDEX_WAIT_SECONDS", 120),
    )



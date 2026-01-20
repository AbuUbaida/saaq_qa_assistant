"""RAGAS metrics helpers (small + practical)."""

from __future__ import annotations

import os
from typing import Any, Optional

from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

from ..core.config import get_settings


def create_metrics(
    *,
    llm: Any,
    embeddings: Any,
    include_faithfulness: bool = True,
    include_answer_relevance: bool = True,
    include_context_precision: bool = True,
) -> list[Any]:
    """Return a plain list of RAGAS metric objects."""
    if not any([include_faithfulness, include_answer_relevance, include_context_precision]):
        raise ValueError("At least one metric must be enabled.")
    if llm is None:
        raise ValueError("llm is required for RAGAS metrics")
    if embeddings is None:
        raise ValueError("embeddings is required for RAGAS metrics")

    metrics: list[Any] = []
    if include_faithfulness:
        metrics.append(Faithfulness(llm=llm))
    if include_answer_relevance:
        metrics.append(AnswerRelevancy(llm=llm, embeddings=embeddings))
    if include_context_precision:
        # Some versions accept embeddings, some accept llm; we keep the simplest working shape.
        metrics.append(ContextPrecision(llm=llm))
    return metrics


def create_ragas_llm_and_embeddings(
    *,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    embeddings_provider: Optional[str] = None,
    embeddings_model: Optional[str] = None,
) -> tuple[Any, Any]:
    """Create RAGAS-compatible LLM + embeddings using env vars.

    Defaults are intentionally boring:
    - HuggingFace, using the same HF key/model as the main app (if available)
    - You can override with env vars: EVAL_LLM_PROVIDER / EVAL_LLM_MODEL / EVAL_EMBEDDINGS_*
    """
    settings = get_settings()

    # Import factories (their location changes between ragas versions).
    try:
        from ragas.llms import llm_factory  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Could not import ragas.llms.llm_factory") from exc

    try:
        from ragas.embeddings.base import embedding_factory  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from ragas.embeddings import embedding_factory  # type: ignore
        except Exception as exc:
            raise ImportError("Could not import RAGAS embedding_factory") from exc

    llm_provider = llm_provider or os.getenv("EVAL_LLM_PROVIDER") or "huggingface"
    embeddings_provider = embeddings_provider or os.getenv("EVAL_EMBEDDINGS_PROVIDER") or llm_provider

    llm_model = llm_model or os.getenv("EVAL_LLM_MODEL") or settings.hf_llm_model
    embeddings_model = embeddings_model or os.getenv("EVAL_EMBEDDINGS_MODEL") or settings.embedding_model

    # Create LLM
    if llm_provider == "huggingface":
        api_key = os.getenv("HF_API_KEY") or settings.hf_api_key
        if not api_key:
            raise ValueError("HF_API_KEY is required for HuggingFace evaluation")
        try:
            llm = llm_factory("huggingface", model=llm_model, api_key=api_key)
        except TypeError:
            llm = llm_factory("huggingface", model=llm_model)  # type: ignore[misc]
    elif llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI evaluation")
        try:
            llm = llm_factory("openai", model=llm_model, api_key=api_key)
        except TypeError:  # pragma: no cover
            llm = llm_factory(llm_model)  # type: ignore[misc]
    else:
        raise ValueError(f"Unsupported EVAL_LLM_PROVIDER: {llm_provider}")

    # Create embeddings
    if embeddings_provider == "huggingface":
        api_key = os.getenv("HF_API_KEY") or settings.hf_api_key
        if not api_key:
            raise ValueError("HF_API_KEY is required for HuggingFace evaluation embeddings")
        try:
            embeddings = embedding_factory("huggingface", model=embeddings_model, api_key=api_key)
        except TypeError:
            embeddings = embedding_factory("huggingface", model=embeddings_model)  # type: ignore[misc]
    elif embeddings_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI evaluation embeddings")
        try:
            embeddings = embedding_factory("openai", model=embeddings_model, api_key=api_key)
        except TypeError:  # pragma: no cover
            embeddings = embedding_factory("openai", model=embeddings_model)  # type: ignore[misc]
    else:
        raise ValueError(f"Unsupported EVAL_EMBEDDINGS_PROVIDER: {embeddings_provider}")

    return llm, embeddings


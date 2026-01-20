"""LLM helpers (HuggingFace endpoint)."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)


def create_llm(
    *,
    provider: str,
    model_name: str,
    temperature: float,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create a chat LLM client (currently HuggingFace endpoints)."""
    settings = get_settings()

    if provider != "huggingface":
        raise ValueError("Only provider='huggingface' is supported in this project")

    api_key = api_key or settings.hf_api_key
    if not api_key:
        raise ValueError("Missing HF_API_KEY (required for HuggingFace LLM calls)")

    endpoint = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",
        huggingfacehub_api_token=api_key,
    )
    return ChatHuggingFace(llm=endpoint, temperature=temperature, **kwargs)


def _normalize_messages(messages: Union[str, list[BaseMessage]]) -> list[BaseMessage]:
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]
    return messages


def generate_response(llm: Any, messages: Union[str, list[BaseMessage]], **kwargs: Any) -> str:
    msgs = _normalize_messages(messages)
    try:
        response = llm.invoke(msgs, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"LLM generation failed: {exc}") from exc

    if hasattr(response, "content"):
        return str(response.content)
    if hasattr(response, "text"):
        return str(response.text)
    return str(response)


def stream_response(llm: Any, messages: Union[str, list[BaseMessage]], **kwargs: Any) -> Iterable[str]:
    msgs = _normalize_messages(messages)
    try:
        for chunk in llm.stream(msgs, **kwargs):
            if hasattr(chunk, "content"):
                yield str(chunk.content)
            elif hasattr(chunk, "text"):
                yield str(chunk.text)
            else:
                yield str(chunk)
    except Exception as exc:
        raise RuntimeError(f"LLM streaming failed: {exc}") from exc

"""Generate embeddings for documents using HuggingFace endpoint embeddings."""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from .text_processor import load_documents_from_jsonl

logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_DELAY_BETWEEN_BATCHES = 1.0


def _require_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("HF_API_KEY")
    if not key:
        raise ValueError("Missing HF_API_KEY (required for HuggingFace endpoint embeddings).")
    return key


@lru_cache(maxsize=4)
def _get_embedder(model_name: str, api_key: str) -> HuggingFaceEndpointEmbeddings:
    return HuggingFaceEndpointEmbeddings(
        model=model_name,
        task="feature-extraction",
        huggingfacehub_api_token=api_key,
    )


def load_documents_for_embedding(jsonl_path: Path | str) -> List[Document]:
    docs = load_documents_from_jsonl(jsonl_path=jsonl_path)
    return [d for d in docs if d.page_content and d.page_content.strip()]


def embed_query(
    query: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
) -> List[float]:
    key = _require_api_key(api_key)
    embedder = _get_embedder(model_name, key)
    return embedder.embed_query(query)


def embed_texts(
    texts: List[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    if not texts:
        return []
    key = _require_api_key(api_key)
    embedder = _get_embedder(model_name, key)
    return embedder.embed_documents(texts)


def embed_documents(
    documents: List[Document],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    texts = [doc.page_content for doc in documents]
    return embed_texts(texts, model_name=model_name, api_key=api_key)


def embed_documents_batch(
    documents: List[Document],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay_between_batches: float = DEFAULT_DELAY_BETWEEN_BATCHES,
) -> List[Optional[List[float]]]:
    """Embed in batches; returns list aligned with input (None for failures)."""
    if not documents:
        return []

    out: List[Optional[List[float]]] = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        texts = [d.page_content for d in batch]
        try:
            out.extend(embed_texts(texts, model_name=model_name, api_key=api_key))
        except Exception as exc:
            logger.warning("Embedding batch failed (%s). Marking %d item(s) as failed.", exc, len(batch))
            out.extend([None] * len(batch))
        if i + batch_size < len(documents):
            time.sleep(delay_between_batches)
    return out


def save_embeddings_to_jsonl(
    *,
    documents: List[Document],
    embeddings: List[List[float]],
    output_dir: Path | str,
    input_path: Path | str,
    model_name: str,
) -> Path:
    if len(documents) != len(embeddings):
        raise ValueError(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    safe_model = model_name.replace("/", "_")
    output_name = f"{input_path.stem}[{safe_model}]{input_path.suffix}"
    output_path = output_dir / output_name

    with output_path.open("w", encoding="utf-8") as handle:
        for doc, emb in zip(documents, embeddings):
            md = dict(doc.metadata or {})
            md["embedding_model"] = model_name
            handle.write(
                json.dumps({"page_content": doc.page_content, "metadata": md, "embedding": emb}, ensure_ascii=False)
                + "\n"
            )
    logger.info("Saved %d embedding record(s) to %s", len(documents), output_path)
    return output_path


def load_embeddings_from_jsonl(jsonl_path: Path | str) -> Tuple[List[Document], List[List[float]]]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    docs: List[Document] = []
    embs: List[List[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        docs.append(Document(page_content=data["page_content"], metadata=data.get("metadata", {})))
        embs.append(data["embedding"])

    if len(docs) != len(embs):
        raise ValueError("Mismatch: documents and embeddings length differ")
    return docs, embs






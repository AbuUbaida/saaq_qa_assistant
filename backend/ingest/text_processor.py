"""Clean & chunk documents (PDF/HTML) for the RAG pipeline."""

from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import DetectorFactory, detect_langs

from ..core.utils import get_repo_root

logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw" / "documents"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"

# Chunking defaults (characters)
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SOURCE_TYPE: Literal["pdf", "html", "auto"] = "auto"


def load_documents_from_jsonl(jsonl_path: Path | str) -> List[Document]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    docs: List[Document] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data: Dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON at line %d in %s: %s (skipping)", line_number, path, exc)
            continue

        page_content = data.get("page_content")
        metadata = data.get("metadata")
        if not isinstance(page_content, str) or not isinstance(metadata, dict):
            logger.warning("Invalid record at line %d in %s (skipping)", line_number, path)
            continue

        docs.append(Document(page_content=page_content, metadata=metadata))

    logger.info("Loaded %d document(s) from %s", len(docs), path)
    return docs


def clean_text_pdf(text: str) -> str:
    if not text:
        return ""

    # Fix hyphenation: word-\nword -> wordword
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "—": "-",
        "–": "-",
        "…": "...",
        "•": "-",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove control chars
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove excessive consecutive repeated characters
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    
    return text


def clean_text_html(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "—": "-",
        "–": "-",
        "…": "...",
        "•": "-",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

    # Normalize whitespace within lines; collapse excessive blank lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_languages(text: str) -> Dict[str, float]:
    if not text or len(text.strip()) < 10:
        return {}

    DetectorFactory.seed = 0
    try:
        langs = detect_langs(text)
        return {lang.lang: lang.prob for lang in langs}
    except Exception:
        return {}


def clean_document(document: Document, source_type: Literal["pdf", "html", "auto"] = DEFAULT_SOURCE_TYPE) -> Document:
    effective_source = source_type if source_type != "auto" else (document.metadata.get("source_type") or "html")
    if effective_source not in ("pdf", "html"):
        effective_source = "html"

    original = document.page_content or ""
    cleaned = clean_text_pdf(original) if effective_source == "pdf" else clean_text_html(original)

    lang_scores = detect_languages(cleaned)
    primary_lang = max(lang_scores.items(), key=lambda x: x[1])[0] if lang_scores else "unknown"

    new_md = dict(document.metadata or {})
    new_md.update(
        {
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
            "language": primary_lang,
            "language_scores": lang_scores,
            "content_length_original": len(original),
            "content_length_cleaned": len(cleaned),
        }
    )
    return Document(page_content=cleaned, metadata=new_md)


def clean_documents(documents: List[Document], source_type: Literal["pdf", "html", "auto"] = DEFAULT_SOURCE_TYPE) -> List[Document]:
    if not documents:
        return []
    cleaned: List[Document] = []
    for doc in documents:
        try:
            cleaned.append(clean_document(doc, source_type=source_type))
        except Exception as exc:
            logger.warning("Failed to clean document (skipping): %s", exc)
    return cleaned


def chunk_documents(
    documents: List[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        add_start_index=True,
        length_function=len,
    )

    chunks: List[Document] = []
    for doc_index, doc in enumerate(documents, start=1):
        try:
            doc_chunks = splitter.split_documents([doc])
        except Exception as exc:
            logger.warning("Failed to chunk document #%d (skipping): %s", doc_index, exc)
            continue

        for chunk_index, chunk in enumerate(doc_chunks, start=1):
            chunk.metadata.update(
                {
                    "chunk_index": chunk_index,
                    "total_chunks": len(doc_chunks),
                    "document_index": doc_index,
                }
            )
            chunks.append(chunk)

    logger.info("Created %d chunk(s) from %d document(s)", len(chunks), len(documents))
    return chunks


def save_jsonl(
    documents: List[Document],
    *,
    output_dir: Path | str,
    input_path: Path | str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    output_path = output_dir / input_path.name
    with output_path.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(
                json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False) + "\n"
            )
    logger.info("Saved %d document(s) to %s", len(documents), output_path)
    return output_path



"""PDF document collector."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Add repo root to sys.path BEFORE importing backend modules
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.core.utils import get_repo_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "raw" / "pdf_sources"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "documents" / "pdf_documents"


def list_pdf_files(source_dir: Path | str = DEFAULT_SOURCE_DIR) -> List[str]:
    path = Path(source_dir)
    if not path.exists():
        return []
    return sorted([p.name for p in path.glob("*.pdf")])


def load_pdf_as_documents(pdf_path: Path | str) -> List[Document]:
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF file not found: %s", path)
        return []

    source_file = path.name
    source_file_id = source_file.split("_")[0]
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        loader = PyPDFLoader(str(path), mode="page")
        documents = loader.load()

        for document in documents:
            document.metadata.update(
                {
                    "collected_at": timestamp,
                    "source_file": source_file,
                    "source_file_id": source_file_id,
                    "source_type": "pdf",
                }
            )
        logger.info("Loaded %d page(s) from %s", len(documents), source_file)
        return documents
    except Exception as exc:
        logger.exception("Failed to load PDF %s: %s", path, exc)
        return []


def collect_pdfs(
    pdf_files: List[str],
    *,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
) -> Dict[str, List[Document]]:
    source_dir = Path(source_dir)
    results: Dict[str, List[Document]] = {}

    for pdf_file in pdf_files:
        candidate_path = Path(pdf_file)
        if candidate_path.exists():
            pdf_path = candidate_path
        else:
            if not pdf_file.endswith(".pdf"):
                pdf_file = f"{pdf_file}.pdf"
            pdf_path = source_dir / pdf_file

        if not pdf_path.exists():
            logger.warning("Source file not found: %s, skipping", pdf_path)
            results[pdf_path.name] = []
            continue

        logger.info("Processing source file: %s", pdf_path.name)
        documents = load_pdf_as_documents(pdf_path)
        if not documents:
            logger.warning("No documents collected from %s", pdf_path)
            results[pdf_path.name] = []
            continue

        results[pdf_path.name] = documents
        logger.info("Collected %d document(s) from %s", len(documents), pdf_path.name)

    return results


def save_documents(
    document_list: List[Document],
    *,
    output_dir: Path | str,
    output_filename: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename

    try:
        with output_file.open("w", encoding="utf-8") as handle:
            for doc in document_list:
                record = {"page_content": doc.page_content, "metadata": doc.metadata}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved %d documents to %s", len(document_list), output_file)
    except Exception:
        logger.exception("Unable to write PDF documents to %s", output_file)
        raise

    return output_file


def collect_and_save(
    pdf_files: List[str],
    *,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, List[Document]]:
    documents_by_file = collect_pdfs(pdf_files=pdf_files, source_dir=source_dir)
    for source_file, documents in documents_by_file.items():
        if not documents:
            continue
        output_filename = Path(source_file).name.replace(".pdf", ".jsonl")
        save_documents(documents, output_dir=output_dir, output_filename=output_filename)
    return documents_by_file


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Collect PDF(s) from data/raw/pdf_sources into JSONL.")
    p.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument(
        "--pdf",
        action="append",
        dest="pdf_files",
        default=None,
        help="PDF filename or full path (repeatable). If omitted, collects all PDFs in source-dir.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pdf_files = args.pdf_files or list_pdf_files(args.source_dir)
    if not pdf_files:
        logger.error("No PDFs found. Put a PDF in %s or pass --pdf <name>.pdf", args.source_dir)
        return 1

    collect_and_save(pdf_files=pdf_files, source_dir=args.source_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Clean + chunk collected documents.

Run:
  python -m scripts.text_processor --input data/raw/documents/pdf_documents/1_drivers_handbook.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add repo root to sys.path BEFORE importing backend modules
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
    
from backend.ingest.text_processor import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SOURCE_TYPE,
    clean_documents,
    chunk_documents,
    load_documents_from_jsonl,
    save_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean & chunk a collected JSONL file.")
    p.add_argument("--input", required=True, help="Input JSONL (raw collected documents).")
    p.add_argument("--output-dir", default=None, help="Output directory (defaults to input file directory).")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument("--source-type", default=DEFAULT_SOURCE_TYPE, choices=["auto", "pdf", "html"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    docs = load_documents_from_jsonl(input_path)
    docs = clean_documents(docs, source_type=args.source_type)
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    save_jsonl(chunks, output_dir=output_dir, input_path=input_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

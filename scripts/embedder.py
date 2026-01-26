"""Generate embeddings for chunked documents.

Run:
  python -m scripts.embedder --input data/processed/pdf_documents/1_drivers_handbook.jsonl
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

from backend.ingest.embedder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DELAY_BETWEEN_BATCHES,
    DEFAULT_EMBEDDING_MODEL,
    embed_documents_batch,
    load_documents_for_embedding,
    save_embeddings_to_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create embeddings JSONL from processed JSONL.")
    p.add_argument("--input", default=None, help="Input processed JSONL file (chunks, no embeddings yet).")
    p.add_argument("--input-dir", default=None, help="Directory of JSONL files to embed.")
    p.add_argument("--output-dir", required=True, help="Output directory for embeddings.")
    p.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY_BETWEEN_BATCHES, help="Delay between batches (seconds).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input and not args.input_dir:
        raise SystemExit("Provide --input or --input-dir")

    input_paths = []
    if args.input:
        input_paths.append(Path(args.input))
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise SystemExit(f"Input directory not found: {input_dir}")
        input_paths.extend(sorted(input_dir.glob("*.jsonl")))

    output_dir = Path(args.output_dir)

    for input_path in input_paths:
    docs = load_documents_for_embedding(input_path)
    embeddings = embed_documents_batch(
        docs,
        model_name=args.model,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
    )

    valid_docs = []
    valid_embs = []
    for doc, emb in zip(docs, embeddings):
        if emb is None:
            continue
        valid_docs.append(doc)
        valid_embs.append(emb)

    save_embeddings_to_jsonl(
        documents=valid_docs,
        embeddings=valid_embs,
            output_dir=output_dir,
            input_path=input_path,
        model_name=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

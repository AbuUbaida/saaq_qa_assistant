"""CLI: index embeddings JSONL into Weaviate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add repo root to sys.path BEFORE importing backend modules
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.ingest.indexer import index_embeddings_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index embeddings JSONL into Weaviate.")
    p.add_argument("--input", default=None, help="Embeddings JSONL file to index.")
    p.add_argument("--input-dir", default=None, help="Directory of embeddings JSONL files to index.")
    p.add_argument("--collection", default=None, help="Override Weaviate collection name.")
    p.add_argument("--cloud", action="store_true", help="Use cloud Weaviate (env vars required).")
    p.add_argument("--batch-size", type=int, default=200, help="Weaviate batch size.")
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

    total_inserted = index_embeddings_files(
        input_paths,
        collection_name=args.collection,
        use_local=not args.cloud,
        batch_size=args.batch_size,
    )
    return 0 if total_inserted >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

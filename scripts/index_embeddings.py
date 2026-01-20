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

from backend.ingest.indexer import index_embeddings_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index embeddings JSONL into Weaviate.")
    p.add_argument("--input", required=True, help="Embeddings JSONL file to index.")
    p.add_argument("--collection", default=None, help="Override Weaviate collection name.")
    p.add_argument("--cloud", action="store_true", help="Use cloud Weaviate (env vars required).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inserted = index_embeddings_jsonl(
        jsonl_path=Path(args.input),
        collection_name=args.collection,
        use_local=not args.cloud,
    )
    return 0 if inserted >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

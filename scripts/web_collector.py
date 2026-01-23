"""Web document collector using FireCrawl."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from urllib.parse import urlsplit
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain_core.documents import Document

# Add repo root to sys.path BEFORE importing backend modules
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.core.utils import get_repo_root

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "raw" / "html_sources"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "documents" / "html_documents"


def _get_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("FIRECRAWL_API_KEY")
    if not key:
        raise ValueError("FireCrawl API key missing. Set FIRECRAWL_API_KEY or pass --api-key.")
    return key


def list_source_files(source_dir: Path | str = DEFAULT_SOURCE_DIR) -> List[str]:
    path = Path(source_dir)
    if not path.exists():
        return []
    return sorted([p.name for p in path.glob("*.txt")])


def read_urls_from_file(file_path: Path | str) -> List[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML source file not found: {path}")

    urls = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    logger.info("Loaded %d URLs from %s", len(urls), path)
    return urls


def _contains_error_codes(documents: List[Document]) -> bool:
    if not documents:
        return False
    joined_content = " ".join(doc.page_content for doc in documents).lower()
    return any(code in joined_content for code in ("401", "403", "500"))


def _output_filename_from_url(url: str) -> str:
    path = urlsplit(url).path
    parts = [p for p in path.split("/") if p]
    if "en" in parts:
        remainder = parts[parts.index("en") + 1 :]
    else:
        remainder = parts
    if not remainder:
        remainder = ["index"]
    cleaned = "_".join(p.replace("-", "_") for p in remainder)
    return f"{cleaned}.jsonl"


def scrape_url(
    url: str,
    *,
    source_file: str,
    api_key: Optional[str] = None,
) -> List[Document]:
    key = _get_api_key(api_key)
    timestamp = datetime.now(timezone.utc).isoformat()
    source_file_id = source_file.split("_")[0]

    try:
        loader = FireCrawlLoader(
            api_key=key,
            url=url,
            mode="scrape",
            params={"onlyMainContent": True},
        )
        documents = loader.load()

        if _contains_error_codes(documents):
            logger.info("Detected error response for %s, retrying with stealth proxy", url)
            loader = FireCrawlLoader(
                api_key=key,
                url=url,
                mode="scrape",
                params={"onlyMainContent": True, "proxy": "stealth"},
            )
            documents = loader.load()

        if not documents:
            logger.warning("No content returned for %s", url)
            return []

        for document in documents:
            document.metadata.update(
                {
                    "collected_at": timestamp,
                    "source_url": url,
                    "source_type": "html",
                    "source_file": source_file,
                    "source_file_id": source_file_id,
                }
            )
        logger.info("Collected %d document(s) from %s", len(documents), url)
        return documents
    except Exception as exc:
        logger.exception("Failed to collect %s: %s", url, exc)
        return []


def collect_web_documents(
    source_files: List[str],
    *,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    api_key: Optional[str] = None,
) -> Dict[str, List[Document]]:
    source_dir = Path(source_dir)
    results: Dict[str, List[Document]] = {}

    for source_file in source_files:
        if not source_file.endswith(".txt"):
            source_file = f"{source_file}.txt"

        source_path = source_dir / source_file
        if not source_path.exists():
            logger.warning("Source file not found: %s, skipping", source_path)
            results[source_file] = []
            continue

        logger.info("Processing source file: %s", source_file)
        url_list = read_urls_from_file(source_path)
        if not url_list:
            logger.warning("No URLs found in %s, skipping", source_file)
            results[source_file] = []
            continue

        collected: List[Document] = []
        for url in url_list:
            documents = scrape_url(url, source_file=source_file, api_key=api_key)
            if documents:
                collected.extend(documents)

            # Respect API limits.
            time.sleep(5)

        if not collected:
            logger.warning("No documents collected from %s", source_path)
            results[source_file] = []
            continue

        results[source_file] = collected
        logger.info("Collected %d document(s) from %s", len(collected), source_file)

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
        logger.exception("Unable to write FireCrawl output to %s", output_file)
        raise

    return output_file


def collect_and_save(
    source_files: List[str],
    *,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    api_key: Optional[str] = None,
) -> Dict[str, List[Document]]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    results: Dict[str, List[Document]] = {}

    for source_file in source_files:
        if not source_file.endswith(".txt"):
            source_file = f"{source_file}.txt"

        source_path = source_dir / source_file
        if not source_path.exists():
            logger.warning("Source file not found: %s, skipping", source_path)
            results[source_file] = []
            continue

        logger.info("Processing source file: %s", source_file)
        url_list = read_urls_from_file(source_path)
        if not url_list:
            logger.warning("No URLs found in %s, skipping", source_file)
            results[source_file] = []
            continue

        collected: List[Document] = []
        for url in url_list:
            output_filename = _output_filename_from_url(url)
            output_path = output_dir / output_filename
            if output_path.exists():
                logger.info("Skipping (already saved): %s", output_filename)
            else:
                documents = scrape_url(url, source_file=source_file, api_key=api_key)
                if documents:
                    save_documents(documents, output_dir=output_dir, output_filename=output_filename)
                    collected.extend(documents)

            # Respect API limits.
            time.sleep(10)

        results[source_file] = collected

    return results


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Collect web pages (Firecrawl) into JSONL.")
    p.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--api-key", default=None, help="FireCrawl API key (optional; falls back to env).")
    p.add_argument(
        "--source",
        action="append",
        dest="source_files",
        default=None,
        help="Text file of URLs to scrape (repeatable). If omitted, uses all *.txt in source-dir.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_files = args.source_files or list_source_files(args.source_dir)
    if not source_files:
        logger.error("No *.txt URL files found. Put one in %s or pass --source <file>.txt", args.source_dir)
        return 1

    collect_and_save(
        source_files=source_files,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        api_key=args.api_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Web document collector using FireCrawl.

Provides WebCollector class for scraping web documents and saving them
to JSONL format with metadata enrichment.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain_core.documents import Document

from ..core.utils import setup_project_path, get_repo_root

setup_project_path()

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "raw" / "html_sources"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "documents" / "html_documents"


class WebCollector:
    """Collects web documents using FireCrawl API.
    
    Handles web scraping, error detection, retry logic, and document
    persistence with metadata enrichment.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        source_dir: Path | str = DEFAULT_SOURCE_DIR,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    ):
        """Initialize web collector.
        
        Args:
            api_key: FireCrawl API key. If None, reads from FIRECRAWL_API_KEY env var.
            source_dir: Default directory containing HTML source files.
            output_dir: Default directory for saving collected documents.
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FireCrawl API key missing. Set the FIRECRAWL_API_KEY env var or pass api_key."
            )
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
    
    @staticmethod
    def read_urls_from_file(file_path: Path | str) -> List[str]:
        """Return sanitized URLs from the provided HTML source files.
        
        Comments (lines starting with ``#``) and blank/whitespace-only lines are ignored.
        
        Args:
            file_path: Path to directory containing HTML source files.
        
        Returns:
            List of sanitized URL strings.
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
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
    
    @staticmethod
    def _contains_error_codes(documents: List[Document]) -> bool:
        """Return True when FireCrawl returned an error page instead of the content.
        
        Args:
            documents: List of Document objects to check.
        
        Returns:
            True if error codes (401, 403, 500) are detected in content.
        """
        if not documents:
            return False
        joined_content = " ".join(doc.page_content for doc in documents).lower()
        return any(code in joined_content for code in ("401", "403", "500"))
    
    def _scrape_urls(
        self,
        urls: List[str],
        source_file: str,
    ) -> List[Document]:
        """Scrape URLs and return documents with metadata.
        
        Args:
            urls: List of URLs to scrape.
            source_file: Optional source file name for metadata.
            
        Returns:
            List of Document objects enriched with metadata.
        """
        collected_docs: List[Document] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        source_file_id = source_file.split("_")[0]
        
        for url in urls:
            try:
                loader = FireCrawlLoader(
                    api_key=self.api_key,
                    url=url,
                    mode="scrape",
                    params={"onlyMainContent": True},
                )
                documents = loader.load()
                
                if self._contains_error_codes(documents):
                    logger.info(
                        "Detected error response for %s, retrying with stealth proxy", url
                    )
                    loader = FireCrawlLoader(
                        api_key=self.api_key,
                        url=url,
                        mode="scrape",
                        params={"onlyMainContent": True, "proxy": "stealth"},
                    )
                    documents = loader.load()
                
                if not documents:
                    logger.warning("No content returned for %s", url)
                    continue

                for document in documents:
                    metadata = {
                        "collected_at": timestamp,
                        "source_url": url,
                        "source_type": "html",
                        "source_file": source_file,
                        "source_file_id": source_file_id,
                    }
                    document.metadata.update(metadata)
                    collected_docs.append(document)
                logger.info("Collected %d document(s) from %s", len(documents), url)
            except Exception as exc:
                logger.exception("Failed to collect %s: %s", url, exc)

        return collected_docs
    
    def collect(
        self,
        source_files: List[str],
        source_dir: Optional[Path | str] = None,
    ) -> Dict[str, List[Document]]:
        """Collect web documents from source files.
        
        - Reads URLs from each source file in source_dir
        - Scrapes all URLs from that file
        - Returns a dictionary mapping source file names to their documents
        
        Args:
            source_files: List of source file names (.txt) to process.
            source_dir: Directory containing HTML source files. If None, uses self.source_dir.
            
        Returns:
            Dictionary mapping source file names to their Document lists.
        """
        source_dir = Path(source_dir) if source_dir else self.source_dir
        
        results: Dict[str, List[Document]] = {}
        
        for source_file in source_files:
            # Ensure source_file has .txt extension
            if not source_file.endswith('.txt'):
                source_file = f"{source_file}.txt"
            
            source_path = source_dir / source_file
            
            if not source_path.exists():
                logger.warning("Source file not found: %s, skipping", source_path)
                results[source_file] = []
                continue
            
            logger.info("Processing source file: %s", source_file)
            
            # Read URLs from source file
            url_list = self.read_urls_from_file(source_path)
            
            if not url_list:
                logger.warning("No URLs found in %s, skipping", source_file)
                results[source_file] = []
                continue
            
            # Scrape URLs
            documents = self._scrape_urls(url_list, source_file=source_file)

            if not documents:
                logger.warning("No documents collected from %s", source_path)
                results[source_file] = []
                continue
            
            results[source_file] = documents
            logger.info(
                "Completed processing %s: collected %d document(s)",
                source_file,
                len(documents),
            )
        
        return results
    
    def save(
        self,
        document_list: List[Document],
        output_dir: Path | str,
        output_filename: str,
    ) -> Path:
        """Persist documents as JSONL under output_dir.
        
        Args:
            document_list: List of Document objects to save.
            output_dir: Directory where the JSONL file will be written.
            output_filename: Name of the output JSONL file.
        
        Returns:
            Path to the created JSONL file.
        
        Raises:
            OSError: If the file cannot be written.
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / output_filename

        try:
            with output_file.open("w", encoding="utf-8") as handle:
                for doc in document_list:
                    record = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Saved %d documents to %s", len(document_list), output_file)
        except Exception:
            logger.exception("Unable to write FireCrawl output to %s", output_file)
            raise

        return output_file
    
    def collect_and_save(
        self,
        source_files: List[str],
        source_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
    ) -> Dict[str, List[Document]]:
        """Collect documents from source files and save them to JSONL.
        
        Args:
            source_files: List of source file names (.txt) to process.
            source_dir: Directory containing source files. If None, uses self.source_dir.
            output_dir: Directory for saving. If None, uses self.output_dir.
        
        Returns:
            Dictionary mapping source file names to their Document lists.
        """
        documents_w_filename = self.collect(
            source_files=source_files,
            source_dir=source_dir,
        )   

        if documents_w_filename:
            output_dir = output_dir or self.output_dir
            for source_file, documents in documents_w_filename.items():
                output_filename = source_file.replace('.txt', '.jsonl')
                self.save(documents, output_dir, output_filename)
        
        return documents_w_filename


# def collect_web_documents(
#     source_files: List[str],
#     api_key: Optional[str] = None,
#     source_dir: Path | str = DEFAULT_SOURCE_DIR,
#     output_dir: Path | str = DEFAULT_OUTPUT_DIR,
# ) -> Dict[str, List[Document]]:
#     """Collect web documents from source files and save them to JSONL.
    
#     Backward compatibility wrapper for WebCollector.
    
#     Args:
#         source_files: List of source file names (.txt) to process. Each file should be in source_dir.
#         api_key: FireCrawl API key. When None we attempt to read FIRECRAWL_API_KEY.
#         source_dir: Directory containing source files.
#         output_dir: Directory for saving.
    
#     Returns:
#         Dictionary mapping source file names to their Document lists.
#     """
#     collector = WebCollector(api_key=api_key)
#     return collector.collect_and_save(source_files=source_files, source_dir=source_dir, output_dir=output_dir)
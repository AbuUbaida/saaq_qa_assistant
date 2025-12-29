"""Text processing and chunking utilities.

Provides TextProcessor class for cleaning, language detection, and chunking
of documents with metadata enrichment.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect_langs, DetectorFactory

from ..core.utils import setup_project_path, get_repo_root

setup_project_path()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw" / "documents"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"

# Chunking configuration constants
DEFAULT_CHUNK_SIZE = 1500  # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 200  # Overlap between chunks


class TextProcessor:
    """Processes and chunks documents with cleaning and language detection.
    
    Handles text cleaning (PDF/HTML specific), language detection, chunking,
    and document persistence to JSONL format.
    """
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        input_dir: Path | str = DEFAULT_INPUT_DIR,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        source_type: Literal["pdf", "html", "auto"] = "auto",
    ):
        """Initialize text processor.
        
        Args:
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            input_dir: Directory containing the JSONL files to process.
            output_dir: Directory where the chunked documents will be saved.
            source_type: Default source type ('pdf', 'html', or 'auto').
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.source_type = source_type
    
    @staticmethod
    def load_documents_from_jsonl(jsonl_path: Path | str) -> List[Document]:
        """Load LangChain Document objects from a JSONL file.
        
        Each line in the JSONL file should contain a JSON object with:
        - page_content: The text content of the document
        - metadata: Dictionary containing document metadata
        
        Args:
            jsonl_path: Path to the JSONL file containing document records.
        
        Returns:
            List of Document objects reconstructed from JSONL records.
        
        Raises:
            FileNotFoundError: If the JSONL file does not exist.
            ValueError: If a line contains invalid JSON or missing required fields.
        """
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        document_list = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    json_data: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Invalid JSON at line %d in %s: %s. Skipping.",
                        line_number,
                        path,
                        e,
                    )
                    continue

                if "metadata" not in json_data:
                    logger.warning(
                        "Missing required 'metadata' key at line %d in %s. Skipping.",
                        line_number,
                        path,
                    )
                    continue

                if "page_content" not in json_data:
                    logger.warning(
                        "Missing required 'page_content' key at line %d in %s. Skipping.",
                        line_number,
                        path,
                    )
                    continue

                document_list.append(
                    Document(
                        page_content=json_data["page_content"],
                        metadata=json_data["metadata"],
                    )
                )

        logger.info("Loaded %d document(s) from %s", len(document_list), path)
        return document_list
    
    @staticmethod
    def clean_text_pdf(text: str) -> str:
        """Apply cleaning rules specifically for PDF-extracted text.
        
        PDFs often have formatting issues like:
        - Excessive whitespace (multiple spaces, tabs, newlines)
        - Broken words across lines (hyphenation artifacts)
        - Special characters and encoding issues
        - Page numbers and headers/footers mixed in
        
        Args:
            text: Raw text extracted from PDF.
        
        Returns:
            Cleaned text with normalized whitespace and formatting issues resolved.
        """
        if not text:
            return ""

        # Fix hyphenation artifacts (word-\nword → wordword)
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        # Normalize unicode characters (NFKC: compatibility + composition)
        text = unicodedata.normalize("NFKC", text)

        # Replace problematic unicode characters with ASCII equivalents
        replacements = {
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
            "—": "-",
            "–": "-",
            "…": "...",
            "•": "-",
            "°": " degrees ",
        }
        for bad_char, good_char in replacements.items():
            text = text.replace(bad_char, good_char)

        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove excessive consecutive repeated characters
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        return text
    
    @staticmethod
    def clean_text_html(text: str) -> str:
        """Apply cleaning rules for HTML-extracted text.
        
        HTML text from FireCrawl is usually cleaner, but may still need:
        - Whitespace normalization
        - Removal of navigation elements that slipped through
        - Fixing of encoding issues
        
        Args:
            text: Raw text extracted from HTML.
        
        Returns:
            Cleaned text with normalized formatting.
        """
        if not text:
            return ""

        # Decode HTML entities first
        text = html.unescape(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)

        # Replace problematic unicode characters
        replacements = {
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
            "—": "-",
            "–": "-",
            "…": "...",
            "•": "-",
            "°": " degrees ",
        }
        for bad_char, good_char in replacements.items():
            text = text.replace(bad_char, good_char)

        # Remove control characters (preserve newlines)
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

        # Normalize whitespace within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Normalize excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
    
    @staticmethod
    def detect_languages(text: str) -> Dict[str, float]:
        """Detect probable languages in the text and return probabilities.
        
        For SAAQ content, expect primarily French (fr) and English (en).
        
        Args:
            text: Text to analyze for language detection.
        
        Returns:
            Dictionary mapping language codes to their probability scores.
        """
        if not text or len(text.strip()) < 10:
            return {}

        DetectorFactory.seed = 0  # Set seed for reproducibility

        try:
            lang_list = detect_langs(text)
            return {lang.lang: lang.prob for lang in lang_list}
        except Exception as e:
            logger.debug("Language detection failed: %s", e)
            return {}
    
    def clean_document(
        self,
        document: Document,
        source_type: Literal["pdf", "html", "auto"] = "auto",
    ) -> Document:
        """Clean a single Document's page_content based on its source type.
        
        Args:
            document: LangChain Document to clean.
            source_type: Type of source ('pdf', 'html', or 'auto').
                If 'auto', uses source_type from document metadata.
        
        Returns:
            New Document instance with cleaned page_content and enriched metadata.
        """
        source_type = source_type or self.source_type
        
        # Determine source type
        if source_type == "auto":
            source_type = document.metadata.get("source_type")
            if source_type not in ("pdf", "html"):
                logger.warning(
                    "Unknown source type '%s', defaulting to 'html'", source_type
                )
                source_type = "html"

        # Clean based on source type
        original_content = document.page_content
        if source_type == "pdf":
            cleaned_content = self.clean_text_pdf(original_content)
        else:  # html
            cleaned_content = self.clean_text_html(original_content)

        # Detect language from cleaned content
        lang_scores = self.detect_languages(cleaned_content)
        primary_lang = max(lang_scores.items(), key=lambda x: x[1])[0] if lang_scores else "unknown"

        # Build new metadata
        timestamp = datetime.now(timezone.utc).isoformat()
        new_metadata = document.metadata.copy()
        new_metadata.update(
            {
                "cleaned_at": timestamp,
                "language": primary_lang,
                "language_scores": lang_scores,
                "content_length_original": len(original_content),
                "content_length_cleaned": len(cleaned_content),
            }
        )

        return Document(page_content=cleaned_content, metadata=new_metadata)
    
    def clean(
        self,
        documents: List[Document],
        source_type: Literal["pdf", "html", "auto"] = "auto",
    ) -> List[Document]:
        """Clean a batch of documents.
        
        Args:
            documents: List of Document objects to clean.
            source_type: Type of source ('pdf', 'html', or 'auto').
                If 'auto', uses source_type from document metadata.
        
        Returns:
            List of successfully cleaned Document objects.
        """
        if not documents:
            logger.warning("No documents provided for cleaning")
            return []

        cleaned_documents = []
        total_documents = len(documents)

        for idx, document in enumerate(documents, start=1):
            try:
                cleaned_doc = self.clean_document(document, source_type)
                cleaned_documents.append(cleaned_doc)

                if idx % 50 == 0:
                    logger.info("Cleaned %d/%d documents...", idx, total_documents)

            except Exception as e:
                logger.warning(
                    "Failed to clean document #%d: %s. Skipping.", idx, e
                )
                continue

        logger.info(
            "Cleaned %d/%d documents successfully", len(cleaned_documents), total_documents
        )
        return cleaned_documents
    
    def chunk(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ) -> List[Document]:
        """Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to chunk.
            chunk_size: Target size of each chunk. If None, uses self.chunk_size.
            chunk_overlap: Overlap between chunks. If None, uses self.chunk_overlap.
            separators: Custom separators for splitting. If None, uses defaults.
        
        Returns:
            List of chunked Document objects with enriched metadata.
        """
        if not documents:
            logger.warning("No documents provided for chunking")
            return []

        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        # Initialize splitter
        if separators is None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
                length_function=len,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                add_start_index=True,
                length_function=len,
            )

        chunked_documents = []
        total_documents = len(documents)

        for doc_idx, document in enumerate(documents, start=1):
            try:
                chunks = splitter.split_documents([document])

                for chunk_idx, chunk in enumerate(chunks, start=1):
                    # chunk.metadata = chunk.metadata.copy()
                    chunk.metadata.update(
                        {
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "document_index": doc_idx,
                        }
                    )
                    chunked_documents.append(chunk)

                logger.debug(
                    "Document %d split into %d chunk(s)", doc_idx, len(chunks)
                )

                if doc_idx % 50 == 0:
                    logger.info("Chunked %d/%d documents...", doc_idx, total_documents)

            except Exception as e:
                logger.warning(
                    "Failed to chunk document #%d: %s. Skipping.", doc_idx, e
                )
                continue

        logger.info(
            "Created %d chunk(s) from %d document(s)",
            len(chunked_documents),
            total_documents,
        )
        return chunked_documents
    
    def save_chunked_documents(
        self,
        documents: List[Document],
        output_filename: str,
        output_dir: Optional[Path | str] = None,
    ) -> Path:
        """Save chunked documents to a JSONL file.
        
        Args:
            documents: List of chunked Document objects to save.
            output_filename: Name of the output JSONL file.
            output_dir: Directory where the JSONL file will be written.
        
        Returns:
            Path to the created JSONL file.
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_path = output_dir / output_filename

        if output_path.is_dir() or output_path.suffix != ".jsonl":
            output_path = output_path / "chunked_documents.jsonl"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with output_path.open("w", encoding="utf-8") as handle:
                for doc in documents:
                    record = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info("Saved %d chunked document(s) to %s", len(documents), output_path)
        except Exception:
            logger.exception("Unable to write chunked documents to %s", output_path)
            raise

        return output_path
    
    def process_and_chunk(
        self,
        jsonl_filenames: List[str],
        input_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        source_type: Optional[str] = None,
    ) -> List[str]:
        """Complete pipeline: load, clean, and chunk documents from JSONL files.
        
        Processes each JSONL file independently and saves output files with the same
        names in the output directory.
        
        Args:
            jsonl_filenames: List of filenames of JSONL files to process.
            input_dir: Directory containing the JSONL files to process. If None, uses
                default path based on source_type (html_documents or pdf_documents).
            output_dir: Directory where the chunked documents will be saved. If None, uses
                default path based on source_type (html_documents or pdf_documents).
            source_type: Type of source ('pdf', 'html', or 'auto'). If None, uses self.source_type.
                If 'auto' and directories are not provided, extracts source_type from document metadata.
        
        Returns:
            List of output JSONL filenames that were successfully created.
        
        Raises:
            FileNotFoundError: If source_type is 'auto' and the first file cannot be found.
            ValueError: If source_type is 'auto' and:
                - No documents are found in the first file
                - Metadata is missing 'source_type' key
                - 'source_type' value is invalid (not 'pdf' or 'html')
        """
        if not jsonl_filenames:
            logger.warning("No JSONL files provided for processing")
            return []

        source_type = source_type or self.source_type
        
        # Determine source_type for directory paths
        # If source_type is "auto" and directories are not provided, extract from document metadata
        dir_source_type = source_type
        if source_type == "auto" and (input_dir is None or output_dir is None):
            # Extract source_type from the first document in the first file
            # Ensure filename has .jsonl extension
            first_filename = jsonl_filenames[0]
            if not first_filename.endswith('.jsonl'):
                first_filename = f"{first_filename}.jsonl"
            
            # Try to find the first file in default locations or provided input_dir
            first_input_file = None
            if input_dir is not None:
                # Use provided input_dir
                first_input_file = Path(input_dir) / first_filename
                if not first_input_file.exists():
                    raise FileNotFoundError(
                        f"Cannot extract source_type: file not found: {first_input_file}"
                    )
            else:
                # Try both default locations (html_documents, pdf_documents)
                for default_type in ["html", "pdf"]:
                    candidate = DEFAULT_INPUT_DIR / f"{default_type}_documents" / first_filename
                    if candidate.exists():
                        first_input_file = candidate
                        break
                
                if first_input_file is None:
                    raise FileNotFoundError(
                        f"Cannot extract source_type: file '{first_filename}' not found in default locations: "
                        f"{DEFAULT_INPUT_DIR / 'html_documents'} or {DEFAULT_INPUT_DIR / 'pdf_documents'}"
                    )
            
            # Load first document to extract source_type
            documents = self.load_documents_from_jsonl(first_input_file)
            if not documents:
                raise ValueError(
                    f"Cannot extract source_type: no documents found in file: {first_input_file}"
                )
            
            # Extract source_type from metadata
            extracted_type = documents[0].metadata.get("source_type")
            if extracted_type is None:
                raise ValueError(
                    f"Cannot extract source_type: metadata is missing 'source_type' key in file: {first_input_file}"
                )
            
            if extracted_type not in ("pdf", "html"):
                raise ValueError(
                    f"Invalid source_type '{extracted_type}' in metadata. Expected 'pdf' or 'html'. "
                    f"File: {first_input_file}"
                )
            
            dir_source_type = extracted_type
            logger.info(
                "Extracted source_type '%s' from document metadata", extracted_type
            )
        
        # Set default input_dir if not provided
        if input_dir is None:
            input_dir = DEFAULT_INPUT_DIR / f"{dir_source_type}_documents"
        else:
            input_dir = Path(input_dir)
        
        # Set default output_dir if not provided
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR / f"{dir_source_type}_documents"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filenames = []
        total_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "languages": {},
        }

        logger.info("Processing %d JSONL file(s) from %s...", len(jsonl_filenames), input_dir)

        for jsonl_filename in jsonl_filenames:
            # Ensure filename has .jsonl extension
            if not jsonl_filename.endswith('.jsonl'):
                jsonl_filename = f"{jsonl_filename}.jsonl"
            
            input_file = input_dir / jsonl_filename
            
            if not input_file.exists():
                logger.warning("Input file not found: %s, skipping", input_file)
                total_stats["files_failed"] += 1
                continue

            logger.info("Processing file: %s", jsonl_filename)

            try:
                # Load documents from this file
                documents = self.load_documents_from_jsonl(input_file)
                
                if not documents:
                    logger.warning("No documents found in %s, skipping", jsonl_filename)
                    total_stats["files_failed"] += 1
                    continue

                total_stats["total_documents"] += len(documents)

                # Clean documents
                logger.debug("Cleaning %d document(s) from %s...", len(documents), jsonl_filename)
                cleaned_documents = self.clean(documents, source_type=source_type)

                if not cleaned_documents:
                    logger.warning("No documents successfully cleaned from %s, skipping", jsonl_filename)
                    total_stats["files_failed"] += 1
                    continue

                # Chunk documents
                logger.debug(
                    "Chunking documents from %s (size=%d, overlap=%d)...",
                    jsonl_filename,
                    self.chunk_size,
                    self.chunk_overlap,
                )
                chunked_documents = self.chunk(cleaned_documents)

                if not chunked_documents:
                    logger.warning("No chunks created from %s, skipping", jsonl_filename)
                    total_stats["files_failed"] += 1
                    continue

                # Save chunked documents with the same filename
                output_filename = jsonl_filename  # Keep the same filename
                saved_path = self.save_chunked_documents(
                    chunked_documents,
                    output_filename=output_filename,
                    output_dir=output_dir,
                )

                # Collect statistics
                for doc in chunked_documents:
                    lang = doc.metadata.get("language", "unknown")
                    total_stats["languages"][lang] = total_stats["languages"].get(lang, 0) + 1

                # Track the output filename
                output_filenames.append(output_filename)
                total_stats["total_chunks"] += len(chunked_documents)
                total_stats["files_processed"] += 1

                logger.info(
                    "Completed %s: %d document(s) -> %d chunk(s) -> %s",
                    jsonl_filename,
                    len(documents),
                    len(chunked_documents),
                    saved_path,
                )

            except Exception as e:
                logger.error("Failed to process %s: %s. Skipping.", jsonl_filename, e, exc_info=True)
                total_stats["files_failed"] += 1
                continue

        # Log final statistics
        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info("  Files processed: %d", total_stats["files_processed"])
        logger.info("  Files failed: %d", total_stats["files_failed"])
        logger.info("  Total documents processed: %d", total_stats["total_documents"])
        logger.info("  Total chunks created: %d", total_stats["total_chunks"])
        logger.info("  Languages detected: %s", dict(total_stats["languages"]))
        logger.info("  Output directory: %s", output_dir)
        logger.info("=" * 60)

        return output_filenames


# def process_and_chunk_documents(
#     jsonl_filenames: List[str],
#     input_dir: Optional[Path | str] = None,
#     output_dir: Optional[Path | str] = None,
#     chunk_size: int = DEFAULT_CHUNK_SIZE,
#     chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
#     source_type: str = "auto",
# ) -> List[str]:
#     """Complete pipeline: load, clean, and chunk documents from JSONL files.
    
#     Backward compatibility wrapper for TextProcessor.process_and_chunk().
#     Adapts the old interface (list of Path objects, single output file) to the
#     new interface (list of filenames, input_dir, output_dir).
    
#     Args:
#         jsonl_filenames: List of JSONL file names to process.
#         input_dir: Directory containing the JSONL files to process. If None, uses
#             default path based on source_type (html_documents or pdf_documents).
#         output_dir: Directory where the chunked documents will be saved. If None, uses
#             default path based on source_type (html_documents or pdf_documents).
#         chunk_size: Target size of each chunk in characters.
#         chunk_overlap: Number of characters to overlap between chunks.
#         source_type: Type of source ('pdf', 'html', or 'auto'). If 'auto' and directories
#             are not provided, extracts source_type from document metadata. Raises ValueError
#             if metadata is missing 'source_type' key or if extraction fails.
    
#     Returns:
#         List of output JSONL filenames that were successfully created.
#     """
#     if not jsonl_filenames:
#         logger.warning("No JSONL files provided for processing")
#         return []
    
#     processor = TextProcessor(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         source_type=source_type,
#     )
#     return processor.process_and_chunk(
#         jsonl_filenames=jsonl_filenames,
#         input_dir=input_dir,
#         output_dir=output_dir,
#     )

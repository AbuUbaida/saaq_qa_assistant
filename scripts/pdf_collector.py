"""PDF document collector.

Provides PDFCollector class for loading PDF files and extracting text
with metadata enrichment.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ..core.utils import setup_project_path, get_repo_root

setup_project_path()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "raw" / "pdf_sources"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "documents" / "pdf_documents"


class PDFCollector:
    """Collects documents from PDF files.
    
    Handles PDF loading, text extraction, metadata enrichment, and
    document persistence to JSONL format.
    """
    
    def __init__(
        self,
        source_dir: Path | str = DEFAULT_SOURCE_DIR,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    ):
        """Initialize PDF collector.
        
        Args:
            source_dir: Default directory containing PDF files.
            output_dir: Default directory for saving collected documents.
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
    
    # @staticmethod
    # def list_pdf_files(directory: Path | str) -> List[Path]:
    #     """Return sorted list of PDF files in the specified directory.
        
    #     Args:
    #         directory: Path to directory containing PDF files.
        
    #     Returns:
    #         Sorted list of Path objects for each PDF file found.
        
    #     Raises:
    #         FileNotFoundError: If the directory does not exist.
    #         NotADirectoryError: If the path exists but is not a directory.
    #     """
    #     path = Path(directory)
    #     if not path.exists():
    #         raise FileNotFoundError(f"PDF source directory not found: {path}")
    #     if not path.is_dir():
    #         raise NotADirectoryError(f"PDF source path is not a directory: {path}")

    #     pdf_files = sorted(path.glob("*.pdf"))
    #     logger.info("Found %d PDF file(s) in %s", len(pdf_files), path)
        
    #     return pdf_files
    
    @staticmethod
    def load_pdf_as_documents(pdf_path: Path) -> List[Document]:
        """Load a PDF file and return LangChain Document instances (one per page).
        
        Each document is enriched with metadata including source file, source file id,
        and collection timestamp.
        
        Args:
            pdf_path: Path to the PDF file to load.
        
        Returns:
            List of Document objects, one per page. Returns empty list on failure.
        
        Note:
            PyPDFLoader uses mode="single" to create one Document per page, which is
            suitable for large PDFs (e.g., 500+ pages) as it preserves page boundaries.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        source_file = pdf_path.name
        source_file_id = source_file.split("_")[0]
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error("PDF file not found: %s", pdf_path)
            return []

        try:
            loader = PyPDFLoader(str(pdf_path), mode="page")
            documents = loader.load()

            # Enrich metadata for each page document
            timestamp = datetime.now(timezone.utc).isoformat()
            for document in documents:
                document.metadata.update(
                    {
                        "collected_at": timestamp,
                        "source_file": source_file,
                        "source_file_id": source_file_id,
                        "source_type": "pdf",
                    }
                )
            logger.debug(
                "Loaded %d page(s) from %s", document.metadata["total_pages"], source_file
            )

            return documents
        except Exception as exc:
            logger.exception("Failed to load PDF %s: %s", pdf_path, exc)
            return []
    
    def collect(
        self,
        pdf_files: List[str],
        source_dir: Optional[Path | str] = None,
    ) -> Dict[str, List[Document]]:
        """Collect PDF documents from specified paths or directory.
        
        Args:
            pdf_files: List of names of specific PDF files to process.
            source_dir: Directory containing the PDF files.
            output_dir: Directory for saving the collected documents.
        Returns:
            Dictionary of PDF file names and their corresponding list of Document instances.
        """
        source_dir = Path(source_dir) if source_dir else self.source_dir
        results: Dict[str, List[Document]] = {}

        for pdf_file in pdf_files:
            # Ensure pdf_file has .pdf extension
            if not pdf_file.endswith('.pdf'):
                pdf_file = f"{pdf_file}.pdf"

            pdf_path = source_dir / pdf_file

            if not pdf_path.exists():
                logger.warning("Source file not found: %s, skipping", pdf_path)
                results[pdf_file] = []
                continue

            logger.info("Processing source file: %s", pdf_file)

            # Load PDF as documents 
            documents = self.load_pdf_as_documents(pdf_path)
            
            if not documents:
                logger.warning("No documents collected from %s", pdf_path)
                results[pdf_file] = []
                continue
            
            results[pdf_file] = documents
            logger.info("Completed processing %s: collected %d document(s)", pdf_file, len(documents))

        return results
    
    def save(
        self,
        document_list: List[Document],
        output_dir: Path | str,
        output_filename: str,
    ) -> Path:
        """Persist PDF documents as JSONL under output_dir.
        
        Each document is written as a single JSON object per line (JSONL format),
        containing page_content and metadata fields.
        
        Args:
            document_list: List of Document objects to save.
            output_dir: Directory where the JSONL file will be created.
            output_filename: Name of the output JSONL file.
        
        Returns:
            Path to the created JSONL file.
        
        Raises:
            OSError: If the file cannot be written (permissions, disk full, etc.).
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
            logger.exception("Unable to write PDF documents to %s", output_file)
            raise

        return output_file
    
    def collect_and_save(
        self,
        pdf_files: List[str],
        source_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
    ) -> Dict[str, List[Document]]:
        """Collect PDF documents and save them to JSONL.
        
        Convenience method that combines collect() and save().
        
        Args:
            pdf_files: List of specific PDF file names to process.
            source_dir: Directory containing the PDF files.
                If None, uses self.source_dir.
            output_dir: Directory for saving. If None, uses self.output_dir.
        
        Returns:
            Dictionary of PDF file names and their corresponding list of Document instances.
        """
        documents_w_filename = self.collect(pdf_files=pdf_files, source_dir=source_dir)

        if documents_w_filename:
            output_dir = output_dir or self.output_dir
            for source_file, documents in documents_w_filename.items():
                output_filename = source_file.replace('.pdf', '.jsonl')
                self.save(documents, output_dir, output_filename)
        
        return documents_w_filename


def collect_pdf_documents(
    pdf_files: List[str],
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, List[Document]]:
    """Collect PDF documents from specified paths or directory and save to JSONL.
    
    Backward compatibility wrapper for PDFCollector.
    
    Args:
        pdf_files: List of specific PDF file names to process.
        source_dir: Directory containing the PDF files.
        output_dir: Directory where the JSONL export will be written.
    
    Returns:
        Dictionary of PDF file names and their corresponding list of Document instances.
    """
    collector = PDFCollector(source_dir=source_dir, output_dir=output_dir)
    return collector.collect_and_save(pdf_files=pdf_files, source_dir=source_dir, output_dir=output_dir)


# if __name__ == "__main__":
#     collect_pdf_documents(pdf_files=["1_drivers_handbook.pdf"])

# def list_pdf_files(directory: Path | str) -> List[Path]:
#     """Return sorted list of PDF files in the specified directory.
    
#     Backward compatibility wrapper for PDFCollector.list_pdf_files().
#     """
#     return PDFCollector.list_pdf_files(directory)


# def load_pdf_as_documents(pdf_path: Path) -> List[Document]:
#     """Load a PDF file and return LangChain Document instances (one per page).
    
#     Backward compatibility wrapper for PDFCollector.load_pdf_as_documents().
#     """
#     return PDFCollector.load_pdf_as_documents(pdf_path)


# def save_documents(document_list: List[Document], output_dir: Path | str) -> Path:
#     """Persist PDF documents as JSONL under output_dir and return the output path.
    
#     Backward compatibility wrapper for PDFCollector.save().
#     """
#     return PDFCollector.save(document_list, output_dir)

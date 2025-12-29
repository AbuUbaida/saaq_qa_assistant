"""Document embedding generation using HuggingFace Endpoint Embeddings.

Provides Embedder class for generating embeddings with batch processing,
retry logic, and I/O operations.
"""

from __future__ import annotations

import json
import logging
import os
import time
import math
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from ..core.utils import setup_project_path, get_repo_root

setup_project_path()

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import after setup_project_path to avoid circular imports
from ..ingest.text_processor import TextProcessor

REPO_ROOT = get_repo_root()
# DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "processed"
# DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "embeddings"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Generates embeddings for documents using HuggingFace Endpoint Embeddings.
    
    Handles embedding generation, batch processing, retry logic for cold starts,
    and I/O operations for loading/saving embeddings.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        batch_size: int = 32,
        delay_between_batches: float = 1.0,
    ):
        """Initialize embedder.
        
        Args:
            model_name: HuggingFace model identifier.
            api_key: HuggingFace API key. If None, reads from HF_API_KEY env var.
            batch_size: Number of documents to process in each batch.
            delay_between_batches: Seconds to wait between batches.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError(
                "HuggingFace API key missing. Set HF_API_KEY env var or pass api_key."
            )
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self._embedder: Optional[HuggingFaceEndpointEmbeddings] = None
    
    @property
    def hf_embedder(self) -> HuggingFaceEndpointEmbeddings:
        """Get or create HuggingFace embedder instance."""
        if self._embedder is None:
            self._embedder = HuggingFaceEndpointEmbeddings(
                model=self.model_name,
                task="feature-extraction",
                huggingfacehub_api_token=self.api_key,
            )
        return self._embedder
    
    @staticmethod
    def load_documents_for_embedding(jsonl_path: Path | str) -> List[Document]:
        """Load chunked documents from JSONL file for embedding.
        
        Args:
            jsonl_path: Path to JSONL file containing chunked documents.
        
        Returns:
            List of Document objects ready for embedding (filtered to exclude empty).
        """
        documents = TextProcessor.load_documents_from_jsonl(jsonl_path=jsonl_path)
        
        valid_documents = []
        for idx, document in enumerate(documents):
            if not document.page_content or not document.page_content.strip():
                logger.warning("Document #%d has empty page_content. Skipping.", idx + 1)
                continue
            valid_documents.append(document)
        
        logger.info(
            "Loaded %d valid document(s) from %s (filtered %d empty)",
            len(valid_documents),
            jsonl_path,
            len(documents) - len(valid_documents),
        )
        return valid_documents
    
    def embed_texts(self, text_list: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            text_list: List of text strings to embed.
        
        Returns:
            List of embedding vectors (list of floats).
        
        Raises:
            RuntimeError: If API request fails.
        """
        if not text_list:
            logger.warning("Empty text list provided for embedding")
            return []

        try:
            embeddings = self.hf_embedder.embed_documents(text_list)
            logger.debug("Generated %d embedding(s) for batch", len(embeddings))
            return embeddings
        except Exception as e:
            logger.exception("Failed to generate embeddings: %s", e)
            raise RuntimeError(f"HuggingFace embedding API error: {e}") from e
    
    def _embed_batch_with_retry(
        self,
        text_list: List[str],
        is_first_batch: bool = False,
        base_delay_seconds: float = 5.0,
    ) -> List[List[float]]:
        """Embed a batch of texts with retry logic to mitigate HF cold starts.
        
        Args:
            text_list: List of texts to embed.
            is_first_batch: Whether this is the first batch (needs more retries).
            base_delay_seconds: Base delay for exponential backoff.
        
        Returns:
            List of embedding vectors.
        """
        max_retries = 5 if is_first_batch else 3
        for attempt in range(1, max_retries + 1):
            try:
                return self.embed_texts(text_list)
            except RuntimeError as exc:
                if attempt == max_retries:
                    logger.error(
                        "Embedding batch failed after %d attempt(s): %s", attempt, exc
                    )
                    raise

                wait_time = base_delay_seconds * attempt * (2 if is_first_batch else 1)
                logger.warning(
                    "Embedding batch attempt %d/%d failed (%s). Retrying in %.1f seconds...",
                    attempt,
                    max_retries,
                    exc,
                    wait_time,
                )
                time.sleep(wait_time)

        raise RuntimeError("Embedding batch failed after retries")
    
    def embed_documents_batch(
        self,
        documents: List[Document],
        batch_size: Optional[int] = None,
        delay_between_batches: Optional[float] = None,
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of documents.
        
        Args:
            documents: List of Document objects to embed.
            batch_size: Number of documents per batch. If None, uses self.batch_size.
            delay_between_batches: Delay between batches. If None, uses self.delay_between_batches.
        
        Returns:
            List of embedding vectors (None for failed embeddings, maintains order).
        """
        if not documents:
            logger.warning("No documents provided for embedding")
            return []

        batch_size = batch_size or self.batch_size
        delay_between_batches = delay_between_batches or self.delay_between_batches

        all_embeddings: List[Optional[List[float]]] = []
        total = len(documents)
        num_batches = math.ceil(total / batch_size)

        logger.info(
            "Starting batch embedding: %d document(s) in %d batch(es) (size=%d)",
            total,
            num_batches,
            batch_size,
        )

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total)
            document_batch = documents[batch_start:batch_end]

            logger.info(
                "Processing batch %d/%d (documents %d-%d)...",
                batch_idx + 1,
                num_batches,
                batch_start + 1,
                batch_end,
            )

            text_list = [doc.page_content for doc in document_batch]

            try:
                batch_embeddings = self._embed_batch_with_retry(
                    text_list=text_list,
                    is_first_batch=(batch_idx == 0),
                )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(
                    "Failed to embed batch %d/%d after retries: %s. "
                    "Adding None placeholders.",
                    batch_idx + 1,
                    num_batches,
                    e,
                )
                all_embeddings.extend([None] * len(document_batch))

            if batch_idx < num_batches - 1:
                time.sleep(delay_between_batches)

        valid_embeddings = [emb for emb in all_embeddings if emb is not None]
        failed_count = len(all_embeddings) - len(valid_embeddings)

        logger.info(
            "Batch embedding complete: %d/%d successful, %d failed",
            len(valid_embeddings),
            total,
            failed_count,
        )

        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.
        
        Args:
            query: Query text to embed.
        
        Returns:
            Embedding vector as a list of floats.
        """
        return self.hf_embedder.embed_query(query)
    
    def save_embeddings_to_jsonl(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        output_directory: Path | str,
        model_name: str,
        output_filename: str,
    ) -> Path:
        """Save documents with their embeddings to a JSONL file.
        
        Args:
            documents: List of Document objects.
            embeddings: List of embedding vectors (same length as documents).
            model_name: HuggingFace model identifier.
            output_directory: Directory where the JSONL file will be written.
            output_filename: Optional filename for the output file. If None, generates a default name.
        
        Returns:
            Path to the created JSONL file.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings"
            )

        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

        sanitized_model = model_name.replace("/", "_").replace("\\", "_")

        # Ensure filename has .jsonl extension
        if not output_filename.endswith('.jsonl'):
            output_filename = f"{output_filename}.jsonl"
        
        output_filename = output_filename.rsplit('.', 1)[0] + f"[{sanitized_model}].jsonl"
        output_path = output_dir / output_filename
        
        try:
            with output_path.open("w", encoding="utf-8") as handle:
                for doc, embedding in zip(documents, embeddings):
                    doc.metadata["embedding_model"] = model_name
                    record = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": embedding,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            embedding_dim = len(embeddings[0]) if embeddings else 0
            logger.info(
                "Saved %d embedded document(s) (dim=%d) to %s",
                len(documents),
                embedding_dim,
                output_path,
            )
        except Exception:
            logger.exception("Unable to write embedded documents to %s", output_path)
            raise

        return output_path
    
    @staticmethod
    def load_embeddings_from_jsonl(
        jsonl_path: Path | str,
    ) -> Tuple[List[Document], List[List[float]]]:
        """Load documents and their embeddings from a JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file containing documents with embeddings.
        
        Returns:
            Tuple of (documents, embeddings).
        """
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        documents = []
        embeddings = []

        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Invalid JSON at line %d in %s: %s. Skipping.",
                        line_num,
                        path,
                        e,
                    )
                    continue

                if "page_content" not in data:
                    logger.warning(
                        "Missing 'page_content' key at line %d. Skipping.", line_num
                    )
                    continue

                if "embedding" not in data:
                    logger.warning(
                        "Missing 'embedding' key at line %d. Skipping.", line_num
                    )
                    continue

                doc = Document(
                    page_content=data["page_content"],
                    metadata=data.get("metadata", {}),
                )
                documents.append(doc)

                embedding = data["embedding"]
                if not isinstance(embedding, list):
                    logger.warning(
                        "Invalid embedding format at line %d. Expected list. Skipping.",
                        line_num,
                    )
                    documents.pop()
                    continue

                embeddings.append(embedding)

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: loaded {len(documents)} documents but {len(embeddings)} embeddings"
            )

        logger.info(
            "Loaded %d document(s) with embeddings from %s", len(documents), path
        )
        return documents, embeddings
    
    def create_embeddings(
        self,
        input_directory: Path | str,
        input_filenames: List[str],
        output_directory: Path | str,
        batch_size: Optional[int] = None,
        delay_between_batches: Optional[float] = None,
    ) -> List[Path]:
        """Complete pipeline: load documents from multiple JSONL files, generate embeddings, and save to JSONL.
        
        Processes each JSONL file independently and saves output files with the same names
        in the output directory.
        
        Args:
            input_directory: Directory containing JSONL files with chunked documents.
            input_filenames: List of JSONL filenames to process from input_directory.
            output_directory: Directory where documents with embeddings will be saved.
            batch_size: Number of documents per batch. If None, uses self.batch_size.
            delay_between_batches: Delay between batches. If None, uses self.delay_between_batches.
        
        Returns:
            List of paths to the created JSONL files with embeddings.
        
        Raises:
            ValueError: If no input files are provided or no documents are loaded.
            RuntimeError: If all embedding generations fail for a file.
        """
        if not input_filenames:
            raise ValueError("No input filenames provided")
        
        input_dir = Path(input_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting embedding generation pipeline...")
        logger.info("  Input directory: %s", input_dir)
        logger.info("  Input files: %d file(s)", len(input_filenames))
        logger.info("  Output directory: %s", output_dir)
        logger.info("  Model: %s", self.model_name)
        
        output_paths = []
        total_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_documents": 0,
            "total_embeddings": 0,
            "total_failed": 0,
        }
        
        for jsonl_filename in input_filenames:
            # Ensure filename has .jsonl extension
            if not jsonl_filename.endswith('.jsonl'):
                jsonl_filename = f"{jsonl_filename}.jsonl"
            
            input_file = input_dir / jsonl_filename
            
            if not input_file.exists():
                logger.warning("Input file not found: %s, skipping", input_file)
                total_stats["files_failed"] += 1
                continue
            
            logger.info("=" * 60)
            logger.info("Processing file: %s", jsonl_filename)
            
            try:
                # Load documents from this file
                document_list = self.load_documents_for_embedding(input_file)
                
                if not document_list:
                    logger.warning("No documents loaded from %s, skipping", jsonl_filename)
                    total_stats["files_failed"] += 1
                    continue
                
                total_stats["total_documents"] += len(document_list)
                
                # Generate embeddings
                embedding_list = self.embed_documents_batch(
                    documents=document_list,
                    batch_size=batch_size,
                    delay_between_batches=delay_between_batches,
                )
                
                # Filter valid embeddings
                valid_pairs = [
                    (doc, emb) for doc, emb in zip(document_list, embedding_list)
                    if emb is not None
                ]
                
                if not valid_pairs:
                    logger.error("No valid embeddings generated for %s. Skipping.", jsonl_filename)
                    total_stats["files_failed"] += 1
                    total_stats["total_failed"] += len(document_list)
                    continue
                
                valid_documents, valid_embeddings = zip(*valid_pairs)
                total_stats["total_embeddings"] += len(valid_embeddings)
                total_stats["total_failed"] += len(document_list) - len(valid_embeddings)
                
                # Save with the same filename
                output_path = self.save_embeddings_to_jsonl(
                    documents=list(valid_documents),
                    embeddings=list(valid_embeddings),
                    model_name=self.model_name,
                    output_directory=output_dir,
                    output_filename=jsonl_filename,
                )
                
                output_paths.append(output_path)
                total_stats["files_processed"] += 1
                
                embedding_dim = len(valid_embeddings[0]) if valid_embeddings else 0
                logger.info(
                    "Completed %s: %d document(s) -> %d embedding(s) (dim=%d) -> %s",
                    jsonl_filename,
                    len(document_list),
                    len(valid_embeddings),
                    embedding_dim,
                    output_path,
                )
                
            except Exception as e:
                logger.error(
                    "Failed to process %s: %s. Skipping.",
                    jsonl_filename,
                    e,
                    exc_info=True,
                )
                total_stats["files_failed"] += 1
                continue
        
        # Log final statistics
        logger.info("=" * 60)
        logger.info("Embedding pipeline complete!")
        logger.info("  Files processed: %d", total_stats["files_processed"])
        logger.info("  Files failed: %d", total_stats["files_failed"])
        logger.info("  Total documents processed: %d", total_stats["total_documents"])
        logger.info("  Total embeddings generated: %d", total_stats["total_embeddings"])
        logger.info("  Total embeddings failed: %d", total_stats["total_failed"])
        logger.info("  Output directory: %s", output_dir)
        logger.info("=" * 60)
        
        if not output_paths:
            raise RuntimeError("No files were successfully processed")
        
        return output_paths


# def create_embeddings(
#     input_directory: Path | str,
#     output_directory: Path | str,
#     input_filenames: List[str],
#     model_name: str = DEFAULT_EMBEDDING_MODEL,
#     api_key: Optional[str] = None,
#     batch_size: int = 32,
#     delay_between_batches: float = 1.0,
# ) -> List[Path]:
#     """Complete pipeline: load documents, generate embeddings, and save to JSONL.
    
#     Backward compatibility wrapper for Embedder.create_embeddings().
#     """
#     embedder = Embedder(
#         model_name=model_name,
#         api_key=api_key,
#         batch_size=batch_size,
#         delay_between_batches=delay_between_batches,
#     )
#     return embedder.create_embeddings(
#         input_directory=input_directory,
#         output_directory=output_directory,
#         input_filenames=input_filenames,
#         batch_size=batch_size,
#         delay_between_batches=delay_between_batches,
#     )
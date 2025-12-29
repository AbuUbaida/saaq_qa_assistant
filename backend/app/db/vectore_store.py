"""Weaviate vector store operations.

Provides WeaviateStore class for managing Weaviate client connections,
collections, indexing, and search operations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from datetime import datetime, timezone
from weaviate.util import generate_uuid5
import weaviate
from dotenv import load_dotenv
from langchain_core.documents import Document
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.outputs.query import Object

from app.core.utils import setup_project_path
from scripts.embedder import Embedder

setup_project_path()

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Weaviate configuration
DEFAULT_COLLECTION_NAME = "SAAQDocuments"


class WeaviateStore:
    """Manages Weaviate vector store operations.
    
    Handles client connection, collection management, document indexing,
    and search operations (vector, keyword, hybrid).
    """
    
    def __init__(
        self,
        client: Optional[weaviate.WeaviateClient] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_local: bool = True,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        """Initialize Weaviate store.
        
        Args:
            client: Pre-initialized Weaviate client. If None, creates one.
            url: Weaviate instance URL. If None, reads from env vars.
            api_key: Weaviate API key. If None, reads from env vars.
            use_local: If True, connects to local Weaviate instance.
            collection_name: Default collection name for operations.
        """
        self.collection_name = collection_name
        self._client = client
        self._client_owner = False
        
        if client is None:
            self._client = self._create_client(url, api_key, use_local)
            self._client_owner = True
    
    @staticmethod
    def _create_client(
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_local: bool = False,
    ) -> weaviate.WeaviateClient:
        """Create and return a Weaviate client connection.
        
        Args:
            url: Weaviate instance URL. If None, reads from env vars.
            api_key: Weaviate API key. If None, reads from env vars.
            use_local: If True, connects to local Weaviate instance.
        
        Returns:
            Configured Weaviate client instance.
        """
        if use_local:
            client = weaviate.connect_to_local()
        else:
            url = url or os.getenv("WEAVIATE_CLOUD_URL")
            api_key = api_key or os.getenv("WEAVIATE_API_KEY")
            if not url:
                raise ValueError("Weaviate cloud URL is required")
            if not api_key:
                raise ValueError("Weaviate API key is required")
            client = weaviate.connect_to_weaviate_cloud(cluster_url=url, auth_credentials=api_key)

        if not client.is_ready():
            raise RuntimeError("Weaviate client is not ready")
        
        logger.info("Weaviate client connected successfully")
        return client
    
    @property
    def client(self) -> weaviate.WeaviateClient:
        """Get Weaviate client."""
        if self._client is None:
            raise RuntimeError("Weaviate client not initialized")
        return self._client
    
    @staticmethod
    def create_weaviate_client(
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_local: bool = False,
    ) -> weaviate.WeaviateClient:
        """Create and return a Weaviate client connection.
        
        Backward compatibility function.
        """
        return WeaviateStore._create_client(url, api_key, use_local)
    
    def create_collection(
        self,
        collection_name: Optional[str] = None,
    ) -> None:
        """Create a Weaviate collection for storing documents.
        
        Args:
            collection_name: Name of the collection. If None, uses self.collection_name.
        
        Raises:
            RuntimeError: If collection creation fails.
        """
        collection_name = collection_name or self.collection_name
        
        if not self.client.is_ready():
            raise RuntimeError("Weaviate client is not ready")
        
        if self.client.collections.exists(collection_name):
            logger.warning("Collection %s already exists. Skipping creation.", collection_name)
            return
        
        try:
            properties = [
                Property(name="page_content", data_type=DataType.TEXT),
                Property(
                    name="metadata",
                    data_type=DataType.OBJECT,
                    nested_properties=[
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="description", data_type=DataType.TEXT),
                        Property(name="keywords", data_type=DataType.TEXT),
                    ],
                ),
                Property(name="source_url", data_type=DataType.TEXT),
                Property(name="source_file", data_type=DataType.TEXT),
                Property(name="source_file_id", data_type=DataType.INT),
                Property(name="start_index", data_type=DataType.INT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="total_chunks", data_type=DataType.INT),
                Property(name="document_index", data_type=DataType.INT),
                Property(name="page_number", data_type=DataType.INT),
                Property(name="source_type", data_type=DataType.TEXT),
                Property(name="language", data_type=DataType.TEXT),
                Property(name="original_length", data_type=DataType.INT),
                Property(name="cleaned_length", data_type=DataType.INT),
                Property(name="collected_at", data_type=DataType.DATE),
                Property(name="cleaned_at", data_type=DataType.DATE),
                Property(name="embedding_model", data_type=DataType.TEXT),
            ]
            vector_config = Configure.Vectors.self_provided()
            
            collection = self.client.collections.create(
                collection_name,
                properties=properties,
                vector_config=vector_config,
            )

            logger.info("Collection %s created successfully", collection_name)
        except Exception as e:
            logger.exception("Failed to create collection %s: %s", collection_name, e)
            raise RuntimeError(f"Failed to create collection {collection_name}: {e}") from e

    def index_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        collection_name: Optional[str] = None,
        batch_size: int = 200,
    ) -> int:
        """Index documents with their embeddings into Weaviate.
        
        Args:
            documents: List of Document objects to index.
            embeddings: List of embedding vectors (same length as documents).
            collection_name: Name of the collection. If None, uses self.collection_name.
            batch_size: Number of documents to index in each batch.
        
        Returns:
            Number of successfully indexed documents.
        """
        collection_name = collection_name or self.collection_name
        
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings"
            )

        if not self.client.is_ready():
            raise RuntimeError("Weaviate client is not ready")

        try:
            collection = self.client.collections.get(collection_name)
        except Exception as e:
            logger.exception("Failed to get collection %s: %s", collection_name, e)
            raise RuntimeError(f"Collection {collection_name} not found") from e
        
        def safe_int(value, default=0):
            """Safely convert metadata values to int, handling both int and string inputs."""
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        failed_count = 0
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for i, doc in tqdm(enumerate(documents), total=len(documents), desc="Indexing documents"):
                try:
                    start_index = safe_int(doc.metadata.get("start_index"), 0)
                    chunk_index = safe_int(doc.metadata.get("chunk_index"), 0)
                    total_chunks = safe_int(doc.metadata.get("total_chunks"), 0)
                    document_index = safe_int(doc.metadata.get("document_index"), 0)
                    original_length = safe_int(doc.metadata.get("content_length_original") or doc.metadata.get("original_length"), 0)
                    cleaned_length = safe_int(doc.metadata.get("content_length_cleaned") or doc.metadata.get("cleaned_length"), 0)
                    page_number = safe_int(doc.metadata.get("page"), 0) or safe_int(doc.metadata.get("page_number"), 0)
                    source_file_id = safe_int(doc.metadata.get("source_file_id"), 0)
                    
                    collected_at_str = doc.metadata.get("collected_at", "")
                    cleaned_at_str = doc.metadata.get("cleaned_at", "")
                    collected_at = (
                        datetime.fromisoformat(collected_at_str).replace(tzinfo=timezone.utc)
                        if collected_at_str
                        else datetime.now(timezone.utc)
                    )
                    cleaned_at = (
                        datetime.fromisoformat(cleaned_at_str).replace(tzinfo=timezone.utc)
                        if cleaned_at_str
                        else datetime.now(timezone.utc)
                    )
                    
                    metadata = {
                        "title": doc.metadata.get("title"),
                        "description": doc.metadata.get("description"),
                        "keywords": doc.metadata.get("keywords"),
                    }
                    
                    collection_object = {
                        "page_content": doc.page_content,
                        "metadata": metadata,
                        "source_url": doc.metadata.get("source_url", ""),
                        "source_file": doc.metadata.get("source_file", ""),
                        "source_file_id": source_file_id,
                        "start_index": start_index,
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "document_index": document_index,
                        "page_number": page_number,
                        "source_type": doc.metadata.get("source_type", ""),
                        "language": doc.metadata.get("language", ""),
                        "original_length": original_length,
                        "cleaned_length": cleaned_length,
                        "collected_at": collected_at,
                        "cleaned_at": cleaned_at,
                        "embedding_model": doc.metadata.get("embedding_model", ""),
                    }
                    
                    # Generate unique document ID for UUID
                    source_file = doc.metadata.get("source_file", "")
                    document_id = f"{source_file}-{chunk_index}-{document_index}-{source_file_id}"
                    vector = embeddings[i]

                    batch.add_object(
                        properties=collection_object,
                        uuid=generate_uuid5(document_id),
                        vector=vector,
                    )
                except Exception as e:
                    logger.warning("Failed to prepare document %d for indexing: %s", i, e)
                    failed_count += 1
                    continue

        succeeded_count = len(documents) - failed_count
        
        if failed_count > 0:
            logger.warning("Failed to index %d/%d documents", failed_count, len(documents))
        
        logger.info("Successfully indexed %d/%d documents", succeeded_count, len(documents))
        return succeeded_count
    
    def index_from_jsonl_files(
        self,
        source_filenames: List[str],
        source_dir: Path | str,
        collection_name: Optional[str] = None,
        batch_size: int = 200,
    ) -> int:
        """Index documents from one or more JSONL files containing embeddings.
        
        Each file is loaded via ``load_embeddings_from_jsonl`` and all documents
        are indexed into the target Weaviate collection.
        
        Args:
            source_filenames: List of JSONL filenames to index. Filenames may or
                may not include the ``.jsonl`` extension.
            source_dir: Directory containing the source files.
            collection_name: Name of the collection. If None, uses ``self.collection_name``.
            batch_size: Number of documents to index in each batch.
        
        Returns:
            Total number of successfully indexed documents across all files.
        
        Raises:
            ValueError: If no source filenames are provided.
            RuntimeError: If the Weaviate client is not ready.
        """
        if not source_filenames:
            raise ValueError("No source filenames provided for indexing")
        
        collection_name = collection_name or self.collection_name
        source_dir = Path(source_dir)
        
        if not self.client.is_ready():
            raise RuntimeError("Weaviate client is not ready")
        
        succeeded_count = 0
        total_count = 0
        
        logger.info("Starting indexing from JSONL files...")
        logger.info("  Source directory: %s", source_dir)
        logger.info("  Files to index: %d", len(source_filenames))
        logger.info("  Target collection: %s", collection_name)
        
        for source_filename in source_filenames:
            # Ensure .jsonl extension
            if not source_filename.endswith(".jsonl"):
                source_filename = f"{source_filename}.jsonl"
            
            source_path = source_dir / source_filename
            if not source_path.exists():
                logger.warning("Source file not found: %s. Skipping.", source_path)
                continue
            
            logger.info("Indexing from file: %s", source_path)
            
            try:
                documents, embeddings = Embedder.load_embeddings_from_jsonl(source_path)
            except Exception as e:
                logger.error("Failed to load embeddings from %s: %s. Skipping.", source_path, e, exc_info=True)
                continue
            
            if not documents:
                logger.warning("No documents found in %s. Skipping.", source_path)
                continue
            
            try:
                indexed = self.index_documents(
                    documents=documents,
                    embeddings=embeddings,
                    collection_name=collection_name,
                    batch_size=batch_size,
                )
            except Exception as e:
                logger.error("Failed to index documents from %s: %s. Skipping.", source_path, e, exc_info=True)
                continue
            
            succeeded_count += indexed
            total_count += len(documents)
            
            logger.info(
                "Indexed %d/%d document(s) from %s",
                indexed,
                len(documents),
                source_path,
            )
        
        logger.info(
            "Finished indexing from JSONL files: %d/%d document(s) successfully indexed",
            succeeded_count,
            total_count,
        )
        return succeeded_count
    
    def search_by_vector(
        self,
        query_vector: List[float],
        collection_name: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Object]:
        """Perform vector similarity search.
        
        Args:
            query_vector: Query embedding vector (list of floats).
            collection_name: Name of the collection. If None, uses self.collection_name.
            top_k: Number of top results to return.
        
        Returns:
            List of raw Weaviate Object results.
        """
        collection_name = collection_name or self.collection_name
        
        if not self.client.is_ready():
            raise RuntimeError("Weaviate client is not ready")

        try:
            collection = self.client.collections.get(collection_name)
        except Exception as e:
            logger.exception("Failed to get collection %s: %s", collection_name, e)
            raise RuntimeError(f"Collection {collection_name} not found") from e

        try:
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True, certainty=True),
            )
        except Exception as e:
            logger.exception("Failed to search collection %s: %s", collection_name, e)
            raise RuntimeError(f"Vector search failed: {e}") from e

        return response.objects
    
    def search_by_keyword(
        self,
        query_keyword: str,
        collection_name: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Object]:
        """Search for documents by keyword using BM25.
        
        Args:
            query_keyword: Keyword query string.
            collection_name: Name of the collection. If None, uses self.collection_name.
            top_k: Number of top results to return.
        
        Returns:
            List of raw Weaviate Object results.
        """
        collection_name = collection_name or self.collection_name
        
        if not self.client.is_ready():
            raise RuntimeError("Weaviate client is not ready")

        try:
            collection = self.client.collections.get(collection_name)
        except Exception as e:
            logger.exception("Failed to get collection %s: %s", collection_name, e)
            raise RuntimeError(f"Collection {collection_name} not found") from e

        try:
            response = collection.query.bm25(
                query=query_keyword,
                limit=top_k,
                return_metadata=MetadataQuery(score=True, explain_score=True),
            )
        except Exception as e:
            logger.exception("Failed to search collection %s: %s", collection_name, e)
            raise RuntimeError(f"Keyword search failed: {e}") from e

        return response.objects
    
    def hybrid_search(
        self,
        query_keyword: str,
        query_vector: List[float],
        collection_name: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Object]:
        """Perform hybrid search (BM25 + vector).
        
        Args:
            query_keyword: Keyword query string for BM25 search.
            query_vector: Query embedding vector for vector search.
            collection_name: Name of the collection. If None, uses self.collection_name.
            top_k: Number of top results to return.
        
        Returns:
            List of raw Weaviate Object results.
        """
        collection_name = collection_name or self.collection_name
        
        if not self.client.is_ready():
            raise RuntimeError("Weaviate client is not ready")

        try:
            collection = self.client.collections.get(collection_name)
        except Exception as e:
            logger.exception("Failed to get collection %s: %s", collection_name, e)
            raise RuntimeError(f"Collection {collection_name} not found") from e

        try:
            response = collection.query.hybrid(
                query=query_keyword,
                vector=query_vector,
                limit=top_k,
                return_metadata=MetadataQuery(score=True, explain_score=True),
            )
        except Exception as e:
            logger.exception("Failed to search collection %s: %s", collection_name, e)
            raise RuntimeError(f"Hybrid search failed: {e}") from e

        return response.objects
    
    def delete_collection(
        self,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Delete a Weaviate collection.
        
        Args:
            collection_name: Name of the collection. If None, uses self.collection_name.
        
        Returns:
            True if collection was deleted successfully, False otherwise.
        """
        collection_name = collection_name or self.collection_name
        
        if not self.client.is_ready():
            raise RuntimeError("Weaviate client is not ready")
        
        if not self.client.collections.exists(collection_name):
            logger.warning("Collection %s does not exist", collection_name)
            return False
        
        try:
            self.client.collections.delete(collection_name)
            logger.info("Collection %s deleted successfully", collection_name)
        except Exception as e:
            logger.exception("Failed to delete collection %s: %s", collection_name, e)
            raise RuntimeError(f"Failed to delete collection {collection_name}: {e}") from e
        
        return True
    
    def close(self) -> None:
        """Close client connection if we own it."""
        if self._client_owner and self._client is not None:
            try:
                self._client.close()
                logger.info("Closed Weaviate client")
            except Exception as e:
                logger.warning("Error closing Weaviate client: %s", e)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

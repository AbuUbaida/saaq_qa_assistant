"""Delete a collection from the Weaviate vector database.

This script allows you to safely remove collections from Weaviate,
with support for both local and cloud instances.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add repo root to sys.path BEFORE importing backend modules
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.db.vector_store import DEFAULT_COLLECTION_NAME, WeaviateStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def delete_collection(
    collection_name: str,
    use_local_weaviate: bool = True,
    confirm: bool = False,
) -> bool:
    """Delete a collection from Weaviate.
    
    Args:
        collection_name: Name of the collection to delete.
        use_local_weaviate: If True, uses local Weaviate instance.
        confirm: If True, skips confirmation prompt.
    
    Returns:
        True if collection was deleted successfully, False otherwise.
    """
    if not confirm:
        response = input(
            f"Are you sure you want to delete collection '{collection_name}'? "
            f"This action cannot be undone. (yes/no): "
        )
        if response.lower() not in ("yes", "y"):
            logger.info("Deletion cancelled by user")
            return False
    
    try:
        store = WeaviateStore(use_local=use_local_weaviate, collection_name=collection_name)
        deleted = store.delete_collection(collection_name=collection_name)
        
        if deleted:
            logger.info("Successfully deleted collection '%s'", collection_name)
        else:
            logger.warning("Collection '%s' does not exist or could not be deleted", collection_name)
        
        store.close()
        return deleted
    
    except Exception as e:
        logger.error("Failed to delete collection '%s': %s", collection_name, e, exc_info=True)
        return False


def list_collections(use_local_weaviate: bool = True) -> list[str]:
    """List all collections in Weaviate.
    
    Args:
        use_local_weaviate: If True, uses local Weaviate instance.
    
    Returns:
        List of collection names.
    """
    try:
        store = WeaviateStore(use_local=use_local_weaviate)
        
        if not store.client.is_ready():
            logger.error("Weaviate client is not ready")
            store.close()
            return []
        
        try:
            collections = store.client.collections.list_all()
        except AttributeError:
            logger.warning("Could not list collections - API may differ in your Weaviate version")
            collections = []
        
        store.close()
        return collections
    
    except Exception as e:
        logger.error("Failed to list collections: %s", e, exc_info=True)
        return []


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Delete a collection from the Weaviate vector database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete default collection (with confirmation)
  python scripts/delete_collection.py
  
  # Delete specific collection
  python scripts/delete_collection.py --collection MyCollection
  
  # Delete without confirmation prompt
  python scripts/delete_collection.py --collection MyCollection --yes
  
  # Delete from cloud Weaviate
  python scripts/delete_collection.py --collection MyCollection --cloud --yes
  
  # List all collections
  python scripts/delete_collection.py --list
        """,
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Weaviate collection name to delete (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use cloud Weaviate instead of local (env vars WEAVIATE_CLOUD_URL/WEAVIATE_API_KEY required).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        dest="confirm",
        help="Skip confirmation prompt and delete immediately.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available collections and exit.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # List collections if requested
    if args.list:
        logger.info("Listing all collections...")
        collections = list_collections(use_local_weaviate=not args.cloud)
        
        if collections:
            logger.info("Found %d collection(s):", len(collections))
            for name in collections:
                logger.info("  - %s", name)
        else:
            logger.info("No collections found.")
        
        return 0
    
    # Delete collection
    success = delete_collection(
        collection_name=args.collection,
        use_local_weaviate=not args.cloud,
        confirm=args.confirm,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())


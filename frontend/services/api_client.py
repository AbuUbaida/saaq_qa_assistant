"""HTTP client for FastAPI backend communication.

Handles POST requests, error handling, and response parsing.
"""

from typing import Any, Dict, Literal, Optional

import requests

from frontend.config.settings import (
    API_CHAT_ENDPOINT,
    BACKEND_BASE_URL,
    REQUEST_TIMEOUT,
)


class APIError(Exception):
    """Base exception for API client errors."""
    pass


class APIConnectionError(APIError):
    """Raised when unable to connect to the backend."""
    pass


class APITimeoutError(APIError):
    """Raised when the request exceeds the timeout."""
    pass


class APIHTTPError(APIError):
    """Raised when the backend returns a non-200 status code."""
    
    def __init__(self, message: str, status_code: int, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def ask_question(
    question: str,
    search_method: Optional[Literal["vector", "keyword", "hybrid"]] = None,
    top_k: Optional[int] = None,
    language: Optional[Literal["en", "fr"]] = None,
    answer_style: Optional[Literal["concise", "detailed", "formal"]] = None,
    include_citations: Optional[bool] = None,
) -> Dict[str, Any]:
    """Send a question to the backend RAG API and return the response.
    
    Args:
        question: The user's question about SAAQ regulations or procedures.
        search_method: Document retrieval method ('vector', 'keyword', or 'hybrid').
        top_k: Number of top documents to retrieve (1-20).
        language: Response language ('en' or 'fr').
        answer_style: Answer style ('concise', 'detailed', or 'formal').
        include_citations: Whether to include source citations in the answer.
    
    Returns:
        Dictionary containing:
            - answer: Generated answer text
            - sources: List of source document references
            - retrieved_count: Number of documents retrieved
            - latency: Processing time in seconds
    
    Raises:
        APIConnectionError: If unable to connect to the backend.
        APITimeoutError: If the request exceeds the timeout.
        APIHTTPError: If the backend returns a non-200 status code.
    """
    # Build request payload with only provided parameters
    payload: Dict[str, Any] = {"question": question}
    
    if search_method is not None:
        payload["search_method"] = search_method
    if top_k is not None:
        payload["top_k"] = top_k
    if language is not None:
        payload["language"] = language
    if answer_style is not None:
        payload["answer_style"] = answer_style
    if include_citations is not None:
        payload["include_citations"] = include_citations
    
    # Construct full URL
    url = f"{BACKEND_BASE_URL}{API_CHAT_ENDPOINT}"
    
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        
        # Raise exception for non-200 status codes
        response.raise_for_status()
        
        return response.json()
    
    except requests.exceptions.ConnectionError as e:
        raise APIConnectionError(
            f"Unable to connect to backend at {BACKEND_BASE_URL}. "
            "Please ensure the backend service is running."
        ) from e
    
    except requests.exceptions.Timeout as e:
        raise APITimeoutError(
            f"Request to backend exceeded timeout of {REQUEST_TIMEOUT} seconds. "
            "The RAG pipeline may be processing a complex query."
        ) from e
    
    except requests.exceptions.HTTPError as e:
        # Extract error details from response if available
        response_body = None
        if e.response is not None:
            try:
                response_body = e.response.text
            except Exception:
                pass
        
        status_code = e.response.status_code if e.response else 0
        
        raise APIHTTPError(
            f"Backend returned error status {status_code}: {str(e)}",
            status_code=status_code,
            response_body=response_body,
        ) from e

